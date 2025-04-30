from abc import ABC, abstractmethod
from typing import Any, Dict, List

from pyrit.models.prompt_request_response import PromptRequestResponse
from pyrit.models.prompt_request_piece import PromptRequestPiece
from pyrit.prompt_converter.charswap_attack_converter import CharSwapGenerator
from pyrit.prompt_normalizer.normalizer_request import NormalizerRequest
from pyrit.prompt_normalizer.prompt_converter_configuration import PromptConverterConfiguration
from pyrit.orchestrator import PromptSendingOrchestrator, CrescendoOrchestrator
from pyrit.prompt_converter import TenseConverter, TranslationConverter
from pyrit.models.filter_criteria import PromptFilterCriteria

from context import TestContext


class TestStrategy(ABC):

    @abstractmethod
    async def __call__(self, ctx: TestContext, params: Dict[str, Any]) -> Any:
        """
        Run the test given a shared TestContext and strategy-specific params.
        """

    async def print_results(self, results: List[Any]) -> None:
        """Print the conversation results."""
        for result in results:
            await result.print_conversation_async()


class SendingPromptsStrategy(TestStrategy):
    async def __call__(self, ctx: TestContext, params: Dict[str, Any]) -> List:
        # 1) optionally load seed prompts
        dataset = params.get("dataset")
        if dataset:
            ctx.load_seed_prompts(dataset_name=dataset, added_by=params.get("user", "auto"))

        # 2) get prompt groups
        groups = ctx.get_prompt_groups(dataset_name=dataset)

        # 3) build and configure orchestrator
        orchestrator = PromptSendingOrchestrator(
            objective_target=ctx.target,
            scorers=ctx.scorers
        )

        if params.get("skip_criteria"):
            skip_criteria = PromptFilterCriteria(**params.get("skip_criteria"))
            orchestrator.set_skip_criteria(
                skip_criteria=skip_criteria, 
                skip_value_type=params.get("skip_value_type", "original")
            )

        # 4) optional SYSTEM preamble
        sys_text = params.get("system_prompt")
        if sys_text:
            orchestrator.set_prepended_conversation(prepended_conversation=[
                PromptRequestResponse(
                    request_pieces=[PromptRequestPiece(original_value=sys_text, role="system")]
                )
            ])

        # 5) create normalizer requests (with CharSwap attack example)
        requests = []

        if params.get("direct_prompts"):
            for prompt_data in params["direct_prompts"]:
                prompt_group = ctx.create_prompt_group([prompt_data])
                requests.append(
                    NormalizerRequest(
                        seed_prompt_group=prompt_group,
                        request_converter_configurations=params.get("converter_configs", [])
                    )
                )

        for group in groups:
            requests.append(
                NormalizerRequest(
                    seed_prompt_group=group,
                    request_converter_configurations=[
                        PromptConverterConfiguration(
                            converters=[CharSwapGenerator()],
                            prompt_data_types_to_apply=["text"]
                        )
                    ],
                    response_converter_configurations=[]
                )
            )

        # 6) send them
        results = await orchestrator.send_normalizer_requests_async(
            prompt_request_list=requests,
            memory_labels=ctx.default_labels()
        )
        
        # 7) print results if requested
        if params.get("print_results", False):
            await orchestrator.print_conversations_async()
            
        return results

    async def analyze_results(
            self,
            ctx: TestContext,
            params: Dict[str, Any],
            _results: List[Any]
        ) -> List[Any]:
        """
        Analyze results and optionally rescore interesting prompts.
        """
        interesting_prompts = ctx.get_interesting_prompts(params.get("filter_labels"))
        
        if params.get("rescore", False) and interesting_prompts:
            return await ctx.rescore_prompts(interesting_prompts)
        
        return interesting_prompts


class CrescendoStrategy(TestStrategy):
    async def __call__(self, ctx: TestContext, params: Dict[str, Any]) -> List:
        # 1) prepare converters
        converter_target = params.get("converter_target", ctx.platform_target)
        
        converters = []
        if params.get("use_tense_converter", True):
            converters.append(TenseConverter(
                converter_target=converter_target, 
                tense=params.get("tense", "past")
            ))
            
        if params.get("use_translation_converter", True):
            converters.append(TranslationConverter(
                converter_target=converter_target, 
                language=params.get("language", "spanish")
            ))
        
        # Allow custom converters
        if params.get("custom_converters"):
            converters.extend(params["custom_converters"])

        # 2) configure orchestrator with more flexible target options
        target = params.get("target", ctx.target)
        adv_target = params.get("adversarial_chat", ctx.gpt4o_target)
        scoring_target = params.get("scoring_target", ctx.target)
        
        orchestrator = CrescendoOrchestrator(
            objective_target=target,
            adversarial_chat=adv_target,
            max_turns=params.get("max_turns", 10),
            max_backtracks=params.get("max_backtracks", 5),
            scoring_target=scoring_target,
            prompt_converters=converters
        )

        # 3) run the attack/objectives
        results = await orchestrator.run_attacks_async(
            objectives=params.get("objectives", []),
            memory_labels=ctx.default_labels()
        )
        
        # 4) print results if requested
        if params.get("print_results", True):
            await self.print_results(results)
            
        return results
