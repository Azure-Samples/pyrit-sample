import os
import pathlib
import asyncio
from typing import Any, Optional

from pyrit.common.initialization import initialize_pyrit
from pyrit.common.path import DATASETS_PATH
from pyrit.memory.central_memory import CentralMemory
from pyrit.prompt_target import OpenAIChatTarget
from pyrit.score import AzureContentFilterScorer, SelfAskRefusalScorer, LikertScalePaths, SelfAskLikertScorer
from pyrit.models.seed_prompt import SeedPromptDataset
from pyrit.models.seed_prompt import SeedPrompt, SeedPromptGroup


class TestContext:
    """
    Owns all shared configuration: memory init, dataset paths,
    default labels, endpoints, converters, scorers, etc.
    """

    def __init__(self,
                 memory_db_type: str = "InMemory",
                 azure_endpoint: Optional[str] = None,
                 azure_key: Optional[str] = None,
                 azure_gpt4o_endpoint: Optional[str] = None,
                 azure_gpt4o_key: Optional[str] = None,
                 default_labels: Optional[dict[str, Any]] = None):
        # initialize storage/memory
        initialize_pyrit(memory_db_type=memory_db_type)
        self.memory = CentralMemory.get_memory_instance()

        self.labels = default_labels or {}

        self.target = OpenAIChatTarget(
            endpoint=azure_endpoint or os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=azure_key or os.getenv("AZURE_OPENAI_KEY")
        )

        self.gpt4o_target = OpenAIChatTarget(
            endpoint=azure_gpt4o_endpoint or os.getenv("AZURE_OPENAI_GPT4O_ENDPOINT"),
            api_key=azure_gpt4o_key or os.getenv("AZURE_OPENAI_GPT4O_KEY")
        )

        self.scorers = [
            AzureContentFilterScorer(),
            SelfAskRefusalScorer(chat_target=self.target)
        ]

        self.likert_scorer = SelfAskLikertScorer(
            likert_scale_path=LikertScalePaths.HARM_SCALE.value, 
            chat_target=self.target
        )

    def load_seed_prompts(self, dataset_name: str, added_by: str = "automated"):
        """
        Load a YAML dataset into memory (once).
        """
        ds_path = pathlib.Path(DATASETS_PATH) / "seed_prompts" / f"{dataset_name}.prompt"
        dataset = SeedPromptDataset.from_yaml_file(ds_path)
        asyncio.run(
            self.memory.add_seed_prompts_to_memory_async(
                prompts=dataset.prompts,
                added_by=added_by
            )
        )

    def get_prompt_groups(self, dataset_name: Optional[str] = None):
        """
        Retrieve seed prompt groups, optionally filtered by dataset_name.
        """
        return self.memory.get_seed_prompt_groups(dataset_name=dataset_name)

    def default_labels(self) -> dict[str, Any]:
        """
        Return a copy of the default labels for this context.
        This is useful for ensuring that the labels are not modified outside of this class.
        """

        return dict(self.labels)  # defensive copy

    def create_prompt_group(self, prompts: list[dict[str, str]]) -> SeedPromptGroup:
        """
        Create a SeedPromptGroup from a list of prompt dictionaries.
        Each dictionary should have 'value' and 'data_type' keys.
        """
        return SeedPromptGroup(
            prompts=[
                SeedPrompt(value=p["value"], data_type=p.get("data_type", "text"))
                for p in prompts
            ]
        )
    
    def get_interesting_prompts(self, filter_labels: Optional[dict[str, Any]] = None) -> list[Any]:
        """
        Return prompts with interesting scores (e.g., high harm scores or refusal failures).
        """
        labels = filter_labels or self.default_labels()
        result_pieces = self.memory.get_prompt_request_pieces(labels=labels)
        
        interesting_prompts = []
        for piece in result_pieces:
            for score in piece.scores:
                if ((score.score_type == "float_scale" and score.get_value() > 0) or
                    (score.scorer_class_identifier["__type__"] == "SelfAskRefusalScorer" and 
                     score.get_value() == False)):
                    interesting_prompts.append(piece)
                    break
                    
        return interesting_prompts
    
    async def rescore_prompts(self, prompts: list[Any], scorer=None) -> list[Any]:
        """
        Apply additional scoring to a list of prompts.
        """
        if scorer is None:
            scorer = self.likert_scorer
            
        return await scorer.score_responses_inferring_tasks_batch_async(
            request_responses=prompts
        )
