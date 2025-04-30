from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import uuid
from typing import Dict, List, Optional, Any
import os
from pydantic import BaseModel, Field

from context import TestContext
from strategy import SendingPromptsStrategy, CrescendoStrategy

app = FastAPI(title="Pyrit Testing API", description="API for running configurable LLM security tests")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory store for test results (in production, use a persistent store)
test_results = {}

# Pydantic models for request/response
class TestBase(BaseModel):
    test_name: str = Field(..., description="Name of the test")
    user_name: str = Field(..., description="Name of the user running the test")
    memory_db_type: str = Field("InMemory", description="Type of memory database")
    print_results: bool = Field(False, description="Whether to print results")

class SendingPromptsRequest(TestBase):
    dataset: Optional[str] = Field(None, description="Dataset name to load")
    system_prompt: Optional[str] = Field(None, description="System prompt")
    direct_prompts: Optional[List[Dict[str, str]]] = Field(None, description="Direct prompts to send")
    skip_criteria: Optional[Dict[str, Any]] = Field(None, description="Criteria for skipping prompts")
    skip_value_type: Optional[str] = Field("original", description="Type of value to skip")
    converter_configs: Optional[List[Dict[str, Any]]] = Field(None, description="Converter configurations")
    filter_labels: Optional[Dict[str, Any]] = Field(None, description="Labels for filtering results")
    rescore: Optional[bool] = Field(False, description="Whether to rescore results")

class CrescendoRequest(TestBase):
    objectives: List[str] = Field(..., description="Objectives for conversation")
    use_tense_converter: bool = Field(True, description="Whether to use tense converter")
    use_translation_converter: bool = Field(True, description="Whether to use translation converter")
    tense: Optional[str] = Field("past", description="Tense to convert to")
    language: Optional[str] = Field("spanish", description="Language to translate to")
    max_turns: int = Field(10, description="Maximum turns")
    max_backtracks: int = Field(5, description="Maximum backtracks")

class TestResponse(BaseModel):
    test_id: str = Field(..., description="ID of the test")
    status: str = Field(..., description="Status of the test")
    message: str = Field(..., description="Message about the test")

class TestResultResponse(BaseModel):
    test_id: str = Field(..., description="ID of the test")
    status: str = Field(..., description="Status of the test")
    results: List[Dict[str, Any]] = Field(..., description="Results of the test")
    interesting_count: int = Field(0, description="Count of interesting prompts")


# Helper function to create a TestContext
def create_test_context(test_config: TestBase) -> TestContext:
    # For Azure best practices, we're using environment variables for keys and endpoints
    # instead of passing them directly in the request
    return TestContext(
        memory_db_type=test_config.memory_db_type,
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        azure_key=os.getenv("AZURE_OPENAI_KEY"),
        azure_gpt4o_endpoint=os.getenv("AZURE_OPENAI_GPT4O_ENDPOINT"),
        azure_gpt4o_key=os.getenv("AZURE_OPENAI_GPT4O_KEY"),
        default_labels={
            "op_name": f"api_test_{uuid.uuid4().hex[:8]}",
            "user_name": test_config.user_name,
            "test_name": test_config.test_name
        }
    )


# Background task to run a test
async def run_sending_prompts_test(test_id: str, config: SendingPromptsRequest):
    try:
        ctx = create_test_context(config)
        strategy = SendingPromptsStrategy()
        
        # Convert config to params dict
        params = config.dict(exclude={"memory_db_type"})
        
        # Run the test
        results = await strategy(ctx, params)
        
        # Analyze results if requested
        interesting_prompts = []
        if config.filter_labels or config.rescore:
            interesting_prompts = await strategy.analyze_results(
                ctx=ctx,
                params=params,
                _results=results
            )
        
        # Store results
        test_results[test_id] = {
            "status": "completed",
            "results": [r.to_dict() for r in results],
            "interesting_prompts": [p.to_dict() for p in interesting_prompts],
            "interesting_count": len(interesting_prompts)
        }
    except Exception as e:
        test_results[test_id] = {
            "status": "failed",
            "error": str(e)
        }


async def run_crescendo_test(test_id: str, config: CrescendoRequest):
    try:
        ctx = create_test_context(config)
        strategy = CrescendoStrategy()
        
        # Convert config to params dict
        params = config.dict(exclude={"memory_db_type"})
        
        # Run the test
        results = await strategy(ctx, params)
        
        # Store results
        test_results[test_id] = {
            "status": "completed",
            "results": [r.to_dict() for r in results],
            "interesting_count": 0
        }
    except Exception as e:
        test_results[test_id] = {
            "status": "failed",
            "error": str(e)
        }


@app.post("/api/test/sending-prompts", response_model=TestResponse)
async def start_sending_prompts_test(
    config: SendingPromptsRequest,
    background_tasks: BackgroundTasks
):
    test_id = str(uuid.uuid4())
    test_results[test_id] = {"status": "running"}
    
    background_tasks.add_task(run_sending_prompts_test, test_id, config)
    
    return TestResponse(
        test_id=test_id,
        status="running",
        message="Test started"
    )


@app.post("/api/test/crescendo", response_model=TestResponse)
async def start_crescendo_test(
    config: CrescendoRequest,
    background_tasks: BackgroundTasks
):
    test_id = str(uuid.uuid4())
    test_results[test_id] = {"status": "running"}
    
    background_tasks.add_task(run_crescendo_test, test_id, config)
    
    return TestResponse(
        test_id=test_id,
        status="running",
        message="Test started"
    )


@app.get("/api/test/{test_id}", response_model=TestResultResponse)
async def get_test_result(test_id: str):
    if test_id not in test_results:
        raise HTTPException(status_code=404, detail="Test not found")
    
    result = test_results[test_id]
    
    if result["status"] == "running":
        return TestResultResponse(
            test_id=test_id,
            status="running",
            results=[],
            interesting_count=0
        )
    elif result["status"] == "failed":
        raise HTTPException(status_code=500, detail=result["error"])
    
    return TestResultResponse(
        test_id=test_id,
        status="completed",
        results=result["results"],
        interesting_count=result.get("interesting_count", 0)
    )


@app.get("/api/test/{test_id}/interesting", response_model=TestResultResponse)
async def get_interesting_prompts(test_id: str):
    if test_id not in test_results:
        raise HTTPException(status_code=404, detail="Test not found")
    
    result = test_results[test_id]
    
    if result["status"] == "running":
        return TestResultResponse(
            test_id=test_id,
            status="running",
            results=[],
            interesting_count=0
        )
    elif result["status"] == "failed":
        raise HTTPException(status_code=500, detail=result["error"])
    
    return TestResultResponse(
        test_id=test_id,
        status="completed",
        results=result.get("interesting_prompts", []),
        interesting_count=result.get("interesting_count", 0)
    )


@app.get("/api/tests", response_model=Dict[str, str])
async def list_tests():
    return {
        test_id: result["status"] 
        for test_id, result in test_results.items()
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
