from pydantic import BaseModel, Field
from typing_extensions import TypedDict
from typing import List, Union
from pydantic import BaseModel, Field, StrictStr, StrictFloat


class Evaluate(BaseModel):
    user_input: str
    agent_id: str



class EvaluateResponse(BaseModel):
    user_input: str
    agent_id: str


class Step(TypedDict):
    """Defines a step in a test case."""
    step: str
    user_input: str
    expected_response: str


class SuccessCriteria(TypedDict):
    """Defines the success criteria for a test case."""
    threshold: float


class TestCase(TypedDict):
    """Defines a test case."""
    description: str
    type: str
    steps: List[Step]
    success_criteria: Union[str, SuccessCriteria]
    failure_criteria: str


class TestCases(BaseModel):
    """Defines the structure for multiple test cases."""
    test_cases: List[TestCase] = Field(
        description="A list of test cases, each containing details like description, type, steps, success criteria, and failure criteria."
    )
