import os
from typing import Union
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import OpenAIEmbeddings
from ..schemas import TestCases
from bson import ObjectId
from ..constants import SYSTEM_PROMPT_CONVERT_USER_TEXT_TO_TESTCASES
from fastapi import HTTPException


OPENAI_KEY = os.getenv("OPENAI_API_KEY")

# Initialize embeddings
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    openai_api_key=OPENAI_KEY
)

llm = ChatOpenAI(
    model="gpt-4o",
    model_kwargs={"response_format": {"type": "json_object"}},
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    api_key=OPENAI_KEY
)

prompt = ChatPromptTemplate.from_messages(
    [("system", "{system}"), ("human", "{input}")])


class EvaluateService:
    def __init__(self, payload: dict):
        self.agent_id = payload.get("agent_id")
        self.testcases = payload.get("testcases")

    async def validate_agent_id(self) -> bool:
        from ..main import app
        """
        Validates the provided agent ID from the MongoDB collection.
        """
        try:
            # Access the MongoDB connection
            db = app.mongodb['faq_kbs']

            # Find the agent ID in the collection
            agent_exists = await db.find_one({"_id": ObjectId(self.agent_id)})

            if not agent_exists:
                raise ValueError("Invalid agent ID provided.")
            return True

        except Exception as e:
            raise ValueError(f"Error validating agent ID: {str(e)}")

    def start_process(self) -> dict:
        """
        Orchestrates the overall process of validation and LLM invocation.
        """
        input_data = {
            "system": SYSTEM_PROMPT_CONVERT_USER_TEXT_TO_TESTCASES,
            "input": self.testcases
        }

        # Set up structured LLM with the strict Pydantic TestCases model
        structured_llm = llm.with_structured_output(TestCases)
        few_shot_structured_llm = prompt | structured_llm

        try:
            # Invoke the LLM with the input data
            response = few_shot_structured_llm.invoke(input_data)
            print(response)
            return response.dict()

        except ValueError as ve:
            raise HTTPException(status_code=422, detail=str(ve))
        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"Error processing test cases: {str(e)}")
    # def process_test_case(self, user_input: str) -> Union[dict, None]:
    #     """
    #     Generates structured JSON output from user input using the LLM.
    #     """
    #     structured_llm = self.llm.with_structured_output(TestCases)
    #     few_shot_structured_llm = prompt | structured_llm

    #     input_data = self.start_process(user_input)
    #     response = few_shot_structured_llm.invoke(input_data)

    #     return response

    # def evaluate_test_case_results(self, response: dict, agent_id: str) -> dict:
    #     """
    #     Evaluates the test case results against the agent.
    #     """
    #     return evaluate_test_cases(response, agent_id)
