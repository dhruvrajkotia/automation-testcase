import os
import requests
from typing import Union, List
from bson import ObjectId
from fastapi.concurrency import run_in_threadpool
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_pinecone import PineconeVectorStore
from langchain.output_parsers.json import SimpleJsonOutputParser
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy
from ..schemas import TestCases
from ..constants import SYSTEM_PROMPT_CONVERT_USER_TEXT_TO_TESTCASES, SYSTEM_PROMPT_FOR_STRING_COMPARE
from langserve import RemoteRunnable
import numpy as np
from fastapi import FastAPI
from dotenv import load_dotenv
from datasets import Dataset

load_dotenv()


def sanitize_for_json(data):
    """Recursively sanitize data for JSON serialization."""
    if isinstance(data, dict):
        return {key: sanitize_for_json(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [sanitize_for_json(item) for item in data]
    elif isinstance(data, np.generic):  # Handle NumPy scalar types
        return data.item()
    elif isinstance(data, np.ndarray):  # Convert NumPy arrays to lists
        return data.tolist()
    elif isinstance(data, (np.float_, np.float32, np.float64)):
        return float(data)
    elif isinstance(data, (np.int_, np.int32, np.int64)):
        return int(data)
    elif isinstance(data, (np.bool_, bool)):
        return bool(data)
    else:
        return data  # Return data as-is if it's already JSON-compatible


# Load environment variables once
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
MIDDLEWARE_URL = os.getenv("MIDDLEWARE_URL")
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
RAGAS_API_KEY = os.getenv("RAGAS_API_KEY")

# Initialize reusable objects
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small", openai_api_key=OPENAI_KEY)
llm = ChatOpenAI(model="gpt-4o", model_kwargs={"response_format": {"type": "json_object"}}, temperature=0,
                 max_retries=2, api_key=OPENAI_KEY)
prompt = ChatPromptTemplate.from_messages(
    [("system", "{system}"), ("human", "{input}")])
remote_runnable = RemoteRunnable("https://devbot.indemn.ai/chat")


string_compare_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT_FOR_STRING_COMPARE),
        ("human", "{input}"),
    ]
)


class EvaluateService:
    def __init__(self, payload: dict):
        self.agent_id = payload.get("agent_id")
        self.user_input = payload.get("user_input")

    async def validate_agent_id(self) -> bool:
        from ..main import app
        """Validates the provided agent ID in MongoDB."""
        try:
            db = app.mongodb['faq_kbs']
            agent_exists = await db.find_one({"_id": ObjectId(self.agent_id)})
            if not agent_exists:
                raise ValueError("Invalid agent ID provided.")
            return True
        except Exception as e:
            raise ValueError(f"Error validating agent ID: {str(e)}")

    async def generate_test_cases(self) -> dict:
        input_data = {
            "system": SYSTEM_PROMPT_CONVERT_USER_TEXT_TO_TESTCASES,
            "input": self.user_input
        }
        structured_llm = ChatOpenAI(
            model="gpt-4o").with_structured_output(TestCases)
        few_shot_structured_llm = prompt | structured_llm
        response = await run_in_threadpool(few_shot_structured_llm.invoke, input_data)
        return response.__dict__ if response else {}

    async def start_process(self):
        """Orchestrates validation and LLM invocation."""
        await self.validate_agent_id()
        result = await self.generate_test_cases()
        test_result = await self.evaluate_test_cases(result, self.agent_id)
        sanitized_test_result = sanitize_for_json(test_result)
        print(sanitized_test_result)
        return sanitized_test_result

    async def evaluate_test_cases(self, payload: dict, agent_id: str) -> Union[dict, None]:
        from ..main import app
        """Evaluates test cases using Pinecone VectorStore and RAGAS."""
        try:
            test_results = []
            # Load MongoDB details
            bot_config, bot_details, connected_kbs = await self._fetch_bot_details(
                app, agent_id)
            kbs = [str(doc['id_kb']) for doc in connected_kbs]

            # Initialize VectorStore
            vectorstore = await self._initialize_vectorstore(bot_config)
            retriever = vectorstore.as_retriever(
                search_kwargs={'k': 5, 'filter': {"id_kb": {"$in": kbs}}})

            # Process test cases
            for test_case in payload['test_cases']:
                session_id = await self._create_session(agent_id)
                if session_id:
                    results = await self._evaluate_test_case(
                        test_case, session_id, retriever, bot_details, kbs)
                    test_results.append(results)
            return test_results 

        except Exception as e:
            return {"error": str(e)}

    async def _fetch_bot_details(self, app, agent_id: str):
        cursor_bot_config = app.mongodb['bot_configurations']
        cursor_faq_kbs = app.mongodb['faq_kbs']
        cursor_bot_kb_mappings = app.mongodb['bot_kb_mappings']

        bot_config = await cursor_bot_config.find_one(
            {"bot_id": ObjectId(agent_id)}, {})
        if not bot_config:
            raise ValueError("Bot configuration not found")
        bot_details = await cursor_faq_kbs.find_one({"_id": ObjectId(agent_id)}, {})
        connected_kbs = await cursor_bot_kb_mappings.find(
            {"id_bot": ObjectId(agent_id)}).to_list(None)  # to_list for async cursor

        return bot_config, bot_details, connected_kbs

    async def _initialize_vectorstore(self, bot_config):
        return PineconeVectorStore(
            index_name=bot_config['ai_config']['kb_configuration']['index_name'],
            embedding=embeddings,
            namespace=bot_config['ai_config']['kb_configuration']['namespace'],
            pinecone_api_key=PINECONE_API_KEY
        )

    async def _create_session(self, agent_id: str) -> Union[str, None]:
        """Create a new session with the bot."""
        url = f"{MIDDLEWARE_URL}/conversations"
        payload = {"bot_id": agent_id, "isTestMode": True}
        try:
            response = await run_in_threadpool(requests.post, url, payload)
            if response.status_code == 200 and response.json().get("ok"):
                return response.json().get("data", {}).get("session_id")
            return None
        except Exception:
            return None

    async def _evaluate_test_case(self, test_case: dict, session_id: str, retriever, bot_details: dict, kbs: List[str]):
        """Evaluates a single test case, either FLOW or QnA."""
        if test_case['type'] == "FLOW":
            test_case = await self._evaluate_flow(
                test_case, session_id, bot_details, kbs)
        elif test_case['type'] == "QnA":
            test_case = await self._evaluate_qna(test_case, session_id,
                                                 retriever, bot_details, kbs)
        return test_case

    async def _evaluate_flow(self, test_case: dict, session_id: str, bot_details: dict, kbs: List[str]) -> dict:
        test_case['test_result'] = {"steps": []}
        for step in test_case['steps']:
            result = await self._invoke_step(session_id, step, bot_details, kbs)
            test_case['test_result']['steps'].append(result)
            if not result['matched']:
                test_case['test_result']['result'] = "Failed"
                return test_case
        test_case['test_result']['result'] = "Passed"
        return test_case

    async def _evaluate_qna(self, test_case: dict, session_id: str, retriever, bot_details: dict, kbs: List[str]) -> dict:
        test_case['test_result'] = {"steps": []}
        for step in test_case['steps']:
            result = await self._invoke_qna_step(
                step, retriever, session_id, bot_details, kbs, test_case)
            test_case['test_result']['steps'].append(result)
            if not result['matched']:
                test_case['test_result']['result'] = "Failed"
                return test_case
        test_case['test_result']['result'] = "Passed"
        return test_case

    async def _invoke_step(self, session_id, step, bot_details, kbs):
        """Invokes a single step for FLOW type test cases."""
        user_input, expected_response = step['user_input'], step['expected_response']
        response = await run_in_threadpool(remote_runnable.invoke, {
            "input": user_input,
            "bot_details": {"bot_id": {"_id": self.agent_id, "id_organization": bot_details['id_organization'], "connected_kbs": kbs}},
            "init_parameters": {}, "session_id": session_id
        })
        matched = await self._check_semantic_similarity(expected_response, response)
        return {"step": step['step'], "user_input": user_input, "expected_response": expected_response,
                "bot_response": response, "matched": matched, "result": "Passed" if matched else "Failed"}

    async def _invoke_qna_step(self, step, retriever, session_id, bot_details, kbs, test_case_details):
        """Invokes a single step for QnA type test cases with RAGAS."""
        question, expected_answer = step['user_input'], step['expected_response']
        answer = await run_in_threadpool(remote_runnable.invoke, {
            "input": question,
            "bot_details": {"bot_id": {"_id": self.agent_id, "id_organization": bot_details['id_organization'], "connected_kbs": kbs}},
            "init_parameters": {}, "session_id": session_id
        })
        contexts = [
            doc.page_content for doc in retriever.invoke(question)]
        relevancy_passed, relevancy_score = await self._evaluate_ragas(
            question, answer, contexts, test_case_details['success_criteria']['threshold'])
        return {"question": question, "expected_response": expected_answer, "bot_response": answer, "contexts": contexts,
                "accuracy_score": relevancy_score, "matched": relevancy_passed, "result": "Passed" if relevancy_passed else "Failed"}

    async def _check_semantic_similarity(self, expected_response: str, bot_response: str) -> bool:
        """Checks semantic similarity between expected and actual bot responses."""
        try:
            res = await run_in_threadpool((string_compare_prompt | llm | SimpleJsonOutputParser()).invoke,
                                          {"input": f"expected_response: {expected_response} bot_response: {bot_response}"})
            return res.get('matched', False)
        except Exception:
            return False

    async def _evaluate_ragas(self, question: str, answer: str, contexts: List[str], threshold: float):
        """Evaluates RAGAS metrics for answer relevancy."""
        dataset = Dataset.from_dict({"question": [question], "answer": [
                                    answer], "contexts": [contexts]})
        result = await run_in_threadpool(evaluate, dataset=dataset, metrics=[
                                         faithfulness, answer_relevancy])
        relevancy_score = result.to_pandas()['answer_relevancy'].iloc[0]
        
        '''os.environ["RAGAS_APP_TOKEN"] = RAGAS_API_KEY

        result = result.upload()'''
        return relevancy_score >= threshold, relevancy_score
    
    
    '''async def trigger_service(self):
        result = await self.start_process()
        return result'''