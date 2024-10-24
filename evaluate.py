import os
import requests
import json
from bson import ObjectId
from dotenv import load_dotenv
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy
from datasets import Dataset
from langchain_openai import OpenAIEmbeddings
from langserve import RemoteRunnable
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.output_parsers.json import SimpleJsonOutputParser
from pymongo import MongoClient


load_dotenv()

class MongoDBConnection:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(MongoDBConnection, cls).__new__(cls)
            cls._instance.connection = MongoClient(os.environ.get(
                'MONGO_URL', 'mongodb://localhost:27017/'), tls=True, tlsAllowInvalidCertificates=True)
        return cls._instance

    def get_connection(self):
        return self.connection

def evaluate_test_cases(payload: dict, id):
    # Load environment variables
    

    # Ask for agent ID at the start
    agent_id = id  # Use agent_id from the payload

    # Initialize embeddings
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=os.environ.get("OPENAI_API_KEY")
    )

    llm = ChatOpenAI(
        model="gpt-4o",
        model_kwargs={"response_format": {"type": "json_object"}},
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2
    )

    SYSTEM_PROMPT = """
    Determine if two strings or sets of instructions have the same meaning, and return `true` if they are equivalent, or `false` if they are not.

    one is the expected_response and another one is the bot_response. 

    expected_response might be the instructions while the bot_response is the actual response.

    # Steps

    1. **Normalize the Input**: 
       - Convert both strings to a consistent format by changing to lowercase and removing punctuation.
    2. **Semantic Analysis**: 
       - Analyze the semantics of both strings to understand their underlying meaning or instructions.
    3. **Comparison**: 
       - Compare the meanings obtained from the semantic analysis to check for equivalence.
    4. **Conclusion**: 
       - Determine if the meanings are the same and return the appropriate boolean value.

    # Output Format

    - Return a boolean value: `true` if the strings have the same meaning, `false` otherwise. below is the JSON format

    "matched": true/false
    """

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            ("human", "{input}"),
        ]
    )

    # Establish MongoDB connection
    mongo_connection = MongoDBConnection()

    # Load MongoDB cursors
    cursor_bot_config = mongo_connection.get_connection().tiledesk['bot_configurations']
    cursor_faq_kbs = mongo_connection.get_connection().tiledesk['faq_kbs']
    cursor_bot_kb_mappings = mongo_connection.get_connection().tiledesk['bot_kb_mappings']

    # Fetch bot and KB details from MongoDB
    bot_config_details = cursor_bot_config.find_one({"bot_id": ObjectId(agent_id)}, {})
    bot_details = cursor_faq_kbs.find_one({"_id": ObjectId(agent_id)}, {})
    connected_kbs = cursor_bot_kb_mappings.find({"id_bot": ObjectId(agent_id)})
    kb_response = list(connected_kbs)
    kbs = [str(doc['id_kb']) for doc in kb_response]

    # Initialize Pinecone and VectorStore
    pinecone = Pinecone()
    vectorstore = PineconeVectorStore(
        index_name=bot_config_details['ai_config']['kb_configuration']['index_name'],
        embedding=embeddings,
        namespace=bot_config_details['ai_config']['kb_configuration']['namespace'],
        pinecone_api_key=os.environ.get("PINECONE_API_KEY")
    )
    retriever = vectorstore.as_retriever(
        search_kwargs={'k': 5, 'filter': {"id_kb": {"$in": kbs}}}
    )

    # Initialize the remote runnable
    remote_runnable = RemoteRunnable("https://devbot.indemn.ai/chat")

    # Function to check semantic similarity
    def check_semantic_similarity(expected_response, bot_response):
        try:
            chain = prompt | llm | SimpleJsonOutputParser()
            res = chain.invoke({
                "input": f"expected_response: {expected_response} bot_response: {bot_response}"
            })
            return res.get('matched', False)
        except Exception as e:
            return False

    # Function to evaluate RAGAS for QnA type
    def evaluate_ragas(question, answer, contexts, threshold):
        data = {
            "question": [question],
            "answer": [answer],
            "contexts": [contexts]
        }
        dataset = Dataset.from_dict(data)

        result = evaluate(
            dataset=dataset,
            metrics=[faithfulness, answer_relevancy]
        )

        df = result.to_pandas()
        relevancy_score = df['answer_relevancy'].iloc[0]
        return relevancy_score >= threshold, relevancy_score

    # Function to create a new session
    def create_session():
        url = f"{os.environ.get('MIDDLEWARE_URL')}/conversations"
        payload = {
            "bot_id": agent_id,
            "isTestMode": True
        }

        try:
            response = requests.post(url, json=payload)
            if response.status_code == 200 and response.json().get("ok"):
                return response.json().get("data", {}).get("session_id")
            else:
                return None
        except Exception as e:
            return None

    # Function to evaluate FLOW type test cases
    def evaluate_flow(test_case, session_id):
        test_case['test_result'] = {"steps": []}

        for step in test_case['steps']:
            user_input = step['user_input']
            expected_response = step['expected_response']

            response = remote_runnable.invoke({
                "input": user_input,
                "bot_details": {
                    "bot_id": {
                        "_id": agent_id,
                        "id_organization": bot_details['id_organization'],
                        "connected_kbs": kbs
                    },
                },
                "init_parameters": {},
                "session_id": session_id,
            })

            step_result = {
                "step": step['step'],
                "user_input": user_input,
                "expected_response": expected_response,
                "bot_response": response,
                "matched": check_semantic_similarity(expected_response, response)
            }

            if not step_result['matched']:
                test_case['test_result']['steps'].append({
                    "result": "Failed",
                    **step_result
                })
                test_case['test_result']['result'] = "Failed"
                return False

            test_case['test_result']['steps'].append({
                "result": "Passed",
                **step_result
            })

        test_case['test_result']['result'] = "Passed"
        return True

    # Function to evaluate QnA type test cases
    def evaluate_qna(test_case, session_id):
        test_case['test_result'] = {"steps": []}

        for step in test_case['steps']:
            question = step['user_input']
            expected_answer = step['expected_response']

            answer = remote_runnable.invoke({
                "input": question,
                "bot_details": {
                    "bot_id": {
                        "_id": agent_id,
                        "id_organization": bot_details['id_organization'],
                        "connected_kbs": kbs
                    },
                },
                "init_parameters": {},
                "session_id": session_id,
            })

            contexts = [doc.page_content for doc in retriever.get_relevant_documents(question)]
            threshold = test_case['success_criteria']['threshold']
            relevancy_passed, relevancy_score = evaluate_ragas(question, answer, contexts, threshold)

            step_result = {
                "question": question,
                "expected_response": expected_answer,
                "bot_response": answer,
                "contexts": contexts,
                "accuracy_score": relevancy_score,
                "matched": relevancy_passed and (expected_answer == answer)
            }

            if not step_result['matched']:
                test_case['test_result']['steps'].append({
                    "result": "Failed",
                    **step_result
                })
                test_case['test_result']['result'] = "Failed"
                return False

            test_case['test_result']['steps'].append({
                "result": "Passed",
                **step_result
            })

        test_case['test_result']['result'] = "Passed"
        return True

    # Loop through the test cases in the payload
    for test_case in payload['test_cases']:
        session_id = create_session()
        if not session_id:
            continue

        if test_case['type'] == "FLOW":
            evaluate_flow(test_case, session_id)
        elif test_case['type'] == "QnA":
            evaluate_qna(test_case, session_id)

    return payload  # Return the updated payload with test results

'''json_inp = {
    "test_cases": [
        {
            "description": "Greeting interaction test",
            "type": "QnA",
            "steps": [
                {
                    "step": "Greet the bot",
                    "user_input": "HI",
                    "expected_response": "Hello, How can I help you"
                }
            ],
            "success_criteria": {
                "threshold": 0.8
            },
            "failure_criteria": "Bot does not respond with a greeting message"
        },
        {
            "description": "Knowledge base answer check for 'indemn'",
            "type": "QnA",
            "steps": [
                {
                    "step": "Ask about Indemn",
                    "user_input": "What is indemn",
                    "expected_response": "Indemn is an insurance company."
                }
            ],
            "success_criteria": {
                "threshold": 0.8
            },
            "failure_criteria": "Response does not match expected answer or answer relevancy metric is below threshold"
        },
        {
            "description": "Human handoff feature test",
            "type": "FLOW",
            "steps": [
                {
                    "step": "Test human handoff",
                    "user_input": "Trigger human handoff",
                    "expected_response": "Bot initiates human handoff process"
                }
            ],
            "success_criteria": {
                "threshold": 0.8
            },
            "failure_criteria": "Handoff process does not initiate as expected"
        }
    ]
}'''
#print(evaluate_test_cases(json_inp, '66f650064afb0c0013611e70'))
