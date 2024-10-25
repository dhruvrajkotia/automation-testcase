import os
import streamlit as st
from typing import List, Union
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from typing_extensions import TypedDict
from langchain_core.prompts import ChatPromptTemplate
from evaluate import evaluate_test_cases, is_valid_agent_id


# Load environment variables
load_dotenv()
OPENAI_KEY = os.getenv("OPENAI_API_KEY")

# Define the TypedDict for the new JSON format


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


class TestCases(TypedDict):
    """Defines the structure for test cases."""
    test_cases: List[TestCase]


# Define the system prompt
system_prompt = """You are an intelligent JSON creation expert who is responsible for testing usecases of chatbot, skilled in intelligently converting user input into structured JSON formats following a predefined schema. Your primary role is to interpret user queries and provide structured JSON outputs based on the context of the conversation.

**Primary Objective:**
Your goal is to interact with the user, interpret their input, correctly identify which are the test cases from the user input itself and generate structured JSON outputs of those test cases in real-time based on the schema and examples provided. You must extract test cases from user inputs efficiently, intelligently and respond with accurate, context-aware JSON formats.

There can be multiple test cases in a single user input, so identify them and create those based on the json schema.

### **Operational Guidelines:**
1. **Functionality:**
   - Your task is to "extract test cases" from the user input and then convert those "test cases" into structured JSON according to specific test case scenarios. Focus on accurately capturing both the **intent** and the **details** provided by the user.
   - Ignore any conversational text that is not part of the test cases.

2. **Identifying Test Cases:**
   - Each test case includes a description, type, steps (with step description, user input, and expected response), success criteria (with step threshold), and failure criteria.

3. **Response Format:**
   - Structure each response in the following format:
     - **description**: A brief description of the test case.
     - **type**: A label indicating the category of the test case, it should be either "FLOW" or "QnA".
     - **steps**: A list of steps, where each step contains:
       - **step**: The description of the step.
       - **user_input**: The user's input for the step.
       - **expected_response**: The expected response for the step.
     - **success_criteria**: This contains below value and it is a compulsory field.
         - There is one condition: If the user input type is 'FLOW' do not provide threshold key in the JSON like provided in the example below.
         - **threshold**: The default value is 0.8 unless it is specified otherwise in the user input.
     - **failure_criteria**: The criteria for considering the test case a failure.

4. **Example User Conversations:**
   - **Example 1:**
     - **User:** "so first ask the connect me to agent then bot must provide with an question related to the email. provide email as a dhruv@indemn.ai then say additional notes as null Success: If the API success and get the response like your request is pass to the human Failure: if something wrong. After that Ask agent what is indemn?The bot will provide the answer from the KNOWLEDGE BASE."
     - **Response:** `{
        {
        "test_cases": [
        {
            "description": "Human handoff flow test",
            "type": "FLOW",
            "steps": [
                {
                    "step": "Request Agent Connection",
                    "user_input": "Connect me to an agent",
                    "expected_response": "Bot prompts for email"
                },
                {
                    "step": "Provide Email",
                    "user_input": "dhruv@indemn.ai",
                    "expected_response": "Bot asks for additional notes"
                },
                {
                    "step": "Provide Additional Notes",
                    "user_input": "null",
                    "expected_response": "API response indicates request passed to human"
                }
            ],
        "success_criteria": "API returns 'your request is passed to the human", // this can be based on the test case. so do not make it default for all the flow test cases. create based on the user description
        "failure_criteria": "Any deviation from expected responses or API error" //this can be based on the test case. so do not make it default for all the flow test cases. create based on the user description
        },
        {
            "description": "Checking the KB Answer",
            "type": "QnA",
            "steps": [
                {
                    "step": "Ask about Indemn",
                    "user_input": "What is indemn?",
                    "expected_response": "Indemn is an insurance company",
                }
            ],
            "success_criteria": {
                "threshold": 0.8
            },
            "failure_criteria": "Response does not match expected answer or answer relevancy metric is below threshold"
        }
    ]
}`

### **Best Practices:**
- **Single Focus:** Focus on processing one user request at a time, converting it into structured JSON format.
- **No Assumptions:** Always extract information directly from user inputs. Do not assume or add details not explicitly mentioned by the user.
- **Context Awareness:** Ensure your responses are coherent and based on the conversation's context.
- **Consistency:** Follow the schema and examples consistently, ensuring each output aligns with the predefined format.
"""

# Create the prompt template
prompt = ChatPromptTemplate.from_messages(
    [("system", "{system}"), ("human", "{input}")])

# Initialize Streamlit app
st.title("IndemnTestGen: AI-Driven Test Automation Excellence")

# Ask for OpenAI API key and agent ID
# openai_api_key = st.text_input("Enter OpenAI API Key", type="password")
agent_id = st.text_input("Agent ID")

if 'result' not in st.session_state:
    st.session_state.result = None

if agent_id:
    if not is_valid_agent_id(agent_id):
        st.error("Invalid Agent ID. Please enter a valid Agent ID.")
    # Initialize the language model with the provided API key
    else:
        os.environ['OPENAI_API_KEY'] = OPENAI_KEY
        llm = ChatOpenAI(model="gpt-4o")
        user_input = st.text_area("Input Test Cases", height=400)
        button = st.button("Run Test Case Evaluation",
                           disabled=not user_input)

        if button and user_input:
            if user_input:
                input_data = {
                    "system": system_prompt,
                    "input": user_input
                }
                structured_llm = llm.with_structured_output(TestCases)
                few_shot_structured_llm = prompt | structured_llm
                response = few_shot_structured_llm.invoke(input_data)
                result = evaluate_test_cases(response, agent_id)
                st.session_state.result = result


if st.session_state.result:
    result = st.session_state.result

    st.subheader("Test Case Results")
    index = 1
    for test_case in result["test_cases"]:
        with st.expander(f"Test Case {index} - {test_case['description']}",
                         icon="✅" if test_case['test_result']['result'] == 'Passed' else '❌'):
            st.json(test_case['test_result']['steps'])
        index += 1

    st.subheader("Detailed Process Log")
    with st.expander("Click here"):
        st.json(result)
