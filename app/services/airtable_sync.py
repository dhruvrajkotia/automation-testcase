import os
from pyairtable import Api
api = Api(os.environ['AIRTABLE_API_KEY'])
table = api.table('appUDawuQbkse8NH0', 'tbly2kI0TujdTQhUH')

'''def push_airtable(input_data, agent_id):
    output_records = []

    # Loop through each description in the input data
    for description_data in input_data:
        description = description_data.get("description")
        steps = description_data.get("steps", [])
        
        # Loop through each step of the description
        for step_data in steps:
            step = step_data.get("step", "")
            user_input = step_data.get("user_input", "")
            expected_response = step_data.get("expected_response", "")
            bot_response = step_data.get("bot_response", "")  # Assuming you get the bot response from test results
            result = "Failed"  # Assuming you get the result from test results
            accuracy_score = 0.0  # Assuming you get accuracy score from test results

            # Generate the output record for this specific step
            output_record = {
                    "agent_id": agent_id,
                    "Testcase": description,
                    "TestCase Type": description_data.get("type", ""),
                    "Step": step,
                    "User Input": user_input,
                    "Expected Response": expected_response,
                    "Bot Response": bot_response,
                    "Result": result,
                    "Accuracy Score": accuracy_score
                }
            
            # Add the generated record to the output list
            output_records.append(output_record)

    for record in output_records:
        print('Record : ',record)
        table.create(record)
        print('\n')'''
def push_airtable(input_data, agent_id):
    output_records = []

    # Loop through each description in the input data
    for description_data in input_data:
        description = description_data.get("description")
        steps = description_data.get("steps", [])
        test_result = description_data.get("test_result", {}).get("steps", [])
        
        # Loop through each step of the description
        for step_data, test_step in zip(steps, test_result):
            step = step_data.get("step", "")
            user_input = step_data.get("user_input", "")
            expected_response = step_data.get("expected_response", "")
            bot_response = test_step.get("bot_response", "")
            result = test_step.get("result", "Failed")  # Get result from test result
            accuracy_score = test_step.get("accuracy_score", 0.0)  # Get accuracy score from test result

            # Generate the output record for this specific step
            output_record = {
                    "agent_id": agent_id,
                    "Testcase": description,
                    "TestCase Type": description_data.get("type", ""),
                    "Step": step,
                    "User Input": user_input,
                    "Expected Response": expected_response,
                    "Bot Response": bot_response,
                    "Result": result,
                    "Accuracy Score": accuracy_score
                }
            
            # Add the generated record to the output list
            output_records.append(output_record)

    for record in output_records:
        print('Record : ', record)
        table.create(record)
        print('\n')



