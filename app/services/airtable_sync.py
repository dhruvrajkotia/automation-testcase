import os
import uuid
from pyairtable import Api

myuuid = uuid.uuid4()

# Initialize Airtable API
api = Api(os.environ['AIRTABLE_API_KEY'])
table = api.table('apppY6a9bwgWBzXgw', 'tblhs1txs1fVhLFYp')

def push_airtable(input_data, agent_id):
    output_records = []
    # Loop through each description in the input data
    for description_data in input_data:
        description = description_data.get("description")
        steps = description_data.get("steps", [])
        testcaseId = str(myuuid)
        test_result = description_data.get("test_result", {}).get("steps", [])
        
        # Loop through each step of the description
        for step_data, test_step in zip(steps, test_result):
            step = step_data.get("step", "")
            user_input = step_data.get("user_input", "")
            expected_response = step_data.get("expected_response", "")
            bot_response = test_step.get("bot_response", "")
            result = test_step.get("result", "Failed")  # Default result is "Failed"
            accuracy_score = test_step.get("accuracy_score", 0.0)  # Default score is 0.0

            # Generate the output record for this specific step
            output_record = {
                "agent_id": agent_id,
                "Testcase ID": testcaseId,
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

    # Batch insert records into Airtable
    try:
        # Airtable allows batch insert of up to 10 records at a time
        batch_size = 10
        for i in range(0, len(output_records), batch_size):
            batch = output_records[i:i + batch_size]
            response = table.batch_create(batch)  # No need to nest under "fields"
            print(f"Batch {i // batch_size + 1} inserted successfully:", response)
    except Exception as e:
        print(f"Error while inserting records: {e}")
