import os
import json
import pika
import time
import traceback
from app.services.evaluate import EvaluateService
from .logger import logger 
import asyncio
from app.services.airtable_sync import push_airtable

async def on_message(ch, method, properties, body) -> None:
    try:
        message = json.loads(body)
        logger.info("Message body is: %r" % message)
        agent_id = message.get("agent_id")

        
        # Initialize the EvaluateService
        evaluate_service = EvaluateService(message)
        
        # Start the evaluation process
        result = await evaluate_service.start_process()
        
        # Log the evaluation result
        #logger.info(f"Evaluation Result: {result}")
        print('Final result: ',result)
        print(agent_id)

        push_airtable(result,agent_id)


    except Exception as e:
        logger.error(e)
    finally:
        ch.basic_ack(delivery_tag=method.delivery_tag)

def start_rabbitmq_consumer():
    max_retries = 3
    retries = 0

    while retries < max_retries:
        try:
            connection_parameters = pika.URLParameters(os.environ.get("RABBITMQ_CONNECT_URL"))
            connection = pika.BlockingConnection(connection_parameters)
            channel = connection.channel()

            # Declare a queue (ensure the queue exists)
            queue_name = os.environ.get("RABBITMQ_EVALUATE_QUEUE_NAME")
            channel.queue_declare(queue=queue_name, durable=True)

            # Limit the number of unacknowledged messages to 1
            channel.basic_qos(prefetch_count=1)

            channel.basic_consume(
                queue=queue_name,
                on_message_callback=lambda ch, method, properties, body: asyncio.run(on_message(ch, method, properties, body)),
                auto_ack=False
            )

            logger.info(' [*] Waiting for messages. To exit press CTRL+C')
            channel.start_consuming()
        except Exception as e:
            retries += 1
            logger.error(
                f"Unexpected error: {e}, retrying {retries}/{max_retries} in 5 seconds...\n{traceback.format_exc()}")
            time.sleep(5)  # Wait before retrying
            if retries >= max_retries:
                logger.error(
                    f"RabbitMQ consumer encountered an unexpected error after {max_retries} attempts. Please investigate.")
                break  # Exit the loop after max retries
        finally:
            try:
                connection.close()
            except Exception as e:
                logger.error(f"Error closing connection: {e}")

