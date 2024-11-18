import os
import json
import pika
import time
import asyncio
from app.services.evaluate import EvaluateService
from .logger import evaluate_service_logger as logger
from dotenv import load_dotenv

load_dotenv()
async def process_message(payload):
    try:
        # Initialize EvaluateService with the payload
        service = EvaluateService(payload)

        # Call start_process method to evaluate the test case
        result = await service.start_process()

        # Log the result
        logger.info("Evaluation result: %r", result)

        return result
    except Exception as e:
        logger.error(f"Error processing message: {e}")
        raise e

def on_message(ch, method, properties, body) -> None:
    payload = json.loads(body)
    logger.info("Received message: %r", payload)
    
    loop = asyncio.get_event_loop()
    try:
        result = loop.run_until_complete(process_message(payload))
        ch.basic_ack(delivery_tag=method.delivery_tag)
    except Exception:
        ch.basic_nack(delivery_tag=method.delivery_tag)  # Negative acknowledgment

def start_rabbitmq_consumer():
    max_retries = 3
    retries = 0

    while retries < max_retries:
        try:
            # Set up RabbitMQ connection parameters
            connection_parameters = pika.URLParameters(os.environ.get("RABBITMQ_CONNECT_URL"))
            connection = pika.BlockingConnection(connection_parameters)
            channel = connection.channel()

            # Declare the queue to ensure it exists
            queue_name = os.environ.get("RABBITMQ_EVALUATE_QUEUE_NAME")
            channel.queue_declare(queue=queue_name, durable=True)

            # Limit the number of unacknowledged messages
            channel.basic_qos(prefetch_count=1)

            # Start consuming messages from the queue
            channel.basic_consume(queue=queue_name, on_message_callback=on_message, auto_ack=False)

            logger.info(' [*] Waiting for messages in the queue. To exit press CTRL+C')
            channel.start_consuming()

        except Exception as e:
            retries += 1
            logger.error(f"Unexpected error: {e}, retrying {retries}/{max_retries} in 5 seconds...")
            time.sleep(5)  # Wait before retrying

            if retries >= max_retries:
                logger.error(f"RabbitMQ consumer encountered an unexpected error after {max_retries} attempts. Please investigate.")
                break  # Exit after max retries

        finally:
            try:
                connection.close()
            except Exception as e:
                logger.error(f"Error closing connection: {e}")

if __name__ == "__main__":
    start_rabbitmq_consumer()
