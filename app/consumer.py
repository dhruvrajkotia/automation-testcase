import os
import json
import traceback
import asyncio
from app.services.evaluate import EvaluateService
from .logger import logger
from app.services.airtable_sync import push_airtable
import aio_pika
from concurrent.futures import ThreadPoolExecutor

executor = ThreadPoolExecutor(max_workers=10)

async def on_message(message: aio_pika.IncomingMessage):
    async with message.process():
        try:
            body = message.body
            message_data = json.loads(body)
            logger.info(f"Message body: {message_data}")
            agent_id = message_data.get("agent_id")

            # Initialize the EvaluateService
            evaluate_service = EvaluateService(message_data)

            # Start the evaluation process
            result = await evaluate_service.start_process()
            logger.info(f"Final result: {result}")
            print('Final result: ', result)
            print(agent_id)

            # Run the synchronous push_airtable in a separate thread
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(executor, push_airtable, result, agent_id)

        except Exception as e:
            logger.error(f"Error processing message: {e}")

async def start_rabbitmq_consumer():
    max_retries = 3
    retries = 0

    while retries < max_retries:
        try:
            # Connect to RabbitMQ using aio_pika
            connection = await aio_pika.connect_robust(os.environ.get("RABBITMQ_CONNECT_URL"))
            async with connection:
                # Create a channel
                channel = await connection.channel()
                queue_name = os.environ.get("RABBITMQ_EVALUATE_QUEUE_NAME")

                # Declare a queue (ensure the queue exists)
                queue = await channel.declare_queue(queue_name, durable=True)

                logger.info(' [*] Waiting for messages. To exit press CTRL+C')

                # Consume messages
                async with queue.iterator() as queue_iter:
                    async for message in queue_iter:
                        # Process each message using on_message
                        await on_message(message)

        except Exception as e:
            retries += 1
            logger.error(f"Unexpected error: {e}, retrying {retries}/{max_retries} in 5 seconds...\n{traceback.format_exc()}")
            await asyncio.sleep(5)  # Wait before retrying
            if retries >= max_retries:
                logger.error(f"RabbitMQ consumer encountered an unexpected error after {max_retries} attempts. Please investigate.")
                break  # Exit the loop after max retries

