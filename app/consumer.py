import pika
import json
import os
import asyncio
from app.services.evaluate import EvaluateService
from fastapi import FastAPI
from app.database.mongodb_connect import startup_db_client, shutdown_db_client
import sys

class Consumer:
    def __init__(self, app: FastAPI):
        self.app = app  # The app is passed here directly

        # Ensure MongoDB is initialized before the consumer starts
        if not hasattr(app, "mongodb"):
            print("MongoDB not initialized, exiting.")
            sys.exit(1)  # Exit if mongodb is not initialized

        # Set up RabbitMQ connection parameters
        self.connection_parameters = pika.URLParameters(os.environ.get("RABBITMQ_CONNECT_URL"))
        self.connection = pika.BlockingConnection(self.connection_parameters)
        self.channel = self.connection.channel()

        # Declare the queue to ensure it exists
        self.queue_name = os.environ.get("RABBITMQ_EVALUATE_QUEUE_NAME")
        self.channel.queue_declare(queue=self.queue_name, durable=True)

    async def callback(self, ch, method, properties, body):
        """
        Callback function to handle messages from RabbitMQ queue.
        """
        try:
            # Decode the received message
            payload = json.loads(body)

            # Print the consumed message
            print(f" [x] Received message: {json.dumps(payload, indent=2)}")

            # Instantiate the EvaluateService with the received payload
            evaluate_service = EvaluateService(payload, self.app.mongodb)

            # Start the evaluation process
            result = await evaluate_service.start_process()  # Await the async method

            # Print or log the evaluation result
            print(f"Evaluation Result: {result}")

            # Acknowledge the message as processed
            ch.basic_ack(delivery_tag=method.delivery_tag)

        except Exception as e:
            print(f"Error processing message: {e}")
            # Raise SystemExit to stop the consumer
            print("Stopping the consumer due to error...")
            ch.basic_nack(delivery_tag=method.delivery_tag)
            sys.exit(1)  # Terminate the program with an error code

    def start_consuming(self):
        """
        Start consuming messages from the queue.
        """
        print(f"Waiting for messages from {self.queue_name}...")
        self.channel.basic_consume(
            queue=self.queue_name,
            on_message_callback=self.on_message_callback,
            auto_ack=False  # We manually acknowledge the message after processing
        )
        self.channel.start_consuming()

    def on_message_callback(self, ch, method, properties, body):
        """
        Wraps the async callback to allow for proper async handling within a synchronous context.
        """
        asyncio.run(self.callback(ch, method, properties, body))  # Run the async callback in an event loop

    def stop_consuming(self):
        """
        Stop consuming messages from the queue.
        """
        self.channel.stop_consuming()


# Ensure this only runs if you execute consumer.py directly, not when imported
if __name__ == "__main__":
    from app.main import app  # Import app from main

    # Initialize MongoDB before starting the consumer
    loop = asyncio.get_event_loop()
    loop.run_until_complete(startup_db_client(app))  # Ensure DB is initialized before consumer starts

    # Now start the consumer
    consumer = Consumer(app)
    consumer.start_consuming()  # Start consuming messages when running consumer.py directly
