import pika
import json
import os

class Publisher:
    def __init__(self):
        # Set up RabbitMQ connection parameters
        self.connection_parameters = pika.URLParameters(os.environ.get("RABBITMQ_CONNECT_URL", "amqp://guest:guest@localhost/"))
        self.connection = pika.BlockingConnection(self.connection_parameters)
        self.channel = self.connection.channel()

        # Declare a queue to ensure it exists
        self.queue_name = os.environ.get("RABBITMQ_EVALUATE_QUEUE_NAME", "evaluate_queue")
        self.channel.queue_declare(queue=self.queue_name, durable=True)

    def publish(self, payload: dict):
        """
        Publishes the payload to the RabbitMQ queue
        """
        # Convert payload to JSON
        payload_json = json.dumps(payload)

        # Publish the message to RabbitMQ
        self.channel.basic_publish(
            exchange='',
            routing_key=self.queue_name,
            body=payload_json,
            properties=pika.BasicProperties(
                delivery_mode=2,  # Make message persistent
            )
        )
        print(f" [x] Sent payload: {payload_json}")
