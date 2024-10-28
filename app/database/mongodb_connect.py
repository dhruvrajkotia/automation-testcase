import os
from motor.motor_asyncio import AsyncIOMotorClient
from fastapi import FastAPI


async def startup_db_client(app: FastAPI):
    """
    Initializes MongoDB connection and assigns it to the FastAPI app instance.
    """
    try:
        MONGO_URI = os.getenv("MONGO_URL")
        MONGO_DB_NAME = os.getenv("MONGO_DB_NAME")
        print(f"Connecting to MongoDB: {MONGO_URI}")

        app.mongodb_client = AsyncIOMotorClient(
            MONGO_URI, tls=True, tlsAllowInvalidCertificates=True)
        app.mongodb = app.mongodb_client.get_database(MONGO_DB_NAME)
        print("MongoDB connected.")
    except Exception as e:
        print(f"Failed to connect to MongoDB: {e}")
        raise e


async def shutdown_db_client(app: FastAPI):
    """
    Closes the MongoDB connection when the app shuts down.
    """
    try:
        app.mongodb_client.close()
        print("Database disconnected.")
    except Exception as e:
        print(f"Failed to disconnect from MongoDB: {e}")
