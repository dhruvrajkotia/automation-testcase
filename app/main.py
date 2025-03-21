# app/main.py
from fastapi import FastAPI
from contextlib import asynccontextmanager
from app.database.mongodb_connect import startup_db_client, shutdown_db_client
from app.routers import evaluate
from dotenv import load_dotenv
import asyncio
# Load environment variables from .env file
load_dotenv()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Start the database connection
    await startup_db_client(app)
    from app.consumer import start_rabbitmq_consumer
    import threading
    consumer_task = asyncio.create_task(start_rabbitmq_consumer())
    yield
    consumer_task.cancel()

    # Close the database connection
    await shutdown_db_client(app)

# Create the FastAPI app with lifespan management
app = FastAPI(lifespan=lifespan)

@app.get("/")
async def read_root():
    return {"message": "App Started"}


# Include other routers if needed
app.include_router(evaluate.router)

