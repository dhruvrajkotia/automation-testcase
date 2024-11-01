from fastapi import FastAPI
from contextlib import asynccontextmanager
from app.database.mongodb_connect import startup_db_client, shutdown_db_client
from app.routers import evaluate
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Start the database connection
    await startup_db_client(app)
    yield
    # Close the database connection
    await shutdown_db_client(app)

# Create the FastAPI app with lifespan management
app = FastAPI(lifespan=lifespan)


@app.get("/")
async def read_root():
    return {"message": "App Started"}

app.include_router(evaluate.router)
