from fastapi import APIRouter, HTTPException
from app import schemas
from ..services.evaluate import EvaluateService
from ..publisher import Publisher


router = APIRouter(prefix="/testcase", tags=["evaluate"])


@router.post("/")
async def evaluate_testcase(testcase: schemas.Evaluate):
    """
    Endpoint to evaluate test cases using the EvaluateService.
    """
    try:
        # Initialize the EvaluateService with payload and request
        service = EvaluateService(testcase.model_dump())

        # Generate the test case prompt using the service and request
        result = await service.start_process()

        return result

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Internal server error: {e}")

@router.post("/trigger")
async def send_payload(payload: schemas.Evaluate):
    """
    Endpoint to send a payload to RabbitMQ using the Publisher.
    """
    try:
        publisher = Publisher()
        publisher.publish(payload.model_dump())
        return {"message": "Payload sent to RabbitMQ"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
