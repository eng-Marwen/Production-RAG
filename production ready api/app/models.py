"""API request and response models.
pydantic models for input validation
"""
from typing import Optional

from pydantic import BaseModel, Field, field_validator 
from datetime import datetime, timezone

class ChatRequest(BaseModel):
    """Incoming chat request model. (first req filter)"""
    message:str =Field(     #Field allows you to add extra rules and metadata to a field.
        ...,
        min_length=1,
        max_length=10000,
        description="The user's message to the agent."
    )

    @field_validator("message")     # Field for simple rules field_validator for custom validation logic
    @classmethod
    def validate_message(cls, value):     #prevent empty messages "           "
        if not value.strip():
            raise ValueError("Message cannot be empty")
        return value
    
    thread_id: str = Field(         #for history
        default="default",
        description="The thread ID for the conversation."
    )

class ChatResponse(BaseModel):
    response:str
    thread_id:str
    model_used:str
    cached:bool=False
    processing_time_ms:float
    timestamp:datetime = Field(default_factory=lambda:datetime.now(timezone.utc), description="The timestamp of the response.")

class HealthCheckResponse(BaseModel):
    """Health check response model."""
    status:str="Healthy"
    environment:str
    version:Optional[str]="1.0.0"
    checks:dict

class MetricsResponse(BaseModel):
    """Metrics response model."""
    total_requests:int
    total_errors:int
    error_rate:float
    avg_latency_ms:float
    cache_hit_rate:float
    total_input_tokens:int
    total_output_tokens:int

class ErrorResponse(BaseModel):
    """Error response model."""
    error:str
    details:str | None = None
    request_id:str | None = None

"""
BaseModel provides(validates data):

validation
type checking
JSON conversion
"""