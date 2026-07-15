"""
Monitoring and structured logging
Production-grade metrics collection and JSON logging.
"""
import logging
import time
import json
from functools import wraps
from typing import Callable, Any, Dict
from datetime import datetime ,timezone

class JSONFormatter(logging.Formatter):
    """
    Converts normal logs into structured JSON format.
    JSON logs are easier to search and analyze in production tools.
    """

    def format(self, record: logging.LogRecord) -> str:

        # Create the basic log structure with useful debugging information
        log_record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),  # UTC time for consistency across servers
            "level": record.levelname,                            # INFO, WARNING, ERROR...
            "message": record.getMessage(),                       # Actual log message
            "module": record.module,                              # File where the log happened
            "function": record.funcName,                          # Function name
            "line_no": record.lineno,                              # Line number
        }

        # Add custom metadata passed using logger(..., extra={"extra_data": {...}})
        if hasattr(record, "extra_data"):
            log_record.update(record.extra_data)

        # Include full error traceback when logger.exception() is used
        if record.exc_info:
            log_record["exception"] = self.formatException(record.exc_info)

        # Return the final JSON log string
        return json.dumps(log_record)


def get_logger(name: str = "prod_rag") -> logging.Logger:
    """
    Creates and configures a reusable application logger.
    """

    # Get existing logger or create a new one with this name
    logger = logging.getLogger(name)

    # Avoid adding multiple handlers if the logger already exists
    if not logger.handlers:

        # Send logs to console (recommended for Docker/Kubernetes)
        handler = logging.StreamHandler()

        # Use our JSON formatter instead of default text format
        handler.setFormatter(JSONFormatter())

        # Attach handler to the logger
        logger.addHandler(handler)

        # Only show INFO level and above
        logger.setLevel(logging.INFO)

        # Prevent duplicate logs from the root logger
        logger.propagate = False

    return logger

#in prod use prometheus
class MetricsCollector:
    def __init__(self):
        self._requests_total=0
        self._errors_total=0
        self._latency_sum=0.0
        self._latency_count=0
        self._tokens_input=0
        self._tokens_output=0
        self._cache_hits=0
        self._cache_misses=0

    def record_request(
            self, 
            latency_ms: float, 
            error: bool, 
            input_tokens: int, 
            output_tokens: int, 
            cache_hit: bool
        ):

        self._requests_total += 1
        if error:
            self._errors_total += 1
        self._latency_sum += latency_ms
        self._latency_count += 1
        self._tokens_input += input_tokens
        self._tokens_output += output_tokens
        if cache_hit:
            self._cache_hits += 1
        else:
            self._cache_misses += 1
    def get_metrics(self):

        return {
            "requests_total": self._requests_total,
            "errors_total": self._errors_total,
            "average_latency_ms":
                self._latency_sum / self._latency_count
                if self._latency_count else 0,
            "tokens_input": self._tokens_input,
            "tokens_output": self._tokens_output,
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses
        }
    
class RequestTimer:

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self


    def __exit__(self, *args):
        self.end_time = time.perf_counter()

        self.elapsed_ms = (
            self.end_time - self.start_time
        ) * 1000





    