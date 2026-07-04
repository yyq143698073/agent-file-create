"""Structured exception hierarchy for document agent workflow.

Usage:
    from agent_file_create.errors import (
        DocAgentError, StepFatalError, StepRecoverableError,
        ConfigurationError, ExtractionError, LLMCallError,
    )

    def some_node(state):
        try:
            ...
        except LLMCallError as e:
            raise StepRecoverableError(f"LLM call failed: {e}") from e
        except IOError as e:
            raise StepFatalError(f"Disk write failed: {e}") from e

In the graph routing:
    - StepFatalError  → route to handle_error node (terminate)
    - StepRecoverableError → continue with degraded state (log warning)
"""


class DocAgentError(Exception):
    """Base exception for all document agent errors.

    All agent-specific exceptions should inherit from this class,
    allowing catch-all error boundaries to distinguish agent errors
    from unexpected system errors.
    """


class StepFatalError(DocAgentError):
    """Non-recoverable error — workflow should route to handle_error.

    Use when:
        - Disk write fails (no fallback)
        - Configuration is invalid at runtime
        - All retries exhausted on a critical operation
        - Corrupted state that cannot be recovered

    The graph should route these to the 'handle_error' node,
    which terminates the task with a user-visible error message.
    """


class StepRecoverableError(DocAgentError):
    """Recoverable error — node can continue with degraded state.

    Use when:
        - LLM call fails (can retry or use cached result)
        - RAG retrieval fails (can skip enrichment step)
        - Single file extraction fails (can proceed with remaining files)
        - Network timeout on optional step

    The caller should log a warning and continue with fallback/default values.
    This class exists to give code a structured way to signal degradation
    without requiring every caller to know the fallback logic.
    """


class ConfigurationError(DocAgentError):
    """Fatal error caused by invalid or missing configuration.

    Use when:
        - Required API key is missing
        - Model name is invalid or unsupported
        - Configuration values are out of valid range

    These should be caught at startup and prevent the application from running.
    """


class ExtractionError(StepRecoverableError):
    """Error during file extraction — single file can be skipped.

    Use when:
        - A single file fails OCR or text extraction
        - A file format is unsupported
        - A file is corrupted

    The extractor should continue with remaining files.
    """


class LLMCallError(StepRecoverableError):
    """LLM API call failed — can retry or use fallback.

    Use when:
        - Network timeout on LLM API
        - Rate limiting / 429 responses
        - Model returns malformed response

    Attributes:
        model: The model name that was called
        attempt: Which retry attempt failed (1-indexed)
        status_code: HTTP status code if available
    """

    def __init__(self, message: str, *, model: str = "", attempt: int = 0, status_code: int = 0):
        super().__init__(message)
        self.model = model
        self.attempt = attempt
        self.status_code = status_code


class RAGRetrievalError(StepRecoverableError):
    """RAG retrieval failure — can skip enrichment and continue.

    Use when:
        - Vector store is unreachable
        - Embedding service returns error
        - Knowledge base is empty or corrupted
    """


class RenderError(StepFatalError):
    """Template rendering failed — cannot produce final output.

    Use when:
        - Template file is missing or corrupted
        - Template variables cannot be resolved
        - PDF generation library fails
    """
