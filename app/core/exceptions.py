class ChatWithDocsError(Exception):
    """Base exception for all application errors."""


class DocumentParsingError(ChatWithDocsError):
    """Raised when a document cannot be parsed."""


class EmbeddingError(ChatWithDocsError):
    """Raised when embedding generation fails."""


class RetrievalError(ChatWithDocsError):
    """Raised when vector retrieval fails."""


class GuardrailViolationError(ChatWithDocsError):
    """Raised when input/output fails a guardrail check."""


class InsufficientContextError(ChatWithDocsError):
    """Raised when retrieved chunks score below similarity threshold."""