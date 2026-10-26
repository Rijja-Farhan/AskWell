"""Errors the CLI turns into clear messages instead of tracebacks."""


class MissingAPIKeyError(RuntimeError):
    """The active provider's API key is not set."""


class IndexNotReadyError(RuntimeError):
    """The vector index is missing, empty, or built with another embedding model."""
