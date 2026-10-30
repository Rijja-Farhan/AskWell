"""Errors the CLI turns into clear messages instead of tracebacks."""


class MissingAPIKeyError(RuntimeError):
    """The active provider's API key is not set."""


class IndexNotReadyError(RuntimeError):
    """The vector index is missing, empty, or built with another embedding model."""


class CorpusParseError(RuntimeError):
    """One or more corpus files failed to parse; the index was not changed."""

    def __init__(self, errors: list[str]) -> None:
        super().__init__(f"{len(errors)} file(s) failed to parse")
        self.errors = errors
