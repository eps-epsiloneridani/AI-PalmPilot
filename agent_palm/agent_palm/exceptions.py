"""
Custom exceptions for agent_palm
"""

class PalmError(Exception):
    """Base exception for agent_palm"""
    pass


class PalmStorageBusyError(Exception):
    """Raised when SQLite returns SQLITE_BUSY/LOCKED after retries"""
    def __init__(self, message: str, retryable: bool = True):
        super().__init__(message)
        self.retryable = retryable


class PalmModelMismatchError(Exception):
    """Raised when database embedding model doesn't match runtime config"""
    def __init__(self, message: str, db_model: str = None, expected_model: str = None):
        super().__init__(message)
        self.db_model = db_model
        self.expected_model = expected_model


class PalmStorageFullError(Exception):
    """Raised when storage is full (SQLITE_FULL)"""
    pass


class PalmValidationError(Exception):
    """Raised when input validation fails"""
    pass


class PalmNotFoundError(Exception):
    """Raised when a memory URI is not found"""
    pass


class PalmPermissionError(Exception):
    """Raised when an operation is not permitted (e.g., forget shared memory)"""
    pass