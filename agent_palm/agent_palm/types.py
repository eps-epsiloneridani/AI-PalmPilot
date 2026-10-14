"""
Type definitions for agent_palm (AI PalmPilot)
"""

from dataclasses import dataclass, field
from typing import Literal, Sequence, Optional, Dict, Any
import time

# Literal types for constrained values
MemoryPool = Literal["personal", "shared"]
MemorySource = Literal["observed", "user", "inferred", "imported"]
ReviewVerdict = Literal["confirmed", "uncertain", "superseded", "disputed"]
MemoryLevel = Literal["l0", "l1", "l2"]


@dataclass(frozen=True)
class MemoryRef:
    """Reference returned after a successful remember()"""
    uri: str
    pool: MemoryPool
    namespace: str
    tokens: Dict[str, int]  # {"l0": int, "l1": int, "l2": int}
    near_duplicate_of: Optional[str] = None


@dataclass(frozen=True)
class MemoryNode:
    """Node returned by search() - contains L0 or L1 text"""
    uri: str
    pool: "MemoryPool"
    owner: str
    author: str
    namespace: str
    level: MemoryLevel
    text: str
    tags: Sequence[str]
    source: "MemorySource"
    latest_verdict: Optional["ReviewVerdict"]
    score: float
    created_at: float
    fields: Optional[Dict[str, Any]] = None


@dataclass(frozen=True)
class RecallNode:
    """Node returned by recall() - can contain L0, L1, or L2"""
    uri: str
    level: Literal["l0", "l1", "l2"]
    text: str
    truncated: bool
    total_tokens: int
    offset: int


@dataclass
class MemoryRecord:
    """Internal representation matching the database schema"""
    id: Optional[int] = None
    uri: str = ""
    pool: "MemoryPool" = "personal"
    owner: str = ""
    author: str = ""
    namespace: str = ""
    type: str = "note"
    source: "MemorySource" = "inferred"
    tags: Sequence[str] = field(default_factory=list)
    fields: Optional[Dict[str, Any]] = None
    l0: str = ""
    l1: str = ""
    l2: str = ""
    tok_l0: int = 0
    tok_l1: int = 0
    tok_l2: int = 0
    content_hash: str = ""
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    last_accessed_at: Optional[float] = None
    use_count: int = 0
    seq: int = 0
    trust: float = 0.5
    expires_at: Optional[float] = None

    def to_node(self, level: "MemoryLevel", score: float = 0.0) -> MemoryNode:
        """Convert to MemoryNode at specified level"""
        text_map = {"l0": self.l0, "l1": self.l1, "l2": self.l2}
        return MemoryNode(
            uri=self.uri,
            pool=self.pool,
            owner=self.owner,
            author=self.author,
            namespace=self.namespace,
            level=level,
            text=text_map[level],
            tags=self.tags,
            source=self.source,
            latest_verdict=None,  # Would be populated from annotations
            score=0.0,
            created_at=self.created_at,
            fields=self.fields,
        )


# Exception classes
class PalmError(Exception):
    """Base exception for agent_palm"""
    pass


class PalmStorageBusyError(PalmError):
    """Raised when SQLite returns SQLITE_BUSY/LOCKED after retries"""
    pass


class PalmModelMismatchError(PalmError):
    """Raised when database embedding model doesn't match runtime config"""
    pass


class PalmStorageFullError(PalmError):
    """Raised when storage is full (SQLITE_FULL)"""
    pass


class PalmValidationError(PalmError):
    """Raised when input validation fails"""
    pass