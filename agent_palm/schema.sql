-- agent_palm schema.sql
-- Single source of truth for database schema
-- Run via: sqlite3 mempalm.db < schema.sql

PRAGMA journal_mode = WAL;
PRAGMA synchronous = NORMAL;
PRAGMA busy_timeout = 5000;
PRAGMA foreign_keys = ON;
PRAGMA secure_delete = ON;

-- Configuration and model integrity lock
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

-- Core memory records
CREATE TABLE IF NOT EXISTS memories (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    uri TEXT UNIQUE NOT NULL,
    pool TEXT NOT NULL CHECK (pool IN ('personal', 'shared')),
    owner TEXT NOT NULL,         -- Agent or profile owning the namespace
    author TEXT NOT NULL,        -- Agent that executed remember()
    namespace TEXT NOT NULL,
    type TEXT NOT NULL DEFAULT 'note',
    source TEXT NOT NULL CHECK (source IN ('observed', 'user', 'inferred', 'imported')),
    tags TEXT NOT NULL DEFAULT '[]', -- JSON string list
    fields TEXT,                     -- Optional JSON object for structured payloads
    l0 TEXT NOT NULL,
    l1 TEXT NOT NULL,
    l2 BLOB,                         -- zlib-compressed full content
    tok_l0 INTEGER NOT NULL,
    tok_l1 INTEGER NOT NULL,
    tok_l2 INTEGER,
    content_hash TEXT NOT NULL,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    last_accessed_at REAL,
    use_count INTEGER NOT NULL DEFAULT 0,
    seq INTEGER NOT NULL,             -- Monotonic increment for change tracking
    trust REAL NOT NULL DEFAULT 0.5,
    expires_at REAL
);

CREATE INDEX IF NOT EXISTS idx_memories_lookup 
ON memories(pool, namespace, created_at);

CREATE UNIQUE INDEX IF NOT EXISTS idx_memories_dedup 
ON memories(pool, namespace, content_hash);

-- Review loop: Append-only annotations
CREATE TABLE IF NOT EXISTS annotations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    memory_id INTEGER NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
    reviewer TEXT NOT NULL,
    verdict TEXT NOT NULL CHECK (verdict IN ('confirmed', 'uncertain', 'superseded', 'disputed')),
    confidence REAL NOT NULL CHECK (confidence >= 0.0 AND confidence <= 1.0),
    review_text TEXT NOT NULL,
    evidence_uri TEXT,
    review_hash TEXT NOT NULL,
    created_at REAL NOT NULL,
    UNIQUE(memory_id, reviewer, review_hash)
);

CREATE INDEX IF NOT EXISTS idx_annotations_mem ON annotations(memory_id);

-- Explicit memory relationships
CREATE TABLE IF NOT EXISTS edges (
    src_id INTEGER NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
    dst_uri TEXT NOT NULL,
    kind TEXT NOT NULL CHECK (kind IN ('supersedes', 'contradicts', 'derived_from', 'related')),
    created_at REAL NOT NULL,
    PRIMARY KEY (src_id, dst_uri, kind)
);

-- Full-Text Search: indexes L0, L1, and raw uncompressed L2
-- Contentless delete mode enables sync with parent table
CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
    l0, 
    l1, 
    l2, 
    tags, 
    content='', 
    contentless_delete=1, 
    tokenize='trigram'
);

-- Vector storage: sqlite-vec virtual table
-- Maps 1:1 to memories.id
CREATE VIRTUAL TABLE IF NOT EXISTS memories_vec USING vec0(
    id INTEGER PRIMARY KEY,
    embedding FLOAT[384]
);

-- Async publish queue for shared pool
CREATE TABLE IF NOT EXISTS outbox (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    uri TEXT NOT NULL,
    target_ns TEXT NOT NULL,
    created_at REAL NOT NULL
);

-- Meta table for config and model integrity
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

-- Schema version tracking
INSERT OR IGNORE INTO meta (key, value) VALUES ('schema_version', '1');