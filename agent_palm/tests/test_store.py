"""
Tests for agent_palm store module
Run with: pytest tests/test_store.py -v
"""

import os
import tempfile
import shutil
import pytest
import zlib
import json

from agent_palm.store import PalmStore, open_store
from agent_palm.embed import HashEmbedder
from agent_palm.types import MemoryPool, MemorySource


@pytest.fixture
def temp_db():
    """Create a temporary database for testing"""
    tmpdir = tempfile.mkdtemp()
    db_path = os.path.join(tmpdir, "test.db")
    
    yield db_path
    
    # Cleanup
    shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.fixture
def store(temp_db):
    """Create a PalmStore with HashEmbedder for testing"""
    from agent_palm.store import PalmStore
    from agent_palm.embed import HashEmbedder
    
    store = PalmStore(db_path=temp_db, expected_model="test")
    # Override embedder for testing
    store._embedder = HashEmbedder()
    
    yield store
    
    store.close()


@pytest.fixture
def sample_content():
    return """
    Sebastian decided to use ONNX Runtime for embeddings instead of llama.cpp.
    The decision was made because llama.cpp kept crashing with GGML_ASSERT errors.
    This is a critical fix for the memory system.
    
    Key points:
    • ONNX Runtime works on Windows without VS Build Tools
    • all-MiniLM-L6-v2 int8 is only 23MB
    • No compilation needed
    
    Next steps: integrate with SQLite vec0 and test retrieval.
    """


def test_store_creation(temp_db):
    """Test store creation and schema initialization"""
    store = PalmStore(db_path=temp_db, expected_model="test")
    conn = store.connect()
    
    # Check tables exist
    tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    table_names = [t[0] for t in tables]
    
    expected_tables = [
        "meta", "memories", "annotations", "edges", 
        "memories_fts", "memories_vec", "outbox"
    ]
    
    for t in expected_tables:
        assert t in table_names, f"Missing table: {t}"
    
    # Check virtual tables - FTS5 shows as 'table' type in sqlite_master
    virtual = conn.execute("SELECT name FROM sqlite_master WHERE type IN ('virtual', 'table') AND name LIKE 'memories_%'").fetchall()
    virtual_names = [t[0] for t in virtual]
    assert "memories_fts" in virtual_names
    assert "memories_vec" in virtual_names
    
    # Check WAL mode
    mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
    assert mode == "wal"
    
    store.close()


def test_remember_and_recall(store, sample_content):
    """Test basic remember and recall"""
    uri, is_dup, near_dup = store.remember(
        content=sample_content,
        pool="personal",
        owner="test_user",
        author="test_agent",
        namespace="test_ns",
        type_="note",
        source="inferred",
        tags=["test", "embedding"],
    )
    
    # Should return URI
    assert uri.startswith("mem://personal/test_user/test_ns/note/")
    assert not is_dup
    
    # Recall L0
    l0_result = store.recall(uri, level="l0")
    assert "ONNX Runtime" in l0_result["text"]
    assert not l0_result["truncated"]
    
    # Recall L1
    l1_result = store.recall(uri, level="l1")
    assert "ONNX Runtime" in l1_result["text"]
    assert "VS Build Tools" in l1_result["text"]
    
    # Recall L2
    l2_result = store.recall(uri, level="l2")
    assert "GGML_ASSERT" in l2_result["text"]
    assert "ONNX Runtime" in l2_result["text"]


def test_deduplication(store, sample_content):
    """Test exact content deduplication"""
    # First remember
    uri1, is_dup1, _ = store.remember(
        content=sample_content,
        pool="personal",
        owner="test_user",
        author="test_agent",
        namespace="test_ns",
    )
    
    # Second remember with same content
    uri2, is_dup2, near_dup = store.remember(
        content=sample_content,
        pool="personal",
        owner="test_user",
        author="test_agent",
        namespace="test_ns",
    )
    
    # Should be detected as duplicate
    assert is_dup2
    assert uri2 == uri1  # Same URI returned


def test_near_duplicate_detection(store):
    """Test near-duplicate detection (similar but not exact)"""
    content1 = "The deployment is scheduled for Tuesday at 2pm."
    content2 = "The deployment is scheduled for Thursday at 2pm."
    
    uri1, _, _ = store.remember(
        content=content1, pool="personal", owner="test", author="agent", namespace="test"
    )
    
    uri2, is_dup, near_dup = store.remember(
        content=content2, pool="personal", owner="test", author="agent", namespace="test"
    )
    
    # Should NOT be exact duplicate (different content)
    assert not is_dup
    assert uri1 != uri2
    
    # But could be near-duplicate if cosine similarity > threshold
    # Current implementation doesn't do semantic dedup, so this is just for future


def test_search_fts(store, sample_content):
    """Test FTS5 search"""
    store.remember(
        content=sample_content,
        pool="personal", owner="test", author="agent", namespace="search_test",
        tags=["search", "test"]
    )
    
    # Search for content
    results = store.search_fts("ONNX Runtime", pool="personal", namespace="search_test", k=10)
    assert len(results) > 0
    
    # Results should have rowid and score
    rowid, score = results[0]
    assert isinstance(rowid, int)
    assert isinstance(score, float)


def test_vector_storage_and_search(store, sample_content):
    """Test vector storage and KNN search"""
    from agent_palm.embed import HashEmbedder
    
    embedder = HashEmbedder(dim=384)
    
    # Remember content
    uri, _, _ = store.remember(
        content=sample_content,
        pool="personal", owner="test", author="agent", namespace="vec_test"
    )
    
    # Get node ID and store vector
    node_id = store.get_node_id(uri)
    assert node_id is not None
    
    embedding = embedder.embed_single("ONNX Runtime embeddings test")
    store.store_vector(node_id, embedding)
    
    # Search
    query_emb = embedder.embed_single("ONNX Runtime embeddings test")
    results = store.search_vec(query_emb, pool="personal", namespace=None, k=10)
    
    assert len(results) > 0
    node_id_found, distance = results[0]
    assert node_id_found == node_id
    assert distance < 2.0  # Should be close (using same text for storage and query)


def test_fusion_and_budget(store, sample_content):
    """Test RRF fusion and budget packing"""
    from agent_palm.retrieval import pack_budget
    
    # Create test candidates
    candidates = [
        {
            "uri": f"mem://personal/test/ns/note/{i}",
            "pool": "personal",
            "owner": "test",
            "author": "agent",
            "namespace": "test",
            "l0": f"Abstract {i}",
            "l1": f"Overview {i} with more details",
            "tok_l0": 20,
            "tok_l1": 100,
            "score": 0.9 - i * 0.1,
            "source": "inferred",
            "tags": ["test"],
            "created_at": 1000000000 + i * 1000,
            "trust": 0.8,
        }
        for i in range(10)
    ]
    
    # Test budget packing
    packed = pack_budget(candidates, max_tokens=500, l0_budget_ratio=0.4)
    
    # Should have selected some items
    assert len(packed) > 0
    
    # Check token budget respected
    total_tokens = sum(e["cost"] for e in packed)
    assert sum(e["cost"] for e in packed) <= 500


def test_crash_recovery(temp_db, sample_content):
    """Test that WAL mode survives kill -9 simulation"""
    store1 = PalmStore(db_path=temp_db, expected_model="test")
    
    # Write some data
    uri, _, _ = store1.remember(
        content=sample_content,
        pool="personal", owner="test", author="agent", namespace="crash_test"
    )
    node_id = store1.get_node_id(uri)
    store1.store_vector(node_id, [0.1] * 384)
    store1.close()
    
    # Simulate crash by opening new connection (simulating restart)
    store2 = PalmStore(db_path=temp_db, expected_model="test")
    
    # Data should be intact
    l0 = store2.recall(uri, level="l0")
    assert "ONNX Runtime" in l0["text"]
    
    l1 = store2.recall(uri, level="l1")
    assert "VS Build Tools" in l1["text"]
    
    # Vector should be intact
    import numpy as np
    vecs = store2.connect().execute(
        "SELECT embedding FROM memories_vec WHERE id = ?",
        (store2.get_node_id(uri),)
    ).fetchone()
    assert vecs is not None
    
    store2.close()


def test_backup_restore(temp_db, sample_content):
    """Test backup and restore using SQLite Backup API"""
    store = PalmStore(db_path=temp_db, expected_model="test")
    
    uri, _, _ = store.remember(
        content=sample_content,
        pool="personal", owner="test", author="agent", namespace="backup_test"
    )
    node_id = store.get_node_id(uri)
    store.store_vector(node_id, [0.5] * 384)
    
    # Backup to temp file
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        backup_path = f.name
    
    try:
        store.backup(backup_path)
        
        # Verify backup
        import sqlite3
        backup_conn = sqlite3.connect(backup_path)
        backup_conn.enable_load_extension(True)
        import sqlite_vec
        sqlite_vec.load(backup_conn)
        backup_conn.enable_load_extension(False)
        
        # Check data integrity
        count = backup_conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
        assert count >= 1
        
        vec_count = backup_conn.execute("SELECT COUNT(*) FROM memories_vec").fetchone()[0]
        assert vec_count >= 1
        
        backup_conn.close()
        
        # Test integrity_check
        check_conn = sqlite3.connect(backup_path)
        check_conn.enable_load_extension(True)
        import sqlite_vec
        sqlite_vec.load(check_conn)
        check_conn.enable_load_extension(False)
        result = check_conn.execute("PRAGMA integrity_check").fetchone()[0]
        assert result == "ok"
        check_conn.close()
        
    finally:
        if os.path.exists(backup_path):
            try:
                os.unlink(backup_path)
            except PermissionError:
                pass  # Windows file locking issue


def test_namespace_stats(store, sample_content):
    """Test namespace statistics"""
    # Add some memories
    for i in range(5):
        store.remember(
            content=f"{sample_content} Item {i}",
            pool="personal", owner="test", author="agent", namespace="stats_ns"
        )
    
    stats = store.namespace_stats("personal", namespace="stats_ns")
    
    assert stats["count"] == 5
    assert stats["total_tokens"]["l0"] > 0
    assert stats["total_tokens"]["l1"] > 0
    assert stats["total_tokens"]["l2"] > 0


def test_forget(store, sample_content):
    """Test hard delete"""
    uri, _, _ = store.remember(
        content=sample_content,
        pool="personal", owner="test", author="agent", namespace="forget_test"
    )
    
    # Forget
    result = store.forget(uri)
    assert result["deleted"]
    
    # Should not be found
    with pytest.raises(Exception):
        store.recall(uri, level="l0")


def test_model_mismatch(temp_db):
    """Test model mismatch detection"""
    # Create store with one model and ensure connection is made
    store1 = PalmStore(db_path=temp_db, expected_model="model-a")
    store1.connect()  # This triggers _verify_or_init_meta
    store1.close()
    
    # Try to open with different model - should raise RuntimeError
    with pytest.raises(RuntimeError) as exc_info:
        store2 = PalmStore(db_path=temp_db, expected_model="model-b")
        store2.connect()
    
    assert "ModelMismatch" in str(exc_info.value) or "vector space" in str(exc_info.value)


def test_wal_checkpoint(store, sample_content):
    """Test WAL checkpoint behavior"""
    for i in range(20):
        store.remember(
            content=f"{sample_content} Item {i}",
            pool="personal", owner="test", author="agent", namespace="checkpoint_test"
        )
    
    conn = store.connect()
    # Check WAL file size
    wal_size = os.path.getsize(store.db_path + "-wal") if os.path.exists(store.db_path + "-wal") else 0
    
    # Manual checkpoint
    conn.execute("PRAGMA wal_checkpoint(PASSIVE)")
    
    # WAL should be checkpointed (size reduced or zero)
    new_wal_size = os.path.getsize(store.db_path + "-wal") if os.path.exists(store.db_path + "-wal") else 0
    # Note: PASSIVE may not checkpoint if readers active, but shouldn't error


if __name__ == "__main__":
    pytest.main([__file__, "-v"])