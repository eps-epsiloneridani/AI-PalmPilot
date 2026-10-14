"""
Retrieval utilities for agent_palm
RRF fusion, budget packing, citation verification, grounded context formatting
"""

import re
import math
from typing import List, Dict, Any, Sequence, Optional, Tuple


def rrf_fuse(
    vec_results: List[Tuple[int, float]], 
    fts_results: List[Tuple[int, float]], 
    k: int = 60,
    vec_weight: float = 1.0,
    fts_weight: float = 0.8
) -> Dict[int, float]:
    """
    Reciprocal Rank Fusion (RRF) for combining vector and FTS results.
    Returns dict of node_id -> fused score.
    """
    scores = {}
    
    # Vector results
    for rank, (node_id, _) in enumerate(vec_results):
        scores[node_id] = scores.get(node_id, 0.0) + vec_weight / (k + rank)
    
    # FTS results
    for rank, (node_id, _) in enumerate(fts_results):
        scores[node_id] = scores.get(node_id, 0.0) + fts_weight / (k + rank)
    
    return scores


def compute_recency_boost(
    created_at: float, 
    memory_type: str = "note",
    trust: float = 0.5
) -> float:
    """
    Time-decay relevance boost.
    Half-life by type, scaled by trust.
    """
    import time
    age_days = (time.time() - created_at) / 86400
    
    half_lives = {
        "decision": 180,
        "fact": 365,
        "status": 7,
        "plan": 30,
        "note": 30,
        "observation": 90,
    }
    
    half_life = half_lives.get(memory_type, 30)
    
    # Exponential decay with floor scaled by trust
    decay = 0.5 ** (age_days / half_life)
    floor = 0.1 + (trust * 0.3)  # 0.1 to 0.4
    
    return max(decay, floor)


def pack_budget(
    candidates: List[Dict[str, Any]], 
    max_tokens: int,
    format_overhead_tokens: int = 35,
    l0_budget_ratio: float = 0.40
) -> List[Dict[str, Any]]:
    """
    Progressive budget packing for tiered context.
    
    Phase 1: Fit L0 items until 40% of budget
    Phase 2: Upgrade to L1 in priority order until budget exhausted
    Phase 3: Never include L2 (explicit recall only)
    
    Returns list of MemoryNode-like dicts with level field.
    """
    if not candidates or max_tokens <= 0:
        return []
    
    total_budget = max_tokens
    l0_ceiling = int(total_budget * l0_budget_ratio)
    
    selected = []
    accumulated_tokens = 0
    
    # Sort by score descending
    candidates = sorted(candidates, key=lambda x: x.get("score", 0), reverse=True)
    
    # Phase 1: Fit L0 items
    for item in candidates:
        l0_cost = item.get("tok_l0", 0) + format_overhead_tokens
        if accumulated_tokens + l0_cost <= l0_ceiling:
            accumulated_tokens += l0_cost
            selected.append({
                "data": item,
                "level": "l0",
                "cost": l0_cost,
            })
        else:
            break
    
    # Phase 2: Upgrade to L1 in priority order
    remaining_budget = total_budget - accumulated_tokens
    
    for entry in selected:
        item = entry["data"]
        upgrade_cost = item.get("tok_l1", 0) - item.get("tok_l0", 0)
        if remaining_budget >= upgrade_cost:
            remaining_budget -= upgrade_cost
            entry["level"] = "l1"
            entry["cost"] = item.get("tok_l1", 0) + format_overhead_tokens
        else:
            # Keep at L0, stop upgrading
            break
    
    return selected


def build_memory_nodes(
    packed: List[Dict[str, Any]],
    format_overhead_tokens: int = 35
) -> List[Dict[str, Any]]:
    """Convert packed entries to MemoryNode dicts"""
    nodes = []
    for entry in packed:
        data = entry["data"]
        level = entry["level"]
        
        nodes.append({
            "uri": data["uri"],
            "pool": data["pool"],
            "owner": data["owner"],
            "author": data["author"],
            "namespace": data["namespace"],
            "level": level,
            "text": data["l1"] if level == "l1" else data["l0"],
            "tags": data.get("tags", []),
            "source": data.get("source", "inferred"),
            "latest_verdict": data.get("verdict"),
            "score": entry["data"].get("score", 0.0),
            "created_at": data.get("created_at", 0),
            "fields": data.get("fields"),
            "trust": data.get("trust", 0.5),
        })
    
    return nodes


def format_context(nodes: List[Dict[str, Any]], max_tokens: int = 4000) -> str:
    """
    Format nodes into grounded context block for agent consumption.
    Each memory wrapped with URI, level, and metadata for citation.
    """
    if not nodes:
        return ""
    
    lines = []
    total_tokens = 0
    overhead_per = 45  # URI header + metadata wrapper
    
    for node in nodes:
        # Estimate tokens for this entry
        text = node.get("text", "")
        token_est = len(text) // 4 + overhead_per
        
        if total_tokens + token_est > max_tokens:
            break
        
        uri = node.get("uri", "")
        level = node.get("level", "l0")
        source = node.get("source", "inferred")
        author = node.get("author", "unknown")
        trust = node.get("trust", 0.5)
        
        header = f'<memory uri="{uri}" level="{level}" source="{source}" author="{author}" trust="{trust:.2f}">'
        footer = '</memory>'
        
        lines.append(f"{header}\n{text}\n{footer}")
        total_tokens += token_est
    
    return "\n".join(lines)


def verify_citations(model_response: str, provided_uris: Sequence[str]) -> List[str]:
    """
    Hallucination guard: detect fabricated citations.
    Returns list of URIs cited in response that were NOT in provided context.
    """
    if not model_response or not provided_uris:
        return []
    
    # Pattern to match mem:// URIs
    uri_pattern = re.compile(r'mem://[^\s\]\)\>]+')
    cited = set(uri_pattern.findall(model_response))
    allowed = set(provided_uris)
    
    return list(cited - allowed)


def extract_cited_uris(model_response: str) -> List[str]:
    """Extract all mem:// URIs from model response"""
    return re.findall(r'mem://[^\s\]\)\>]+', model_response)


def citation_coverage(model_response: str, provided_uris: Sequence[str]) -> Dict[str, Any]:
    """
    Analyze citation quality in model response.
    Returns dict with coverage stats.
    """
    if not model_response:
        return {"total_citations": 0, "valid_citations": 0, "invalid_citations": 0, "coverage": 0.0}
    
    cited = extract_cited_uris(model_response)
    allowed = set(provided_uris)
    
    valid = [u for u in cited if u in allowed]
    invalid = [u for u in cited if u not in allowed]
    
    coverage = len(valid) / len(cited) if cited else 1.0
    
    return {
        "total_citations": len(cited),
        "valid_citations": len(valid),
        "invalid_citations": len(invalid),
        "coverage": coverage,
        "invalid_uris": invalid,
    }


def rrf_score(rank: Optional[int], k: int = 60, weight: float = 1.0) -> float:
    """Single RRF score for a rank"""
    if rank is None:
        return 0.0
    return weight / (k + rank)


def format_memory_for_context(node: Dict[str, Any], level: str = "l1") -> str:
    """Format a single memory node for context injection"""
    uri = node.get("uri", "")
    level_str = node.get("level", level)
    source = node.get("source", "inferred")
    author = node.get("author", "unknown")
    trust = node.get("trust", 0.5)
    text = node.get("text", "")
    
    return f'<memory uri="{uri}" level="{level_str}" source="{source}" author="{author}" trust="{trust:.2f}">\n{text}\n</memory>'


def build_grounded_context(nodes: List[Dict[str, Any]], max_tokens: int = 2000) -> str:
    """
    Build grounded context block with token budgeting.
    Returns formatted string ready for agent context.
    """
    if not nodes:
        return ""
    
    # Use pack_budget for tiered selection
    candidates = []
    for node in nodes:
        candidates.append({
            "uri": node.get("uri", ""),
            "pool": node.get("pool", "personal"),
            "owner": node.get("owner", ""),
            "author": node.get("author", ""),
            "namespace": node.get("namespace", ""),
            "l0": node.get("l0", ""),
            "l1": node.get("l1", ""),
            "tok_l0": len(node.get("l0", "")) // 4,
            "tok_l1": len(node.get("l1", "")) // 4,
            "score": node.get("score", 0.0),
            "source": node.get("source", "inferred"),
            "verdict": node.get("verdict"),
            "tags": node.get("tags", []),
            "fields": node.get("fields"),
            "created_at": node.get("created_at", 0),
            "trust": node.get("trust", 0.5),
        })
    
    packed = pack_budget(candidates, max_tokens)
    return format_context(build_memory_nodes(packed), max_tokens)


if __name__ == "__main__":
    # Quick test
    test_nodes = [
        {
            "uri": "mem://personal/sebastian/test/note/abc123",
            "pool": "personal",
            "owner": "sebastian",
            "author": "sebastian",
            "namespace": "test",
            "level": "l1",
            "text": "Decided to use ONNX Runtime for embeddings instead of llama.cpp.",
            "tags": ["decision", "embeddings"],
            "source": "inferred",
            "score": 0.85,
            "created_at": 1700000000,
            "trust": 0.8,
        },
        {
            "uri": "mem://shared/hermes/decision/xyz789",
            "pool": "shared",
            "owner": "hermes",
            "author": "ling",
            "namespace": "decisions",
            "level": "l0",
            "text": "Team agreed to use Python 3.11+ for all new services.",
            "tags": ["policy", "python"],
            "source": "observed",
            "score": 0.72,
            "created_at": 1699000000,
            "trust": 0.9,
        },
    ]
    
    context = build_grounded_context(test_nodes, max_tokens=500)
    print("Context:")
    print(context)
    print("---")
    
    # Test citation verification
    response = 'We decided to use ONNX [mem://personal/sebastian/test/note/abc123] and the team agreed on Python 3.11 [mem://shared/hermes/decision/xyz789].'
    provided = ["mem://personal/sebastian/test/note/abc123", "mem://shared/hermes/decision/xyz789"]
    
    invalid = verify_citations(response, provided)
    print(f"Invalid citations: {invalid}")
    
    coverage = citation_coverage(response, provided)
    print(f"Coverage: {coverage}")