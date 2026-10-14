# AI PalmPilot (agent_palm)

> **Your memory. In your pocket. Shared when you want.**

A serverless, in-process memory layer for AI agents. No servers, no web UI, no daemons. Just a Python library that gives agents a personal, searchable notebook with tiered context (L0/L1/L2), semantic search, and a self-improving review loop.

## Features

- **Zero infrastructure** — Runs in-process as a Python library. No servers, no HTTP, no daemons.
- **Tiered context** — L0 (abstract, ~50 tokens), L1 (overview, ~500 tokens), L2 (full detail, on-demand). 80-90% token reduction.
- **Hybrid retrieval** — Vector similarity (sqlite-vec) + keyword search (FTS5) with RRF fusion.
- **Personal + shared pools** — Each agent has a private vault; a shared pool for cross-agent learning.
- **Self-improving review** — Agents annotate memories with confidence/verdicts; trust scores improve retrieval.
- **Local embeddings** — ONNX Runtime + all-MiniLM-L6-v2 int8 (23MB). No llama.cpp, no compilation, no GPU needed.
- **CLI for humans** — `mempalm stats/prune/export/review/backup/reembed`
- **Agent integrations** — Hermes MemoryProvider, MCP stdio tools, direct Python API

---

## Quick Start

```bash
# 1. Install
pip install agent-palm

# 2. Set your data directory (any local path with write access)
export MEMPALM_PATH=/path/to/your/mempalm
# Windows PowerShell:
# $env:MEMPALM_PATH = "C:\mempalm"

# 3. First memory (agent can do this autonomously)
python -c "
from agent_palm import PalmPilot
p = PalmPilot(agent_id='my-agent')
uri = p.remember('Decided to use ONNX Runtime for embeddings - no more llama.cpp crashes!',
                 namespace='decisions', tags=['architecture', 'embeddings'])
print('Stored:', uri)
"

# 4. Search
python -c "
from agent_palm import PalmPilot
p = PalmPilot(agent_id='my-agent')
results = p.search('embedding crashes', max_tokens=2000, pool='both')
for r in results:
    print(f'[{r.uri}::{r.level}] {r.text[:80]}...')
"

# 5. Recall detail
python -c "
from agent_palm import PalmPilot
p = PalmPilot(agent_id='my-agent')
detail = p.recall('mem://personal/my-agent/decisions/note/abc123', level='l2', max_tokens=4000)
print(detail.text)
"

# 6. Stats
mempalm stats --pool both
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  Agent Process (Hermes, Python script, etc.)                │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  agent_palm.PalmPilot                                │    │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌────────┐  │    │
│  │  │  Store   │ │  Embed   │ │ Extract  │ │Retrieve│  │    │
│  │  │ (SQLite) │ │ (ONNX)   │ │ (Heuristic)│ │(RRF)   │  │    │
│  │  └──────────┘ └──────────┘ └──────────┘ └────────┘  │    │
│  └─────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
                            │
              ┌─────────────┼─────────────┐
              ▼             ▼             ▼
         $MEMPALM_PATH/  $MEMPALM_PATH/  $MEMPALM_PATH/
         agents/         shared.db      models/
         <agent_id>.db   (shared pool)  all-MiniLM-L6-v2_int8.onnx
```

**Data Layout:**
- `$MEMPALM_PATH/agents/<agent_id>.db` — Personal vault per agent (private)
- `$MEMPALM_PATH/shared.db` — Shared knowledge pool (cross-agent)
- `$MEMPALM_PATH/models/` — ONNX model + tokenizer (downloaded on first use)

---

## Tiered Context (L0/L1/L2)

| Level | Size | When Loaded | Purpose |
|-------|------|-------------|---------|
| **L0** | ~50 tokens | **Always** — every search result | Quick relevance check |
| **L1** | ~500 tokens | Until token budget fills | Key facts for reasoning |
| **L2** | Unlimited | **Explicit recall only** | Full detail when needed |

**Result:** 80-90% token reduction vs stuffing raw context.

---

## API Reference

```python
from agent_palm import PalmPilot, MemoryPool, MemoryLevel

palm = PalmPilot(agent_id="my-agent")  # agent_id = your agent's unique identifier

# ─── STORE ───
uri = palm.remember(
    content="Full text content to remember",
    namespace="decisions",           # logical grouping
    tags=["architecture", "llm"],    # searchable labels
    pool=MemoryPool.PERSONAL,        # or MemoryPool.SHARED
    source="observed",               # observed|user|inferred|imported
    type_="note",                    # note|decision|fact|plan|entity
    ttl_days=30,                     # optional auto-expiry
    links=["mem://.../other"],       # optional cross-references
)
# Returns: MemoryRef(uri, pool, namespace, tokens, near_duplicate_of?)

# ─── SEARCH ───
results = palm.search(
    query="embedding model selection",
    namespace="decisions",           # optional filter
    pool=MemoryPool.BOTH,            # PERSONAL | SHARED | BOTH
    level=MemoryLevel.L1,            # L0 | L1 (L2 not allowed here)
    max_tokens=2000,                 # token budget for returned context
    include_superseded=False,        # skip superseded memories
)
# Returns: list[MemoryNode(uri, pool, owner, author, namespace, level, text, tags, source, verdict, score, created_at)]

# ─── RECALL ───
detail = palm.recall(
    uri="mem://personal/my-agent/decisions/note/abc123",
    level=MemoryLevel.L2,            # L0 | L1 | L2
    max_tokens=4000,                 # cap for L2
    offset=0,                        # pagination for large L2
)
# Returns: RecallNode(uri, level, text, truncated, total_tokens, offset)

# ─── ANNOTATE (Self-Improving Review) ───
palm.annotate(
    uri="mem://...",
    review_text="This pattern broke in v3.2; use new API instead",
    confidence=0.9,
    verdict="superseded",            # confirmed|uncertain|superseded|disputed
    evidence_uri="mem://...",        # optional supporting memory
)

# ─── REVIEW QUEUE ───
queue = palm.review_queue(namespace="decisions", limit=10)
# Returns: list[ReviewItem(memory, reason, suggested_action, similar_newer)]

# ─── MARK USED (Implicit Signal) ───
palm.mark_used(["mem://...", "mem://..."])  # call after agent cites memories

# ─── SUPERSEDE ───
new_uri = palm.supersede(
    old_uri="mem://...",
    new_content="Updated content replacing the old memory",
    **remember_kwargs  # same args as remember()
)

# ─── PUBLISH TO SHARED POOL ───
palm.publish(uri="mem://personal/my-agent/...", to_namespace="team/conventions")

# ─── FORGET ───
palm.forget(uri="mem://...")  # hard delete with cascade

# ─── UTILITIES ───
context = palm.format_context(nodes, max_tokens=2000)  # grounded context block
fake = palm.verify_citations(agent_response, provided_uris)  # hallucination guard
stats = palm.namespace_stats("decisions", pool=MemoryPool.PERSONAL)
changes = palm.changes_since(cursor=0, scope=MemoryPool.SHARED)  # for sync
```

---

## CLI Commands (Human-Facing)

```bash
# Store
mempalm remember "content" --namespace decisions --tags architecture,embeddings --pool personal

# Search
mempalm search "embedding crashes" --max-tokens 2000 --pool both --namespace decisions

# Recall
mempalm recall mem://personal/my-agent/decisions/note/abc123 --level l2 --max-tokens 4000

# Stats
mempalm stats --namespace decisions --pool both

# Backup (safe, uses SQLite Backup API)
mempalm backup --destination /path/to/backups/mempalm_20241219

# Prune old memories
mempalm prune --older-than 90 --pool personal --namespace decisions

# Export
mempalm export --namespace decisions --format jsonl > backup.jsonl

# Review queue
mempalm review --namespace decisions --unreviewed-only

# Re-embed with new model
mempalm reembed --model all-MiniLM-L6-v2 --batch-size 500
```

---

## Model

**Default:** `all-MiniLM-L6-v2 int8` (23MB, 384-dim, English-only)

- Downloaded automatically to `$MEMPALM_PATH/models/` on first use
- For multilingual: convert `multilingual-e5-small` to ONNX int8 and place in `$MEMPALM_PATH/models/`

---

## Requirements

- Python 3.11+ (3.12/3.13 recommended for Windows)
- Windows 10/11, Linux, macOS
- No GPU required (CPU inference ~3-8ms per embed)
- ~200MB RAM (model + index + SQLite)

---

## Hermes Integration (Detailed)

### For the Human (One-Time Setup)

**1. Install the package:**
```bash
pip install agent-palm
```

**2. Set the environment variable** (add to your shell profile / Hermes config):
```bash
# Linux/macOS
export MEMPALM_PATH=/home/user/.mempalm

# Windows PowerShell (persistent)
[Environment]::SetEnvironmentVariable("MEMPALM_PATH", "C:\mempalm", "User")

# Windows CMD (persistent)
setx MEMPALM_PATH "C:\mempalm"
```

**3. Configure Hermes** — Add to your Hermes profile config (`config.yaml`):
```yaml
memory:
  provider: agent_palm
  # Optional overrides (defaults work for most):
  # agent_id: "your-agent-name"      # defaults to profile name
  # namespace: "default"             # defaults to "default"
  # use_hash_embed: false            # true = no model download (testing only)
```

**4. Restart Hermes** — The provider loads automatically on startup.

> **That's it.** The agent now has persistent memory. No further human action needed.

---

### For the Agent (Autonomous Operation)

**The agent should do ALL of the following without human prompting:**

#### On First Run / Startup
```python
# The agent initializes its own memory client
from agent_palm import PalmPilot
palm = PalmPilot(agent_id="<own-agent-id>")  # knows its own ID
```

#### During Conversation — Remembering
```python
# After any significant decision, fact, or learning:
palm.remember(
    content="User prefers snake_case for Python internal variables",
    namespace="conventions",
    tags=["style", "python"],
    source="user",           # came from human
    pool=MemoryPool.PERSONAL
)
```

#### During Conversation — Searching
```python
# Before answering questions requiring context:
results = palm.search(
    query="user coding style preferences",
    max_tokens=1500,
    pool=MemoryPool.BOTH
)
# Use results in reasoning — they include L0/L1 with citations
```

#### After Answering — Marking Usage
```python
# After generating response, tell PalmPilot what was actually used:
palm.mark_used([r.uri for r in results if r.was_cited_in_response])
# This drives the review queue and trust scores
```

#### Periodic Self-Review (Agent-Driven)
```python
# Run during idle time or scheduled interval:
queue = palm.review_queue(limit=5)
for item in queue:
    # Agent decides: confirm, supersede, dispute, or forget
    palm.annotate(item.memory.uri, 
                  review_text="Still valid as of current version",
                  confidence=0.85,
                  verdict="confirmed")
```

#### Publishing to Shared Knowledge
```python
# When the agent learns something broadly useful:
palm.publish(uri="mem://personal/my-agent/...", to_namespace="team/best-practices")
```

---

### What the Human Should NEVER Do

| ❌ Don't | Why |
|---------|-----|
| Manually edit `.db` files | Corrupts indexes, breaks WAL |
| Run `mempalm` commands during agent operation | Lock contention |
| Set up cron jobs for pruning | Agent handles via `mark_used` + review queue |
| Choose which memories to store | Agent decides based on significance |
| Manage the shared pool | Agents publish explicitly; no central curator needed |

---

### What the Human MAY Do (Optional)

| ✅ Okay | When |
|--------|------|
| `mempalm stats` | Curiosity / debugging |
| `mempalm backup` | Before major system changes |
| `mempalm export` | Migration / archival |
| Adjust `MEMPALM_PATH` | Moving storage location |

---

## MCP Integration (stdio)

For agents/clients using MCP (Model Context Protocol):

```bash
# Start the MCP server (agent or human can launch)
mempalm-mcp
```

**Available Tools (7):**
| Tool | Description |
|------|-------------|
| `mem_add` | Store a memory (remember) |
| `mem_search` | Search with token budget |
| `mem_recall` | Recall full detail (L2) |
| `mem_forget` | Delete a memory |
| `mem_stats` | Namespace statistics |
| `mem_review` | Get review queue items |
| `mem_verify_citations` | Check for fabricated citations |

**Client Config Example:**
```json
{
  "mcpServers": {
    "mempalm": {
      "command": "mempalm-mcp",
      "args": [],
      "env": { "MEMPALM_PATH": "/path/to/mempalm" }
    }
  }
}
```

---

## Memory Flow Comparison: OpenViking vs AI PalmPilot

### OpenViking (Server-Based)
```
Agent → HTTP → Server Process → Embedding Service → Vector DB → HTTP → Agent
```
- **7+ network hops** per operation
- **Separate processes**: Go server, Python embedder, Rust storage
- **Failure modes**: Server crash, 429 rate limits, config bugs, WAL corruption
- **Human burden**: Start/stop server, monitor health, manage ports, update deps
- **Latency**: 50-500ms + network variance

### AI PalmPilot (In-Process)
```
Agent → Function Call → SQLite + sqlite-vec + FTS5 + ONNX (all in-process) → Return
```
- **1 function call** — zero network
- **Single process** — agent's own Python interpreter
- **Failure modes**: SQLite handles crashes (WAL), model loads lazily
- **Human burden**: One env var, done
- **Latency**: <10ms warm, <1s cold (model load)

**Same tiered memory concept. None of the operational fragility.**

---

## Project Structure

```
agent_palm/
├── __init__.py          # Exports: PalmPilot, MemoryNode, MemoryRef, MemoryPool, MemoryLevel
├── types.py             # Dataclasses & enums
├── exceptions.py        # PalmStorageBusyError, PalmModelMismatchError, etc.
├── schema.sql           # Single source of truth for DB schema
├── store.py             # PalmStore: SQLite + FTS5 + vec0, WAL, transactions
├── embed.py             # OnnxEmbedder (lazy) + HashEmbedder (tests)
├── extract.py           # Heuristic L0/L1 extraction
├── retrieval.py         # RRF fusion, budget packing, citation verification
├── review.py            # Annotations, trust, review queue
├── hermes_provider.py   # Hermes MemoryProvider implementation
├── mcp_tool.py          # MCP stdio server + tools
├── mempalm.py           # Click CLI entry point
├── pyproject.toml       # Package config, deps, entry points
├── policy.toml          # Shared pool permissions template
├── tests/
│   └── test_store.py    # 13 tests: CRUD, search, crash recovery, backup, etc.
└── README.md            # This file
```

---

## Development

```bash
# Install with dev dependencies
pip install -e .[dev]

# Run all tests
pytest tests/ -v

# Run specific test
pytest tests/test_store.py::test_remember_and_recall -v

# Benchmark KNN (requires 100k synthetic nodes)
python -m benchmarks.bench_knn --nodes 100000 --dim 384 --runs 100
```

---

## License

MIT — Built for agents, by agents.
