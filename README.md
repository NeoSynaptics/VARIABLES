# VARIABLES

A local vault of self-contained, production-tested code modules.

Each module is one folder. Drop it into any project and it works.
The LLM uses this as a smart router — it reads the index, picks the right module, wires it in.

---

## How to use a module

```bash
# Option A: copy the folder
cp -r VARIABLES/StreamingLLM/ my-project/

# Option B: sparse-checkout from GitHub
python VARIABLES/SparseCheckout/sparse.py NeoSynaptics/AlchemyTools StreamingLLM
```

## How to save a new module

Say **"Claude save this tool"** — Claude reads `CLAUDE_SAVE.md` and handles the rest.

Or call the API directly:

```http
POST http://localhost:8000/variables/save
{
  "name": "MyFunction",
  "language": "python",
  "body": "def my_function(x): ...",
  "tags": ["utils", "math"]
}
```

## How to search

```http
GET http://localhost:8000/variables/modules/search?q=streaming&lang=python
```

---

## Internal Modules

| Module | Language | Description | Tags |
|--------|----------|-------------|------|
| [StreamingLLM](StreamingLLM/) | python+typescript | SSE streaming server (Ollama/OpenAI) + frontend client | llm, streaming, sse, ollama |
| [OllamaRouter](OllamaRouter/) | python | Complexity-based model routing — trivial→7B, complex→14B | ollama, routing, llm |
| [OllamaGate](OllamaGate/) | python+typescript | AI-powered permission reviewer for Claude Code actions | security, ollama, claude-code |
| [RealtimeWS](RealtimeWS/) | python+typescript | Bidirectional WebSocket relay with rooms + auto-reconnect client | websocket, realtime, relay |
| [TerminalCapture](TerminalCapture/) | typescript | Capture and filter VS Code terminal output | vscode, terminal, capture |
| [ReuseScanner](ReuseScanner/) | python | Detect reusable functions across repos. Scores 0-4. Zero deps. | code-analysis, reuse, scanner |
| [DiffEngine](DiffEngine/) | python | Smart diff detection + semantic change classification | diff, git, code-analysis |
| [TaskQueue](TaskQueue/) | python | Async FIFO task queue with pause/resume/stop + retry | async, queue, worker |
| [TaskDecomposer](TaskDecomposer/) | python | Break a PRD or goal into ordered, executable subtasks | tasks, llm, planning |
| [MCPServer](MCPServer/) | python | Minimal MCP server template for Claude Code tool registration | mcp, claude, tools |
| [SparseCheckout](SparseCheckout/) | python+bash | Pull a single folder from any git repo without cloning everything | git, utility |
| [DiagView](DiagView/) | html+javascript | 3D force-graph visualization of AlchemyGoldOS module health | diagnostics, visualization, 3d |

---

## External Catalog

Curated repos worth pulling. Use `SparseCheckout` to grab only what you need.

See [external.json](external.json) for the full list with repo URLs and tags.

| Name | Description | Tags |
|------|-------------|------|
| DeepMapping | Interactive knowledge graph from any GitHub repo (D3.js) | knowledge-graph, code-analysis |
| TreeSitterLight | Pythonic tree-sitter wrapper with auto-language loading | ast, parsing |
| LocalRAG | Minimal RAG from scratch — no LangChain, no cloud | rag, embeddings, local |
| MotionTrack | ByteTrack multi-object tracker standalone | tracking, computer-vision |
| InjectGuard | Prompt injection detection (YARA + transformer + canary) | security, llm |
| Agno | Full agentic runtime — memory, tools, 100+ integrations | agent, ollama |
| ToolBridge | Tool calling for ANY Ollama model that lacks it natively | ollama, tool-calling |
| OllamaMCPBridge | Bridge between Ollama and MCP servers | ollama, mcp |
| *+17 more* | See external.json | — |

---

## Structure

```
VARIABLES/
├── README.md               ← this file
├── CLAUDE_SAVE.md          ← ritual for saving new modules
├── build_index.py          ← generates index.json from all module.json files
├── save_candidate.py       ← ReuseScanner → vault handoff
├── index.json              ← auto-generated, do not edit manually
├── external.json           ← curated external repo catalog
│
├── StreamingLLM/
│   ├── module.json         ← name, description, language, tags, entry
│   ├── stream_server.py    ← entry point
│   └── stream_client.ts
│
├── DiagView/
│   ├── module.json
│   └── diag_view.html      ← self-contained 3D diagnostic viewer
│
└── ...                     ← one folder per module
```

### `module.json` schema

```json
{
  "name": "ModuleName",
  "description": "One sentence what it does.",
  "language": "python | typescript | python+typescript | html+javascript",
  "tags": ["tag1", "tag2"],
  "source": "self | external",
  "size": "small | medium | large",
  "entry": "main_file.py"
}
```

---

## Rebuild the index

```bash
python build_index.py
# → index.json: 12 modules (12 internal, 25 external)
```

Or via API: `POST http://localhost:8000/variables/modules/rebuild`

---

## CLAUDE_SAVE ritual

When you say **"Claude save this tool"**:

1. Claude reads this README + `CLAUDE_SAVE.md`
2. Extracts the function(s), adds imports, strips project-specific names
3. Creates `VARIABLES/{ModuleName}/` with code + `module.json`
4. Calls `build_index.py` to update `index.json`
5. Updates the table in this README
6. Commits to GitHub

No manual steps needed.
