# Vera Agent
Vera Agent is an AI agent with tool-calling capabilities, generated based on an exercise proposed by the Vera Money engineering team. It can:
- Answer general questions using an LLM.
- Call tools for current weather and stock prices.
- Search an internal knowledge base via semantic similarity (Chroma DB).

## Architecture
Key architectural choices:
- **FastAPI API layer:** `app/main.py` exposes the HTTP surface and wires routers, keeping I/O at the edges.
- **Agent orchestration in services:** `VeraAgent` composes tools and uses LangChain’s tool-calling agent to decide when to call them.
- **Ports and adapters:** `app/domain/ports.py` defines interfaces; `app/infrastructure/` provides concrete adapters (OpenAI + Chroma), making swaps easier.
- **Vector store with persistence:** Chroma runs in persistent mode and stores embeddings under `data/chroma` by default.
- **Config via env + dotenv:** `app/core/config.py` loads settings from `.env` or environment variables.

## Folder structure
- `app/`: application code
  - `app/main.py`: FastAPI app wiring.
  - `app/routers/`: HTTP endpoints (`/health`, `/agents/answer`).
  - `app/services/`: agent orchestration and tools.
  - `app/infrastructure/`: adapters for OpenAI and Chroma.
  - `app/domain/`: ports (interfaces) and data contracts.
  - `app/core/`: config + logging.
- `scripts/`: operational scripts (e.g., Chroma ingestion).
- `data/`: local data and Chroma persistence directory.
- `assets/`: static assets (if any).
- `tests/`: test suite.

## Docker
Build the image:
```bash
docker build -t vera-agent .
```

Run the container (requires `OPENAI_API_KEY` and, optionally, `ALPHAVANTAGE_API_KEY`):
```bash
docker run --rm -p 8000:8000 \
  -e OPENAI_API_KEY=your_key_here \
  -e ALPHAVANTAGE_API_KEY=your_key_here \
  -e CHROMA_PERSIST_DIR=/data/chroma \
  -v vera_chroma:/data/chroma \
  vera-agent
```

Alternatively, with Compose (reads `.env`):
```bash
docker compose up --build
```

## Chroma DB ingestion scripts
There is one ingestion script: `scripts/ingest_file.py`.

What it does:
- Reads a `.txt` file.
- Splits content into paragraphs (double newline).
- Filters short paragraphs (`--min-length`, default 20 chars).
- Generates embeddings via Embedding Port.
- Upserts chunks into Chroma with metadata (`source`, `file_name`, `paragraph_index`, `file_path`).

How to run it:
1) Ensure environment variables are set (at least `OPENAI_API_KEY`). Optionally set:
   - `CHROMA_PERSIST_DIR` (default: `data/chroma`)
   - `CHROMA_COLLECTION` (default: `vera_docs`)
2) Execute the script:

```bash
python scripts/ingest_file.py \
  --file path/to/your_doc.txt \
  --source docs \
  --name vera_terms \
  --min-length 20
```

After ingestion, the knowledge base tool will retrieve chunks from Chroma and attach a `Sources:` line to agent responses when available.

## Conversation memory
The `/agents/answer` endpoint supports short-term, in-memory conversation history.
- Request includes `message` and an optional `conversation_id`.
- If `conversation_id` is omitted or blank, the API generates one and returns it with the answer.
- Subsequent requests with the same `conversation_id` reuse a short rolling memory window.

Example request/response:
```json
// POST /agents/answer
{
  "message": "I need AAPL stock price",
  "conversation_id": "optional"
}
```

```json
{
  "answer": "And google's?",
  "conversation_id": "generated_if_missing"
}
```

Notes:
- Memory lives only in RAM (resets on restart) and is per-process.
- The current memory window is 6 turns.

## Clarifications
1. The exercise specifies a weather tool that accepts a city name. I used an API that works with latitude and longitude instead. The agent can still be asked for the weather in a city, but it will resolve that to coordinates and pass latitude/longitude to the tool for a precise location.
