# llm-zoomcamp-2026-code
# Module 1 — Agentic RAG Pipeline

A Retrieval-Augmented Generation (RAG) pipeline built with **sqlitesearch**, **Ollama**, and **function calling** — from a fixed pipeline to a fully agentic loop.

Part of the [DataTalks.Club LLM Zoomcamp](https://github.com/DataTalksClub/llm-zoomcamp) course.

---

## What It Does

Takes a student question, searches a persistent FAQ knowledge base, and returns a grounded answer using a local LLM. In agentic mode, the LLM decides when and how many times to search before answering.

---

## Architecture

### Part 1 — Fixed RAG Pipeline

```
User question → search_documents() ↔ SQLite index → build_prompt() → LLM → Answer
```

- `build_index()` — one-time setup, indexes all FAQ documents into `faq.db`
- `search_documents()` — inference-time tool, queries the persistent index
- `build_prompt()` — assembles system instructions + retrieved context + user question
- LLM — OpenAI API (`gpt-5.4-mini` or similar)

### Part 2 — Agentic Loop

```
User question → LLM agent → function_call? → yes: run tool → append result → loop
                                           → no:  return final answer
```

The LLM drives the loop. It decides when to search, what keywords to use, and when it has enough context to answer. The loop runs until no more tool calls are returned.

---

## Project Structure

```
.
├── main.py                 # Entry point — runs the RAG pipeline or agentic loop
├── rag_helper.py           # RAGBase class, build_index(), search_documents(), tool schema
├── notebook.ipynb          # Exploratory notebook — step-by-step walkthrough
├── pyproject.toml          # Project dependencies (managed with uv)
├── uv.lock                 # Locked dependency versions
├── .python-version         # Python version pin
├── .gitignore
└── README.md
```

---

## Quickstart

### 1. Install dependencies

```bash
uv venv
source .venv/bin/activate
uv pip install sqlitesearch python-dotenv openai
```

### 2. Set environment variables

```bash
cp .env.example .env
# Add your OpenAI API key
echo "OPENAI_API_KEY=sk-..." >> .env
```

### 4. Build the index (one-time)

```python
from search import build_index
import json

with open("data/documents.json") as f:
    documents = json.load(f)

build_index(documents)
```

### 5. Run the fixed RAG pipeline

```python
from rag import rag

answer = rag("I just discovered the course. Can I still join?")
print(answer)
```

### 6. Run the agentic loop

```python
from agent import agent_loop

answer = agent_loop("How do I run the course locally?")
print(answer)
```

---

## Key Components

### `build_index(documents)`

Indexes all documents into a persistent SQLite database. Call this **once** at setup — not at inference time.

```python
index = TextSearchIndex(
    text_fields=["question", "section", "answer"],
    keyword_fields=["course"],
    db_path="faq.db"
)
for doc in documents:
    index.add(doc)
index.close()
```

### `search_documents(question, course)`

Opens the existing `faq.db` and queries it. Reopening the index does **not** reindex — SQLite persists data on disk.

```python
results = index.search(
    question,
    boost_dict={"question": 2.0, "section": 0.5},
    filter_dict={"course": course}
)
```

### Tool Schema

The search function is exposed to the LLM as an OpenAI-compatible JSON tool schema. The same schema works across OpenAI, Groq, Ollama, and Anthropic with minor response-parsing differences.

```python
tools = [
    {
        "type": "function",
        "function": {
            "name": "search_documents",
            "description": "Search the FAQ knowledge base for relevant documents",
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {"type": "string", "description": "The search query"},
                    "course":   {"type": "string", "description": "Course name to filter by"}
                },
                "required": ["question", "course"]
            }
        }
    }
]
```

### Agentic Loop

```python
while True:
    response = openai_client.responses.create(
        model=model,
        input=messages,
        tools=tools
    )

    messages.extend(response.output)
    has_function_calls = False

    for item in response.output:
        if item.type == "function_call":
            args = json.loads(item.arguments)
            result = search_documents(args["question"], args["course"])
            messages.append({
                "type": "function_call_output",
                "call_id": item.call_id,
                "output": json.dumps(result)
            })
            has_function_calls = True
        elif item.type == "message":
            last_answer = item.content[0].text

    if not has_function_calls:
        return last_answer
```

---

## Design Decisions

| Decision | Reason |
|---|---|
| Separate `build_index()` and `search_documents()` | Indexing is one-time setup; searching is per-request. Combining them re-indexes on every query. |
| System message carries instructions + context | User message carries only the question. Mixing them breaks role separation and multi-turn patterns. |
| Tool schema defined once | Same JSON schema works across Ollama, OpenAI, Groq — only response parsing differs. |
| `RAGBase` class pattern | One-time setup (fetching docs, fitting the index, creating clients) goes outside the class; per-request behavior uses `self.*` inside. |

---

## Requirements

- Python 3.11+
- OpenAI API key (~$1–5 in credits for the whole course)
- No GPU required

---

## Part of LLM Zoomcamp

This project covers **Module 1** of the [DataTalks.Club LLM Zoomcamp](https://github.com/DataTalksClub/llm-zoomcamp):

- Lessons 1–10: Fixed RAG pipeline with keyword search
- Lessons 11–16: Agentic loop with function calling

Course is free and self-paced. [Sign up or follow along →](https://github.com/DataTalksClub/llm-zoomcamp)
---


## Quickstart

### 1. Install dependencies

```bash
uv venv
source .venv/bin/activate
uv pip install sqlitesearch python-dotenv ollama
```

### 2. Pull a model with Ollama

```bash
ollama pull qwen2.5:7b-instruct
# or
ollama pull llama3.1:8b
```

> Both models support tool/function calling. Check [ollama.com/search](https://ollama.com/search) and filter by **Tools** for other options.

### 3. Set environment variables

```bash
cp .env.example .env
# Edit .env to set OLLAMA_MODEL if needed (default: qwen2.5:7b-instruct)
```

### 4. Build the index (one-time)

```python
from search import build_index
import json

with open("data/documents.json") as f:
    documents = json.load(f)

build_index(documents)
```

### 5. Run the fixed RAG pipeline

```python
from rag import rag

answer = rag("I just discovered the course. Can I still join?")
print(answer)
```

### 6. Run the agentic loop

```python
from agent import agent_loop

answer = agent_loop("How do I run Ollama locally?")
print(answer)
```

---

## Key Components

### `build_index(documents)`

Indexes all documents into a persistent SQLite database. Call this **once** at setup — not at inference time.

```python
index = TextSearchIndex(
    text_fields=["question", "section", "answer"],
    keyword_fields=["course"],
    db_path="faq.db"
)
for doc in documents:
    index.add(doc)
index.close()
```

### `search_documents(question, course)`

Opens the existing `faq.db` and queries it. Reopening the index does **not** reindex — SQLite persists data on disk.

```python
results = index.search(
    question,
    boost_dict={"question": 2.0, "section": 0.5},
    filter_dict={"course": course}
)
```

### Tool Schema

The search function is exposed to the LLM as an OpenAI-compatible JSON tool schema. The same schema works across Ollama, OpenAI, Groq, and Anthropic with minor response-parsing differences.

```python
tools = [
    {
        "type": "function",
        "function": {
            "name": "search_documents",
            "description": "Search the FAQ knowledge base for relevant documents",
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {"type": "string", "description": "The search query"},
                    "course":   {"type": "string", "description": "Course name to filter by"}
                },
                "required": ["question", "course"]
            }
        }
    }
]
```

### Agentic Loop

```python
while True:
    response = client.chat(model=model, messages=messages, tools=tools)

    if response.message.tool_calls:
        for call in response.message.tool_calls:
            args = call.function.arguments
            result = search_documents(args["question"], args["course"])
            messages.append({"role": "tool", "content": str(result)})
    else:
        return response.message.content
```

---

## Design Decisions

| Decision | Reason |
|---|---|
| Separate `build_index()` and `search_documents()` | Indexing is one-time setup; searching is per-request. Combining them re-indexes on every query. |
| System message carries instructions + context | User message carries only the question. Mixing them breaks role separation and multi-turn patterns. |
| Tool schema defined once | Same JSON schema works across Ollama, OpenAI, Groq — only response parsing differs. |
| `RAGBase` class pattern | One-time setup (fetching docs, fitting the index, creating clients) goes outside the class; per-request behavior uses `self.*` inside. |

---

## Requirements

- Python 3.11+
- [Ollama](https://ollama.com) running locally
- A tool-calling capable model (`qwen2.5`, `llama3.1`, or `mistral`)
- No GPU required — runs on CPU

---

## Part of LLM Zoomcamp

This project covers Module 1 of the [DataTalks.Club LLM Zoomcamp](https://github.com/DataTalksClub/llm-zoomcamp):

- Lessons 1–10: Fixed RAG pipeline with keyword search
- Lessons 11–16: Agentic loop with function calling

