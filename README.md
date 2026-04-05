# Multi-Bot AI System

A modular multi-agent AI platform with specialized bots for coding, research, trading analysis, and generative AI — accessible via Telegram and a React web chat UI.

## Local Development

```bash
# 1. Copy and fill in environment variables
cp .env.example .env

# 2. Start infrastructure (Redis, PostgreSQL, Ollama)
docker compose up redis postgres ollama -d

# 3. Install Python dependencies
pip install -e ".[test]"

# 4. Run the backend
uvicorn backend.main:app --reload --port 8000

# 5. Run the frontend (separate terminal)
cd frontend && npm install && npm run dev
```

## Running Tests

```bash
pytest tests/unit tests/property -v
```

## Docker Compose (full stack)

```bash
docker compose up --build
```

Services: `backend` (port 8000), `frontend` (port 3000), `redis`, `postgres`, `ollama`.

## Deployment

Build and push images, then deploy to any container platform (Cloud Run, Fly.io, VPS):

```bash
docker compose -f docker-compose.yml build
docker compose push
```

Set all variables from `.env.example` as environment secrets in your deployment target.
