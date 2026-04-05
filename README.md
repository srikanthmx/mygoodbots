# Multi-Bot AI System

A modular multi-agent AI platform with specialized bots for coding, research, trading analysis, news, and generative AI — accessible via Telegram and a React web chat UI.

## Bots

| Command | Bot | Description |
|---------|-----|-------------|
| `/code` | Coding Bot | AI code generation, review, refactoring — with git push + auto-deploy |
| `/research` | Research Bot | Web research and summaries with citations |
| `/trade` | Trading Bot | Market analysis and technical commentary |
| `/gen` | GenAI Bot | Creative writing, image prompts, style transfer |
| `/news` | News Bot | Financial news digest (also runs every 30 min automatically) |
| `/cron` | Cron Bot | Schedule recurring tasks (owner only) |

## Production Stack (free)

- **Backend** — Render (Docker, free tier)
- **Database** — Supabase (Postgres, free tier)
- **Frontend** — Vercel (Vite/React, free tier)
- **LLM** — Groq (free tier, OpenAI-compatible)
- **Redis** — not required, in-memory fallback used

See [deployment.md](deployment.md) for the full step-by-step guide.

## Local Development

```bash
# 1. Copy and fill in environment variables
cp .env.example .env

# 2. Start infrastructure (PostgreSQL — Redis and Ollama optional)
docker compose up postgres -d

# 3. Install Python dependencies
pip install -e ".[test]"

# 4. Run the backend
uvicorn backend.main:app --reload --port 8000

# 5. Run the frontend (separate terminal)
cd frontend && npm install && npm run dev
```

For local LLM support, start Ollama separately and set `OLLAMA_BASE_URL=http://localhost:11434` in `.env`.
For production, set `OPENAI_BASE_URL=https://api.groq.com/openai/v1` and use your Groq API key.

## Running Tests

```bash
pytest tests/unit -v
```

## Docker Compose (full local stack)

```bash
docker compose up --build
```

Services: `backend` (port 8000), `frontend` (port 3000), `postgres`.

## Environment Variables

Copy `.env.example` to `.env` and fill in your values. Key variables:

| Variable | Description |
|----------|-------------|
| `DATABASE_URL` | Postgres connection string (Supabase in production) |
| `OPENAI_API_KEY` | Groq API key (or OpenAI) |
| `OPENAI_BASE_URL` | `https://api.groq.com/openai/v1` for Groq |
| `TELEGRAM_TOKEN` | From @BotFather |
| `GIT_REMOTE` | Token-embedded GitHub URL for coding bot auto-deploy |
| `REDIS_URL` | Optional — omit to use in-memory fallback |
