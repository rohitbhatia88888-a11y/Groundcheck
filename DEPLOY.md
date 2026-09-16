# Deploying to Fly.io

⚠️ **The Dockerfile has not been test-built** — there's no Docker daemon
available in the environment that wrote it. It follows documented,
widely-used patterns (the official `astral-sh/uv` base image, `uv sync
--frozen`), but **run `docker build .` locally yourself before trusting it
in production.** If it fails, the likely culprits are the base image tag
or the two-stage `uv sync` — see comments in the `Dockerfile`.

## One-time setup

```bash
brew install flyctl        # or see fly.io/docs/flyctl/install
fly auth login
```

`fly.toml` already exists in this repo (app name `rag-eval-app`, region
`iad`) — edit `app` to something unique to you before your first deploy, or
run `fly launch --no-deploy` and let it rewrite `fly.toml` for you.

## Secrets

Never put `OPENROUTER_API_KEY` in `fly.toml`, the `Dockerfile`, or a commit —
Fly secrets are the only place it belongs:

```bash
fly secrets set OPENROUTER_API_KEY=your-key-here
```

## Deploy

```bash
fly deploy
```

This builds the `Dockerfile` remotely (or locally with `--local-only`) and
ships it. `/health` is the configured health check — Fly won't route traffic
to a machine until it passes.

## Switching which experiment config is served

`RAG_CONFIG_PATH` (default `configs/baseline.yaml` in `fly.toml`) selects
which `configs/*.yaml` the running service builds its pipeline from — same
mechanism as every other entry point in this project, no code change needed.
Two ways to change it:

- **Edit `fly.toml`'s `[env]` block and redeploy** — simplest, and fine
  since the config path itself isn't sensitive.
- **`fly secrets set RAG_CONFIG_PATH=configs/hybrid.yaml`** — restarts the
  machine with the new value, no rebuild needed. Secrets take precedence
  over `[env]` if both are set.

## Things worth knowing before you rely on this

- **Cold starts re-embed the whole corpus.** `src/api/app.py`'s startup
  calls the same `ingest_raw_documents` the eval runner uses — parse,
  chunk, embed, index, from scratch, every time the process starts. Locally
  this took ~20s for the EU AI Act's ~850 chunks. `fly.toml` defaults to
  `auto_stop_machines = "stop"` (scales to zero when idle) so **every
  request after an idle period pays that cost**. Set
  `min_machines_running = 1` in `fly.toml` to keep one warm instance always
  running instead — for a bigger corpus, or if 20s cold starts aren't
  acceptable, this is the first thing to change.
- **The vector store is in-memory by default** (`location: ":memory:"` in
  most `configs/*.yaml`) — it lives only inside the running process, rebuilt
  every cold start (see above). For a corpus too large to comfortably
  re-embed on every start, point a config's `vector_store.location` at a
  real Qdrant instance instead (Qdrant Cloud, or your own Fly app running
  `qdrant/qdrant` — see `docker-compose.yml` for the image) — then ingestion
  becomes a separate one-time step (`python -m src.ingestion.run`) rather
  than something the API redoes on every boot.
- **Rate limiting is per-machine, not global.** `slowapi`'s default storage
  is in-process memory (see `src/api/app.py`'s `Limiter`) — 10 req/min per
  IP holds true for a single machine, but Fly could run more than one under
  load, each with its own independent counter. Fine for `min_machines_running
  = 0/1` as configured; if you scale out, give `Limiter` a shared
  `storage_uri` (e.g. Redis) so all machines share one counter.
- **`data/raw/` ships inside the image** (see `Dockerfile` — it's gitignored
  but not dockerignored, deliberately). Swapping documents means rebuilding
  and redeploying the image, not just changing a config.

## Testing the image locally

```bash
docker build -t rag-eval .
docker run --rm -p 8080:8080 \
  -e OPENROUTER_API_KEY=your-key-here \
  -e RAG_CONFIG_PATH=configs/baseline.yaml \
  rag-eval
curl localhost:8080/health
```
