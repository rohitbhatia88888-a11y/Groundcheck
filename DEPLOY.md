# Deploying to Fly.io

Deployed and debugged for real on 2026-09-16 (app `groundcheck`, region
`ams`) — the `Dockerfile` build itself worked first try via Fly's remote
builder (no local Docker needed), but the running container hit two real
problems on first deploy, both fixed below and reflected in this repo's
`Dockerfile`/`fly.toml` already:

1. **OOM killed** (`fly.toml`'s original `memory = '1gb'` was too small —
   `torch`/`sentence-transformers` need more, confirmed via `flyctl logs`
   showing `anon-rss:849012kB` at kill time, mid-corpus-embedding). Now `2gb`.
2. **`uv run` at runtime re-downloaded `ruff`** (a dev-only dependency) on
   every cold start — `uv run` re-syncs all dependency groups by default,
   undoing the `--no-dev` used at build time. Fixed: the container's `CMD`
   now invokes `.venv/bin/uvicorn` directly instead of `uv run uvicorn ...`.

If you're deploying a version of this repo from before that fix, redeploy
to pick both up — you'd otherwise hit the same OOM kill.

## One-time setup

```bash
brew install flyctl        # or see fly.io/docs/flyctl/install
flyctl auth login
```

Homebrew installs the binary as `flyctl`, not `fly` — use `flyctl` for every
command below (some Fly docs assume the `fly` alias, which isn't always
present).

`fly.toml` already exists in this repo (currently app name `groundcheck`,
region `ams` — the actual deployed app) — if you're deploying your own
copy, change `app` to something unique to you first (Fly app names are
globally unique), or run `flyctl launch --no-deploy` and let it rewrite
`fly.toml` for you (it preserves the rest of the config — health check,
memory, env — and just fills in `app`/`primary_region`).

Fly's free trial caps a machine at 5 minutes of runtime before stopping it
(`flyctl logs` will say `Trial machine stopping... add a credit card`) — fine
for confirming a deploy works, but add a payment method before relying on
this for anything actually running continuously.

## Secrets

Never put `OPENROUTER_API_KEY` in `fly.toml`, the `Dockerfile`, or a commit —
Fly secrets are the only place it belongs:

```bash
flyctl secrets set OPENROUTER_API_KEY=your-key-here
```

## Deploy

```bash
flyctl deploy
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
- **`flyctl secrets set RAG_CONFIG_PATH=configs/hybrid.yaml`** — restarts the
  machine with the new value, no rebuild needed. Secrets take precedence
  over `[env]` if both are set.

## Things worth knowing before you rely on this

- **Cold starts re-embed the whole corpus, and need real memory to do it.**
  `src/api/app.py`'s startup calls the same `ingest_raw_documents` the eval
  runner uses — parse, chunk, embed, index, from scratch, every time the
  process starts. Locally (on a Mac) this took ~20s for the EU AI Act's
  ~850 chunks; on Fly's `shared-cpu-1x`, image pull alone took ~1 minute,
  and the first real deploy OOM-killed mid-embedding at `memory = '1gb'`
  (see the top of this file) — `2gb` is what's configured now, not yet
  confirmed sufficient end-to-end on Fly's hardware, so watch `flyctl logs`
  on your own first deploy and bump further if it recurs. `fly.toml`
  defaults to `auto_stop_machines = "stop"` (scales to zero when idle) so
  **every request after an idle period pays this cost again**. Set
  `min_machines_running = 1` to keep one warm instance always running
  instead, if that's not acceptable.
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

Optional — `flyctl deploy` builds remotely via Fly's own builder and works
fine without Docker installed at all (confirmed: that's how the real
`groundcheck` deploy above was built). Only useful if you want a faster
local iterate-on-build-errors loop:

```bash
docker build -t rag-eval .
docker run --rm -p 8080:8080 \
  -e OPENROUTER_API_KEY=your-key-here \
  -e RAG_CONFIG_PATH=configs/baseline.yaml \
  rag-eval
curl localhost:8080/health
```
