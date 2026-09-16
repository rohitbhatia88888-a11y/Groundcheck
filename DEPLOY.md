# Deploying to Fly.io

Deployed and debugged for real on 2026-09-16 (app `groundcheck`, region
`ams`). The `Dockerfile` build itself worked first try via Fly's remote
builder (no local Docker needed) — everything else took four rounds of
`flyctl logs` to actually get healthy. All four fixes are already reflected
in this repo's `Dockerfile`/`fly.toml`:

1. **OOM killed** — `memory = '1gb'` was too small for `torch`/
   `sentence-transformers` (`flyctl logs` showed `anon-rss:849012kB` at kill
   time, mid-embedding). Now `2gb`.
2. **`uv run` at runtime re-downloaded `ruff`** (a dev-only dependency) on
   every cold start — `uv run` re-syncs all dependency groups by default,
   undoing the `--no-dev` used at build time. `CMD` now invokes
   `.venv/bin/uvicorn` directly instead.
3. **Indexing blocked the app from responding at all** — the original
   `src/api/app.py` ran `ingest_raw_documents` inside the blocking
   `lifespan`, so `/health` couldn't answer until the *entire* corpus
   finished embedding. On a slow VM (see #4) that took long enough that
   Fly's orchestrator kept restarting the machine before it ever got there
   — three consecutive restarts, zero uvicorn output, never once reaching
   `Started server process`. Fixed: indexing now runs as an `asyncio`
   background task; the app responds to `/health` within ~2s of boot,
   reporting `{"status":"indexing"}` (503) honestly until the corpus is
   actually ready.
4. **`min_machines_running = 0` let Fly autostop the machine mid-embed** —
   `flyctl logs`: `"App groundcheck has excess capacity, autostopping
   machine ... 0 out of 1 machines left running"`. Background indexing
   doesn't count as traffic to Fly's idle detector, so a cold start could
   be killed before it ever finished, independent of #3's fix. Now `1`.

Even after all four, embedding the EU AI Act's ~850 chunks took **3+
minutes** on `shared-cpu-1x` (vs. ~15-20s on a Mac) — severe CPU throttling
on the shared tier, not a bug. `fly.toml` is now `shared-cpu-2x` /
`cpus = 2`. With `min_machines_running = 1` this slow step only ever runs
once per deploy, not per request, but it's still worth a faster VM so a
fresh deploy doesn't leave the app down for minutes.

If you're deploying a version of this repo from before these fixes,
redeploy to pick up all four — you'd otherwise hit the same failures in
whatever order you're unlucky enough to find them.

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

- **Cold starts re-embed the whole corpus in the background.** `src/api/app.py`
  kicks off the same `ingest_raw_documents` the eval runner uses — parse,
  chunk, embed, index, from scratch — as a background task right after
  boot; `/health` responds within seconds either way, reporting `"indexing"`
  (503) honestly until it's done rather than going silent. Locally this
  takes ~15-20s for the EU AI Act's ~850 chunks; on Fly it took 3+ minutes
  even on `shared-cpu-2x` — budget for that gap after every `flyctl deploy`
  before the app can actually answer queries. `min_machines_running = 1`
  (set — see the top of this file) means this only happens once per deploy,
  not per request after idle.
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
  IP holds true for the single always-on machine as configured now, but if
  you ever scale to more than one machine under load, give `Limiter` a
  shared `storage_uri` (e.g. Redis) so all machines share one counter.
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
