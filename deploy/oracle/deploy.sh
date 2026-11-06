#!/usr/bin/env bash
# Deploy Groundcheck to the Oracle Cloud VM that already runs the Folio demo.
#
#   deploy/oracle/deploy.sh <public-ip> [ssh-key]
#
# Copies the committed tree (git archive HEAD) plus the EU AI Act PDF from data/raw/ (gitignored, so
# not in HEAD), builds the image on the VM, and serves it through Folio's Caddy with automatic HTTPS at
# https://groundcheck.<ip-with-dashes>.sslip.io. The OpenRouter key is asked for once and kept in
# ~/groundcheck/.env on the VM.
set -euo pipefail

IP=${1:?usage: deploy.sh <public-ip> [ssh-key]}
KEY=${2:-$HOME/.ssh/oracle_rag}
HOST="ubuntu@$IP"
SITE="groundcheck.${IP//./-}.sslip.io"
ROOT=$(git rev-parse --show-toplevel)
SSH=(ssh -i "$KEY" -o StrictHostKeyChecking=accept-new "$HOST")

ls "$ROOT"/data/raw/*.pdf >/dev/null 2>&1 || { echo "No PDF in data/raw/; the service would index nothing."; exit 1; }
if [ -n "$(git -C "$ROOT" status --porcelain -- src configs eval/golden_set.json pyproject.toml uv.lock Dockerfile)" ]; then
  echo "Note: uncommitted changes are NOT deployed (only HEAD is copied)."
fi

echo "== Check the VM"
"${SSH[@]}" 'grep -q "import /etc/caddy/sites" ~/rag-demo/Caddyfile' || {
  echo "Folio's Caddy on the VM doesn't import /etc/caddy/sites/*.caddy yet; redeploy Folio first."; exit 1; }

echo "== Copy $(git -C "$ROOT" rev-parse --short HEAD)"
"${SSH[@]}" 'rm -rf ~/groundcheck/src && mkdir -p ~/groundcheck/src ~/rag-demo/sites'
git -C "$ROOT" archive HEAD src configs eval/golden_set.json pyproject.toml uv.lock .python-version Dockerfile .dockerignore |
  "${SSH[@]}" 'tar -x -C ~/groundcheck/src'
# COPYFILE_DISABLE: without it macOS tar adds ._<name>.pdf AppleDouble files, which the parser then
# tries (and fails) to open as PDFs.
COPYFILE_DISABLE=1 tar -C "$ROOT" --no-xattrs -c data/raw | "${SSH[@]}" 'tar -x -C ~/groundcheck/src'
scp -q -i "$KEY" "$ROOT/deploy/oracle/compose.yml" "$HOST:groundcheck/"

echo "== Secrets"
if ! "${SSH[@]}" 'test -f ~/groundcheck/.env'; then
  read -rsp "OpenRouter API key for Groundcheck (one with a credit limit; input hidden): " LLM_KEY; echo
  [ -n "$LLM_KEY" ] || { echo "No key given; stopping."; exit 1; }
  printf 'OPENROUTER_API_KEY=%s\n' "$LLM_KEY" | "${SSH[@]}" 'umask 077 && cat > ~/groundcheck/.env'
  unset LLM_KEY
fi

echo "== Build and start (the first build takes several minutes)"
"${SSH[@]}" 'cd ~/groundcheck && sudo docker compose up -d --build && sudo docker image prune -f >/dev/null'

echo "== Route https://$SITE through Folio's Caddy"
"${SSH[@]}" "cat > ~/rag-demo/sites/groundcheck.caddy" <<EOF
$SITE {
	encode zstd gzip
	reverse_proxy groundcheck:8080 {
		# /query streams the answer; send each chunk as soon as it arrives.
		flush_interval -1
	}
	header {
		Strict-Transport-Security "max-age=31536000"
		X-Content-Type-Options nosniff
		Referrer-Policy strict-origin-when-cross-origin
		-Server
	}
}
EOF
"${SSH[@]}" 'cd ~/rag-demo && sudo docker compose exec -T caddy caddy reload --config /etc/caddy/Caddyfile'

# /health answers 503 {"status":"indexing"} until the corpus is embedded, then 200.
echo "== Waiting for https://$SITE to finish indexing"
for _ in $(seq 1 120); do
  if curl -fsS -o /dev/null "https://$SITE/health" 2>/dev/null; then echo "Live: https://$SITE"; exit 0; fi
  sleep 5
done
echo "Not ready yet. Check: ssh -i $KEY $HOST 'cd groundcheck && sudo docker compose logs --tail 50'"
exit 1
