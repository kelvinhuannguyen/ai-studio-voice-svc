#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# fpt-bootstrap.sh — one-shot setup for ai-studio-voice-svc on FPT GPU Ubuntu
#
# Usage (on a fresh FPT GPU Ubuntu 22.04 instance after SSH-ing in):
#
#   curl -fsSL https://raw.githubusercontent.com/kelvinhuannguyen/ai-studio-voice-svc/phase-i-unified-service/scripts/fpt-bootstrap.sh | bash
#
# Or (interactive mode with all prompts):
#
#   curl -fsSL <url> -o bootstrap.sh && bash bootstrap.sh
#
# Idempotent: re-running picks up where the last attempt left off.
# All steps log to /var/log/voice-svc-bootstrap.log for debugging.
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

# ─── Pretty output ──────────────────────────────────────────────────────────
if [[ -t 1 ]]; then
  C_RESET='\033[0m'; C_BOLD='\033[1m'
  C_BLUE='\033[34m'; C_GREEN='\033[32m'; C_YELLOW='\033[33m'; C_RED='\033[31m'; C_CYAN='\033[36m'
else
  C_RESET=''; C_BOLD=''; C_BLUE=''; C_GREEN=''; C_YELLOW=''; C_RED=''; C_CYAN=''
fi

LOG_FILE="/tmp/voice-svc-bootstrap.log"
exec > >(tee -a "$LOG_FILE") 2>&1

step()    { echo -e "\n${C_BOLD}${C_BLUE}━━━ $* ━━━${C_RESET}"; }
ok()      { echo -e "  ${C_GREEN}✓${C_RESET} $*"; }
warn()    { echo -e "  ${C_YELLOW}⚠${C_RESET} $*"; }
fail()    { echo -e "  ${C_RED}✗${C_RESET} $*"; }
info()    { echo -e "  ${C_CYAN}ℹ${C_RESET} $*"; }
ask()     { echo -e -n "${C_BOLD}${C_YELLOW}?${C_RESET} $* "; }

trap 'fail "Bootstrap failed at line $LINENO. Full log: $LOG_FILE"' ERR

# ─── Configuration (override via env vars before running) ──────────────────
REPO_URL="${REPO_URL:-https://github.com/kelvinhuannguyen/ai-studio-voice-svc.git}"
REPO_BRANCH="${REPO_BRANCH:-phase-i-unified-service}"
INSTALL_DIR="${INSTALL_DIR:-$HOME/ai-studio-voice-svc}"
NVIDIA_DRIVER="${NVIDIA_DRIVER:-nvidia-driver-535}"
COMPOSE_PROJECT="${COMPOSE_PROJECT:-voice-svc}"

echo -e "${C_BOLD}╔══════════════════════════════════════════════════════════════╗${C_RESET}"
echo -e "${C_BOLD}║  ai-studio-voice-svc — FPT GPU bootstrap                      ║${C_RESET}"
echo -e "${C_BOLD}║  Branch: $REPO_BRANCH                            ║${C_RESET}"
echo -e "${C_BOLD}╚══════════════════════════════════════════════════════════════╝${C_RESET}"
info "Logging to $LOG_FILE"
info "Install dir: $INSTALL_DIR"

# ─── Step 0: sanity checks ─────────────────────────────────────────────────
step "Step 0 — Sanity checks"

if [[ "$EUID" -eq 0 ]]; then
  fail "Don't run as root. SSH in as 'ubuntu' user, then run this script."
  exit 1
fi
ok "Running as non-root user ($USER)"

if ! grep -q "Ubuntu 22.04" /etc/os-release 2>/dev/null; then
  warn "Not Ubuntu 22.04 (this script tested on 22.04 — proceeding anyway)"
else
  ok "Ubuntu 22.04 detected"
fi

if ! command -v sudo >/dev/null; then
  fail "sudo not installed?? Wrong AMI."
  exit 1
fi

# ─── Step 1: base packages ─────────────────────────────────────────────────
step "Step 1 — Base packages (curl, git, ffmpeg, etc.)"
sudo apt-get update -qq
sudo apt-get install -y -qq \
  curl git ffmpeg vim htop tmux ca-certificates jq openssl
ok "Base packages installed"

# ─── Step 2: NVIDIA driver ─────────────────────────────────────────────────
step "Step 2 — NVIDIA driver"
if command -v nvidia-smi >/dev/null && nvidia-smi >/dev/null 2>&1; then
  ok "NVIDIA driver already installed:"
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader | sed 's/^/    /'
else
  info "Installing $NVIDIA_DRIVER (will take ~3 min)..."
  sudo apt-get install -y -qq "$NVIDIA_DRIVER"
  warn "NVIDIA driver installed — REBOOT REQUIRED."
  ask "Reboot now? (script will resume after you SSH back in and re-run it) [Y/n]:"
  read -r reboot_choice
  reboot_choice=${reboot_choice:-Y}
  if [[ "$reboot_choice" =~ ^[Yy]$ ]]; then
    info "Rebooting in 5s. Re-run this script after SSH-ing back in."
    sleep 5
    sudo reboot
  else
    fail "NVIDIA driver needs reboot before Docker can use GPU. Re-run after sudo reboot."
    exit 1
  fi
fi

# ─── Step 3: Docker ────────────────────────────────────────────────────────
step "Step 3 — Docker"
if command -v docker >/dev/null && docker --version >/dev/null 2>&1; then
  ok "Docker already installed: $(docker --version | head -1)"
else
  info "Installing Docker via get.docker.com..."
  curl -fsSL https://get.docker.com | sudo sh
  sudo usermod -aG docker "$USER"
  ok "Docker installed. You will need to log out and back in for group changes to apply."
  warn "If 'docker ps' fails below, exit + SSH back in + re-run this script."
fi

# Verify Docker daemon
if ! docker ps >/dev/null 2>&1; then
  if groups "$USER" | grep -q docker; then
    sudo systemctl start docker || true
    sleep 2
    if ! docker ps >/dev/null 2>&1; then
      fail "Docker daemon not responding. Try: sudo systemctl status docker"
      exit 1
    fi
  else
    warn "Your shell doesn't have docker group yet. Run: newgrp docker  OR  logout + login + re-run this script"
    exit 1
  fi
fi
ok "Docker daemon ready"

# ─── Step 4: NVIDIA Container Toolkit ──────────────────────────────────────
step "Step 4 — NVIDIA Container Toolkit"
if docker run --rm --gpus all nvidia/cuda:12.1.1-base-ubuntu22.04 nvidia-smi >/dev/null 2>&1; then
  ok "NVIDIA Container Toolkit already working"
else
  info "Installing nvidia-container-toolkit..."
  distribution=$(. /etc/os-release; echo "$ID$VERSION_ID")
  curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
    sudo gpg --batch --yes --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
  curl -s -L "https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list" | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list >/dev/null
  sudo apt-get update -qq
  sudo apt-get install -y -qq nvidia-container-toolkit
  sudo nvidia-ctk runtime configure --runtime=docker
  sudo systemctl restart docker

  info "Verifying GPU passthrough in Docker..."
  if docker run --rm --gpus all nvidia/cuda:12.1.1-base-ubuntu22.04 nvidia-smi >/dev/null 2>&1; then
    ok "GPU passthrough working"
  else
    fail "GPU passthrough not working after install. Check /var/log/syslog and try a reboot."
    exit 1
  fi
fi

# ─── Step 5: clone repo ────────────────────────────────────────────────────
step "Step 5 — Clone ai-studio-voice-svc"
if [[ -d "$INSTALL_DIR/.git" ]]; then
  ok "Repo already cloned at $INSTALL_DIR. Pulling latest..."
  cd "$INSTALL_DIR"
  git fetch origin
  git checkout "$REPO_BRANCH"
  git pull --ff-only origin "$REPO_BRANCH" || warn "Branch diverged — manual merge needed"
else
  info "Cloning $REPO_URL (branch=$REPO_BRANCH)..."
  git clone --branch "$REPO_BRANCH" "$REPO_URL" "$INSTALL_DIR"
  cd "$INSTALL_DIR"
fi
ok "Repo ready at $INSTALL_DIR"

# ─── Step 6: .env (generate token if missing) ──────────────────────────────
step "Step 6 — Configure .env"
ENV_FILE="$INSTALL_DIR/.env"
if [[ -f "$ENV_FILE" ]] && grep -q "^VOICE_SVC_AUTH_TOKEN=." "$ENV_FILE"; then
  ok ".env already configured (token present)"
  TOKEN="$(grep '^VOICE_SVC_AUTH_TOKEN=' "$ENV_FILE" | cut -d= -f2-)"
else
  TOKEN="$(openssl rand -hex 32)"
  cat > "$ENV_FILE" <<EOF
# Auto-generated by fpt-bootstrap.sh — keep this file secret.
VOICE_SVC_AUTH_TOKEN=$TOKEN
ALLOWED_ORIGIN=*
VOICE_SVC_VRAM_CEILING_GB=12.0
EOF
  ok "Generated VOICE_SVC_AUTH_TOKEN (saved to $ENV_FILE)"
fi

# ─── Step 7: Docker build ──────────────────────────────────────────────────
step "Step 7 — Docker build (~20-30 min first time, downloads ~15GB)"
info "This pulls CUDA base image + PyTorch wheels + Whisper/CosyVoice/Demucs."
info "Tail progress with: tail -f $LOG_FILE"
cd "$INSTALL_DIR"
if docker compose --project-name "$COMPOSE_PROJECT" build 2>&1 | tail -50; then
  ok "Docker image built"
else
  fail "Docker build failed. Check $LOG_FILE for details."
  exit 1
fi

# ─── Step 8: Start container ───────────────────────────────────────────────
step "Step 8 — Start voice-svc container"
docker compose --project-name "$COMPOSE_PROJECT" up -d
ok "Container started — waiting for /health to respond (cold start ~30-60s)"

# Poll /health up to 5 min
for i in $(seq 1 30); do
  if curl -fsS http://localhost:8000/health >/dev/null 2>&1; then
    ok "/health responding"
    break
  fi
  echo -n "."
  sleep 10
done
echo ""

if curl -fsS http://localhost:8000/health >/dev/null 2>&1; then
  HEALTH="$(curl -s http://localhost:8000/health)"
  info "Health response:"
  echo "$HEALTH" | jq . 2>/dev/null || echo "$HEALTH"
else
  fail "/health did not respond within 5 min. Check: docker compose logs voice-svc"
  exit 1
fi

# ─── Step 9: Smoke test ────────────────────────────────────────────────────
step "Step 9 — Smoke test /api/render-line-mp3"
TMP_MP3="/tmp/voice-svc-smoke.mp3"
HTTP_CODE=$(curl -sS -o "$TMP_MP3" -w "%{http_code}" \
  -X POST http://localhost:8000/api/render-line-mp3 \
  -H "X-VOICE-SVC-Token: $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"text":"Xin chào, đây là test khởi động voice service.","speaker":"narration"}')

if [[ "$HTTP_CODE" == "200" ]] && [[ -s "$TMP_MP3" ]]; then
  SIZE=$(stat -c %s "$TMP_MP3")
  ok "Smoke test passed — MP3 size: ${SIZE} bytes"
  rm -f "$TMP_MP3"
else
  warn "Smoke test returned HTTP $HTTP_CODE — may be cold start; retry manually:"
  warn "  curl -X POST http://localhost:8000/api/render-line-mp3 -H \"X-VOICE-SVC-Token: \$TOKEN\" -H 'Content-Type: application/json' -d '{\"text\":\"test\"}' -o test.mp3"
fi

# ─── Done ──────────────────────────────────────────────────────────────────
PUBLIC_IP=$(curl -fsS --max-time 5 https://api.ipify.org 2>/dev/null || echo "unknown")

echo ""
echo -e "${C_BOLD}${C_GREEN}╔══════════════════════════════════════════════════════════════╗${C_RESET}"
echo -e "${C_BOLD}${C_GREEN}║  ✓ voice-svc is up                                            ║${C_RESET}"
echo -e "${C_BOLD}${C_GREEN}╚══════════════════════════════════════════════════════════════╝${C_RESET}"
echo ""
echo -e "  ${C_BOLD}Local endpoint:${C_RESET}   http://localhost:8000"
echo -e "  ${C_BOLD}Public IP:${C_RESET}        $PUBLIC_IP"
echo -e "  ${C_BOLD}Auth token:${C_RESET}       $TOKEN"
echo -e "  ${C_BOLD}Health:${C_RESET}           http://localhost:8000/health  (no auth)"
echo ""
echo -e "${C_BOLD}Next steps:${C_RESET}"
echo -e "  1. Add to V2 worker-py/.env on your laptop:"
echo -e "       ${C_CYAN}VOICE_SVC_URL=http://$PUBLIC_IP:8000${C_RESET}    # quick test (insecure)"
echo -e "       ${C_CYAN}VOICE_SVC_TOKEN=$TOKEN${C_RESET}"
echo -e "       ${C_CYAN}VOICE_SVC_ENABLED=true${C_RESET}"
echo ""
echo -e "  2. (Production) Setup Cloudflare Tunnel for HTTPS:"
echo -e "       See: docs/FPT-GPU-DEPLOY-VN.md Section 3.6"
echo ""
echo -e "  3. View live logs:"
echo -e "       ${C_CYAN}cd $INSTALL_DIR && docker compose logs -f voice-svc${C_RESET}"
echo ""
echo -e "  4. Stop instance to save \$\$:"
echo -e "       ${C_CYAN}docker compose down${C_RESET}  (on this instance)"
echo -e "       Then: FPT dashboard → instance → ${C_BOLD}Power Off${C_RESET}"
echo ""
info "Full setup log: $LOG_FILE"
