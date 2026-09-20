#!/bin/bash
# Vast.ai onstart script: Qwen3.8-Flash-Next-NVFP4 on a single GB10.
# Ports the repo's start.sh (which needs host Docker) to run INSIDE the
# vllm/vllm-openai:qwen38-flash-next container. Uses the .env.sample defaults.
# Re-runnable: finished steps are skipped. Log: /workspace/qwen38/serve.log
set -uo pipefail

W=/workspace/qwen38
REPO=$W/repo
MODEL_ID="Mia-AiLab/Qwen3.8-Flash-Next-NVFP4"
PORT="${PORT:-8888}"
HOST_RESERVE_GIB="${HOST_RESERVE_GIB:-26}"      # repo default; do not lower
MAX_MODEL_LEN="${MAX_MODEL_LEN:-262144}"
export HF_HOME=$W/hf
mkdir -p "$W" "$HF_HOME"
exec >>"$W/serve.log" 2>&1
echo "=== $(date) onstart ==="

die() { echo "[ERR] $*"; exit 1; }
[[ "$(uname -m)" == "aarch64" ]] || echo "[WARN] not aarch64 - these patches target GB10"

# Public-template safety: never serve an open or default-keyed endpoint.
[[ -n "${HF_TOKEN:-}" ]] || unset HF_TOKEN
if [[ -z "${API_KEY:-}" || "$API_KEY" == "CHANGE_ME" ]]; then
    [[ -s $W/api_key ]] || python3 -c "import secrets;print('sk-'+secrets.token_urlsafe(24))" > "$W/api_key"
    API_KEY=$(cat "$W/api_key"); chmod 600 "$W/api_key"
    echo "[INFO] no API_KEY set - generated one, saved in $W/api_key"
fi

# 1. Repo
[[ -d $REPO/.git ]] || git clone --depth 1 \
    https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark "$REPO" || die "clone failed"
F=$REPO/files

# 2. Checkpoint (~99 GB, resumable)
python3 - <<PY || die "download failed"
from huggingface_hub import snapshot_download
snapshot_download("$MODEL_ID")
PY
SNAP_DIR=$(ls -d "$HF_HOME"/hub/models--Mia-AiLab--Qwen3.8-Flash-Next-NVFP4/snapshots/*/ | head -1)
SNAP_DIR=${SNAP_DIR%/}
[[ -f $SNAP_DIR/config.json ]] || die "no snapshot"

# 3. Patches: same generators as start.sh, but originals are copied from the
#    live install (we are already inside the image) and results copied back.
VLLM_PKG=$(python3 -c "import importlib.util as u;print(u.find_spec('vllm').submodule_search_locations[0])")
NV=$VLLM_PKG/models/qwen3_8_flash_next/nvidia
[[ -d $NV ]] || die "image has no qwen3_8_flash_next model - wrong image?"
mkdir -p "$F/ple_offload/orig"
# "<installed file>|<.orig the generator reads>|<file the generator writes>"
MAP=(
"$NV/ple_layer.py|$F/ple_layer_patched.py.orig|$F/ple_layer_patched.py"
"$VLLM_PKG/model_executor/layers/quantization/modelopt.py|$F/modelopt_patched.py.orig|$F/modelopt_patched.py"
"$NV/ops/qsa.py|$F/qsa_ops_patched.py.orig|$F/qsa_ops_patched.py"
"$NV/qsa.py|$F/qsa_nvidia_patched.py.orig|$F/qsa_nvidia_patched.py"
"$NV/mtp.py|$F/mtp_patched.py.orig|$F/mtp_patched.py"
"$VLLM_PKG/model_executor/layers/ple_offload_layer.py|$F/ple_offload/orig/ple_offload_layer.py|$F/ple_offload/ple_offload_layer.py"
"$VLLM_PKG/v1/ple_offload/connector.py|$F/ple_offload/orig/connector.py|$F/ple_offload/connector.py"
"$VLLM_PKG/v1/ple_offload/worker.py|$F/ple_offload/orig/worker.py|$F/ple_offload/worker.py"
"$VLLM_PKG/v1/ple_offload/protocol.py|$F/ple_offload/orig/protocol.py|$F/ple_offload/protocol.py"
)
for m in "${MAP[@]}"; do IFS='|' read -r live orig _ <<<"$m"
    [[ -f $orig ]] || cp "$live" "$orig" || die "missing $live"   # pristine copy, taken once
done
for p in patch_ple_layer patch_modelopt_mxfp8 patch_qsa_fp8_kv patch_mtp_draft_vocab patch_ple_offload; do
    python3 "$F/$p.py" || die "$p failed"
done
for m in "${MAP[@]}"; do IFS='|' read -r live _ out <<<"$m"
    [[ -f $out ]] || die "generator did not write $out"
    cp "$out" "$live"
done
echo "[OK] patches applied"

# 4. Packed PLE table (one-time, ~40 s, ~27 GB)
PLE_DIR=$W/ple_cache/Mia-AiLab--Qwen3.8-Flash-Next-NVFP4
if ! ls "$PLE_DIR"/*.packed_u8 >/dev/null 2>&1; then
    mkdir -p "$PLE_DIR"
    python3 -u "$F/build_ple_packed_table.py" "$SNAP_DIR" "$PLE_DIR" || die "PLE build failed"
fi

# 5. Memory budget: start.sh's cap (MemTotal - HOST_RESERVE), also honouring
#    any cgroup limit Vast put on this container. Hard ceiling 0.78.
GMU=$(python3 - <<PY
import math
m={l.split(':')[0]:int(l.split()[1]) for l in open('/proc/meminfo') if ':' in l}
tot=m['MemTotal']/1048576
lim=tot
for p in ('/sys/fs/cgroup/memory.max','/sys/fs/cgroup/memory/memory.limit_in_bytes'):
    try:
        v=open(p).read().strip()
        if v!='max': lim=min(lim,int(v)/2**30)
    except OSError: pass
print(min(0.78, math.floor(max(lim-$HOST_RESERVE_GIB,0)/tot*1000)/1000))
PY
)
GMU="${GPU_MEMORY_UTILIZATION:-$GMU}"
echo "[INFO] gpu-memory-utilization=$GMU"
python3 -c "import sys;sys.exit(0 if $GMU>=0.60 else 1)" || die "only $GMU of memory usable - instance too small for a 99 GB checkpoint"

# 6. Watchdog: exhausting unified memory HANGS the host (no OOM kill).
( n=0; while sleep 2; do
    a=$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo)
    if (( a < 6 )); then n=$((n+1)); else n=0; fi
    if (( n >= 5 )); then echo "[WATCHDOG] MemAvailable ${a} GiB - stopping vLLM"
        pkill -TERM -f "vllm serve"; sleep 30; pkill -KILL -f "vllm serve"; exit; fi
  done ) &

# 7. Serve (args = start.sh output for .env.sample defaults)
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export VLLM_PLE_CPU_OFFLOAD=1                     # never turn off at TP=1
export VLLM_PLE_PACKED_TABLE_DIR=$PLE_DIR
export VLLM_PLE_OFFLOAD_STEP_TIMEOUT=300 MAX_JOBS=2 FLASHINFER_NVCC_THREADS=1
export VLLM_MTP_DRAFT_VOCAB=$F/draft_vocab_en_code_47k.txt
ulimit -l unlimited 2>/dev/null || echo "[WARN] cannot raise memlock (ulimit -l = $(ulimit -l))"
ulimit -s 65536 2>/dev/null || true

exec vllm serve "$MODEL_ID" \
  --enable-prompt-tokens-details \
  --served-model-name qwen3.8-flash-next \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization "$GMU" \
  --max-num-seqs 4 --max-num-batched-tokens 2048 \
  --max-model-len "$MAX_MODEL_LEN" \
  --kv-cache-dtype fp8 --mamba-ssm-cache-dtype bfloat16 \
  --load-format safetensors --safetensors-load-strategy lazy \
  --enable-chunked-prefill \
  --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_coder \
  --distributed-executor-backend mp \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3,"use_local_argmax_reduction":true}' \
  --compilation-config '{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[4,8,12,16]}' \
  --host 0.0.0.0 --port "$PORT" \
  ${API_KEY:+--api-key "$API_KEY"}
