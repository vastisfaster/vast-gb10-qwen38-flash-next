# Qwen3.8-Flash-Next on a single GB10 — Vast.ai template

Serves [`Mia-AiLab/Qwen3.8-Flash-Next-NVFP4`](https://huggingface.co/Mia-AiLab/Qwen3.8-Flash-Next-NVFP4) on one NVIDIA GB10 (DGX Spark) rented through Vast.ai, as an OpenAI-compatible vLLM endpoint.

It is a port of [MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark). That repo's `start.sh` drives Docker on the host, which a Vast instance cannot do because the instance already *is* a container. `qwen38_onstart.sh` performs the same steps from inside the container instead.

> **Status: experimental.** If you own the Spark, run the upstream repo directly. It has fuller memory safety logic and host tuning that a container cannot apply.

## What the script does

1. Clones the upstream repo into `/workspace/qwen38/repo`.
2. Downloads the checkpoint (about 99 GB, resumable).
3. Runs upstream's patch generators against the vLLM files installed in the image, keeping pristine `.orig` copies, and copies the patched files into place.
4. Builds the packed PLE table (one time, about 40 s, about 27 GB).
5. Sets `--gpu-memory-utilization` to (memory limit − 26 GiB) / MemTotal, never above 0.78, and refuses to start if the instance is too small.
6. Starts a watchdog that stops vLLM if `MemAvailable` stays under 6 GiB for 10 s.
7. Runs `vllm serve` with upstream's `.env.sample` defaults: 262k context, FP8 KV cache, MTP speculative decoding (3 tokens), up to 4 concurrent sequences.

Everything lives under `/workspace/qwen38`. The script is safe to re-run; finished steps are skipped.

## Vast.ai template settings

| Field | Value |
|---|---|
| Image path : tag | `vllm/vllm-openai` : `qwen38-flash-next` |
| Docker options | `-p 8888:8888 -e PORT=8888 -e HF_TOKEN= -e API_KEY=CHANGE_ME` |
| Launch mode | SSH, direct connection on |
| Disk space | 160 GB |
| GPU filter | GB10, 1 GPU |

On-start script (replace `<COMMIT_SHA>` with a commit of this repo, so the template always runs a version you can read):

```bash
curl -fsSL https://raw.githubusercontent.com/vastisfaster/vast-gb10-qwen38-flash-next/bfc82c755b36d1d24b6633e3b8a47f6339203bfd/qwen38_onstart.sh -o /root/qwen38_onstart.sh
nohup bash /root/qwen38_onstart.sh >/dev/null 2>&1 &
```

## Environment variables

| Variable | Default | Notes |
|---|---|---|
| `API_KEY` | generated | If empty or `CHANGE_ME`, a random key is written to `/workspace/qwen38/api_key`. The endpoint is never served without a key. |
| `HF_TOKEN` | unset | Only needed if Hugging Face requires it for the checkpoint. |
| `PORT` | `8888` | Must match the `-p` mapping. |
| `MAX_MODEL_LEN` | `262144` | Lower it to leave more memory headroom. |
| `GPU_MEMORY_UTILIZATION` | derived | Overrides step 5. Read the warning below first. |
| `HOST_RESERVE_GIB` | `26` | Upstream's host-side reserve. Do not lower. |

## Connecting

First boot takes the 99 GB download plus about 10–12 minutes of loading. Follow it with:

```bash
tail -f /workspace/qwen38/serve.log      # ready at "Application startup complete"
```

Vast maps container port 8888 to a random external port. Find it on the instance card (IP/ports panel) or inside the instance with `echo $PUBLIC_IPADDR:$VAST_TCP_PORT_8888`. Note that the other mapped port is SSH, not the API.

```bash
curl http://<ip>:<mapped_port>/v1/chat/completions \
  -H "Authorization: Bearer <API_KEY>" -H "Content-Type: application/json" \
  -d '{"model":"qwen3.8-flash-next","messages":[{"role":"user","content":"hi"}]}'
```

The connection is plain HTTP. To keep the key and prompts off the wire, tunnel instead: `ssh -p <ssh_port> root@<ip> -L 8888:localhost:8888`, then use `http://localhost:8888/v1`.

Reasoning is on by default and arrives in a separate `reasoning` field; disable it per request with `"chat_template_kwargs":{"enable_thinking":false}`. Images and video use the standard OpenAI `image_url` / `video_url` content parts.

## Warning: unified memory

On a GB10 the CPU and GPU share one memory pool. Upstream reports that exhausting it **hangs the host kernel** with no OOM kill and no logs. On a rented machine that takes down someone else's hardware. Do not raise `GPU_MEMORY_UTILIZATION`, lower `HOST_RESERVE_GIB`, or disable PLE offload. If you need more headroom, lower `MAX_MODEL_LEN`.

## Troubleshooting

| Symptom | Cause |
|---|---|
| `curl: (1) Received HTTP/0.9` | You hit the SSH port. Use the port mapped to `8888/tcp`. |
| Connection refused | vLLM is still loading, or the script stopped. Check `serve.log` for an `[ERR]` line. |
| `image has no qwen3_8_flash_next model` | Wrong image. Stock vLLM images cannot load this checkpoint. |
| `only 0.xx of memory usable` | Vast gave the container too little RAM for a 99 GB checkpoint. |
| `cannot raise memlock`, then pinned-memory errors | Host limit on locked memory; cannot be changed from inside the instance. |
| No `serve.log` at all | The on-start never ran. Run `bash /root/qwen38_onstart.sh &` by hand. |

## License and credits

All model-specific work (patches, PLE table builder, tuned defaults) is from [MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark), licensed AGPL-3.0-or-later and fetched at run time. This port is released under the same license. vLLM is Apache-2.0. The model checkpoint has its own license terms on Hugging Face; check that they cover your use before serving it to others.
