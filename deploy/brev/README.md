# quorum on NVIDIA Brev

quorum is a local re-implementation of the SystemOne request contract from TypeSafe's
[Jev](https://typesafe.ai). This folder turns it into a
[Brev Launchable](https://docs.nvidia.com/brev/concepts/launchables.md): one click gives
you a GPU VM running a 4B model behind quorum, with a Jupyter notebook to try it and to
rerun part of the [benchmark](../../docs/benchmark.md). quorum is not affiliated with
TypeSafe.

- [setup.sh](setup.sh) runs once at deploy. It downloads the model at a pinned revision,
  checks its sha256, starts llama-server (CUDA, Docker) on 127.0.0.1:8005 and quorum on
  127.0.0.1:8017 as systemd units that come back after a stop/start, then asks one
  question as a smoke test. It stops with an error if any layer of the model lands on
  the CPU. Its log is `~/workspace/quorum-setup.log`.
- [quickstart.ipynb](quickstart.ipynb) checks the services, sends one request with all
  three question types, measures latency and reruns SST-2, AG News and SST-5. The setup
  script copies it to `~/workspace/quorum-quickstart.ipynb`.

## Creating the Launchable

| field | value |
|---|---|
| Name | `quorum` (cannot be changed later) |
| Default hardware | one L4 (24 GB) is plenty for a 4B model at Q8_0; deployers can pick another NVIDIA GPU |
| Software configuration | VM mode, Jupyter on, setup script below |
| Source | No code files: the setup script clones this repo itself |
| Network | the Jupyter secure link only. Do not open 8005 or 8017: quorum has no auth |
| Launch parameters | optional, see below |
| View access | your organization, until you have deployed it once yourself |

Setup script:

```bash
#!/usr/bin/env bash
set -euo pipefail
curl -fsSL https://raw.githubusercontent.com/sypherin/quorum/master/deploy/brev/setup.sh | bash
```

Launch parameters (both optional):

| name | values | default |
|---|---|---|
| `QUORUM_MODEL` | `judge`: our [judgment-qc-gate-qwen3-4b](https://huggingface.co/AltronisSG/judgment-qc-gate-qwen3-4b-GGUF) (Apache-2.0) · `qwen3-4b-2507`: [Qwen3-4B-Instruct-2507](https://huggingface.co/unsloth/Qwen3-4B-Instruct-2507-GGUF), better for general classification per the benchmark | `judge` |
| `QUORUM_REF` | a branch, tag or commit of this repo | `master` |

## Using it

Open Jupyter from the instance page and run `workspace/quorum-quickstart.ipynb`. To call
quorum from your own machine, forward the port with the
[Brev CLI](https://docs.nvidia.com/brev/cli/connectivity.md) and send requests to
localhost:

```bash
brev port-forward <instance> --port 8017:8017
curl -s 127.0.0.1:8017/v1/systemone -H 'content-type: application/json' -d '{
  "state": "Customer: I was charged twice for one order.",
  "questions": {"urgent": {"type": "noul", "instructions": "Does this express urgency?"}}}'
```

To switch models on a running instance, run the setup script again from a terminal on
it:

```bash
curl -fsSL https://raw.githubusercontent.com/sypherin/quorum/master/deploy/brev/setup.sh | QUORUM_MODEL=qwen3-4b-2507 bash
```

## Cost

Brev bills by the hour while an instance runs. A stopped instance costs storage only,
and a deleted one costs nothing. The model, the code and the notebook live in
`~/workspace`, which survives a stop, so stop the instance when you are done and start it
again later. Check the hourly price in the Brev console before you deploy.

## Status

The notebook and the quorum service ran end to end against llama-server serving the
judge model, and the setup script's user handling ran in an Ubuntu container. The CUDA
path of the setup script (GPU container, layer offload check) has not run on a Brev
instance yet.
