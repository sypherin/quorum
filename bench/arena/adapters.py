"""One adapter per system under test. Each takes an arena item and returns
{qid: raw SystemOne answer}; normalize_answer() maps every system's answer to
the same {"label", "probs", "score"} shape that metrics.py scores.

Systems:
  quorum-direct / quorum-cot        in-process quorum.serve.handle_systemone against :8005,
                                    calibration OFF (raw probabilities; calibration is fit
                                    offline, cross-validated) and judgment logging OFF
  quorum-fanout / quorum-fanout-cot the same, one upstream call per question (fanout=true)
  jev                               cloud api.typesafe.ai jev-latest. Never sees a private item.
  laya                              laya.Router defaults: script routing english/multilingual,
                                    auto_task_detection OFF (the generalist)
  laya-td                           laya.Router(auto_task_detection=True): typed-decisions items
                                    go to the checkpoint fine-tuned on that benchmark's train split
  <any>+sl                          that system asked over the shared embedding shortlist of each
                                    high-cardinality choice question (shortlist.py, _data/<task>.sl.json)
"""
from __future__ import annotations

import asyncio
import http.client
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
QUORUM = HERE.parent.parent

from metrics import normalize  # noqa: E402


class PrivateItem(RuntimeError):
    pass


def options_of(q: dict) -> list[str]:
    t = q["type"]
    if t == "noul":
        return ["yes", "no"]
    if t == "choice":
        return list(q["criteria"].keys())
    return [str(i) for i in range(len(q["criteria"]))]


def normalize_answer(ans: dict | None, q: dict) -> dict:
    """Any system's SystemOne answer -> {"label", "probs", "score", "conf"}.

    noul: the contract field `noul` is P(yes); label = P(yes) >= 0.5. A system
          without a distribution falls back to its committed `answer`.
    choice: label = the committed choice; probs over the declared options.
    score: probs over levels; label = argmax level (committed level if no probs);
           score = expected level.
    """
    opts = options_of(q)
    if not isinstance(ans, dict) or ans.get("error"):
        return {"label": None, "probs": None, "score": None, "conf": None,
                "error": (ans or {}).get("error", "no answer") if isinstance(ans, dict) else "no answer"}
    t = q["type"]
    if t == "noul":
        p = ans.get("noul")
        if p is None:
            a = str(ans.get("answer", "")).lower()
            label = "yes" if a in ("yes", "true") else "no" if a in ("no", "false") else None
            return {"label": label, "probs": None, "score": None, "conf": None}
        p = float(p)
        return {"label": "yes" if p >= 0.5 else "no", "probs": {"yes": p, "no": 1 - p},
                "score": None, "conf": max(p, 1 - p)}
    if t == "choice":
        probs = normalize(ans.get("probabilities"), opts)
        label = ans.get("choice")
        if label not in opts:
            label = max(probs, key=probs.get) if probs else None
        return {"label": label, "probs": probs, "score": None,
                "conf": probs.get(label) if probs and label else None}
    probs = normalize(ans.get("probabilities"), opts)
    if probs:
        label = max(probs, key=probs.get)
        score = sum(int(k) * v for k, v in probs.items())
    else:
        s = ans.get("score")
        score = float(s) if s is not None else None
        label = str(int(round(score))) if score is not None else None
    return {"label": label, "probs": probs, "score": score, "conf": probs.get(label) if probs else None}


# ---------------------------------------------------------------------------
# quorum (in-process, the real serve path)
# ---------------------------------------------------------------------------

class Quorum:
    """In-process quorum (serve.handle_systemone) against the live :8005 judge.
    Raw probabilities (no calibration file), no log, no judgment cache: the
    report fits calibration itself under cross-validation, and a cache would
    fake the latency."""

    def __init__(self, cot: bool = False, fanout: bool = False, prose: bool = False,
                 upstream: str | None = None, brain: str | None = None):
        if upstream:
            os.environ["QUORUM_UPSTREAM"] = upstream  # read at import
        os.environ["QUORUM_LOG"] = "off"
        os.environ["QUORUM_CACHE_SIZE"] = "0"  # read at import
        os.environ["QUORUM_CALIBRATION"] = str(HERE / "_no_calibration.json")  # absent => raw
        assert not Path(os.environ["QUORUM_CALIBRATION"]).exists()
        sys.path.insert(0, str(QUORUM))
        from quorum import serve
        assert serve.CACHE_SIZE == 0, "quorum.serve was imported before the arena disabled its cache"
        assert upstream is None or serve.UPSTREAM == upstream, "quorum.serve was imported with another upstream"
        self.serve = serve
        self.cot, self.fanout, self.prose = cot, fanout, prose
        # which weights answer: every row carries it, and a tagged brain must match exactly
        self.brain = upstream_model_id(serve.UPSTREAM)
        if brain is not None and self.brain != brain:
            raise SystemExit(f"{serve.UPSTREAM} serves {self.brain!r}, not the brain {brain!r} this run is named for")

    def ask(self, item: dict) -> tuple[dict, dict]:
        state = item["state"]
        body = {"state": state, "questions": item["questions"], "reasoning": self.cot, "fanout": self.fanout}
        if self.prose:
            try:
                obj = json.loads(state)
            except (TypeError, json.JSONDecodeError):
                obj = None
            if isinstance(obj, (dict, list)):  # only structured states have a prose form
                body["state"], body["state_format"] = obj, "prose"
        status, out = asyncio.run(self.serve.handle_systemone(body))
        if status != 200:
            raise RuntimeError(f"quorum {status}: {str(out)[:300]}")
        meta = {k: out.get(k) for k in ("mode", "prob_method", "usage", "degraded", "calls", "fanout")}
        meta["prose"] = body.get("state_format") == "prose"
        meta["brain"] = self.brain
        return out["answers"], meta


def upstream_model_id(base_url: str) -> str:
    """The model id an OpenAI-style server reports (llama-server: --alias, else the file)."""
    import urllib.request
    with urllib.request.urlopen(base_url + "/v1/models", timeout=10) as r:
        data = json.load(r)["data"]
    if len(data) != 1:
        raise RuntimeError(f"{base_url} serves {len(data)} models; the arena needs exactly one")
    return data[0]["id"]


# ---------------------------------------------------------------------------
# cloud Jev
# ---------------------------------------------------------------------------

JEV_HOST, JEV_PATH = "api.typesafe.ai", "/v1/systemone"
JEV_KEY_FILE = os.path.expanduser("~/.config/typesafe/env")


def _jev_key() -> str:
    k = os.environ.get("TYPESAFE_API_KEY")
    if k:
        return k
    for line in open(JEV_KEY_FILE):
        line = line.strip()
        if line and not line.startswith("#"):
            return line.split("=", 1)[1].strip().strip('"').strip("'") if "=" in line else line
    raise RuntimeError(f"no TypeSafe key in {JEV_KEY_FILE}")


class Jev:
    """Cloud only. Fails loud after retries; never substitutes a local answer."""

    def __init__(self, model: str = "jev-latest", retries: int = 3):
        self.key, self.model, self.retries = _jev_key(), model, retries

    def ask(self, item: dict) -> tuple[dict, dict]:
        if item.get("private"):
            raise PrivateItem(f"{item['id']} is private: never sent to the cloud")
        body = json.dumps({"model": self.model, "state": item["state"], "questions": item["questions"]})
        last = None
        for attempt in range(1, self.retries + 1):
            conn = http.client.HTTPSConnection(JEV_HOST, timeout=60)
            try:
                conn.request("POST", JEV_PATH, body=body, headers={
                    "content-type": "application/json", "authorization": f"Bearer {self.key}"})
                r = conn.getresponse()
                raw = r.read()
                if r.status == 200:
                    out = json.loads(raw)
                    return out["answers"], {"model": out.get("model"), "usage": out.get("usage"),
                                            "request_id": r.getheader("x-typesafe-request-id")}
                last = f"HTTP {r.status}: {raw[:300]!r}"
                if 400 <= r.status < 500 and r.status != 429:
                    break  # a request we built wrong will not fix itself
            except Exception as e:  # noqa: BLE001 - counted and re-raised below
                last = f"{type(e).__name__}: {e}"
            finally:
                conn.close()
            time.sleep(1.5 * attempt)
        raise RuntimeError(f"jev failed: {last}")


# ---------------------------------------------------------------------------
# laya
# ---------------------------------------------------------------------------

_ST_DTYPES = {"BF16": "bfloat16", "F16": "float16", "F32": "float32", "I64": "int64",
              "I32": "int32", "BOOL": "bool", "U8": "uint8"}


def mmap_safetensors(path: str) -> dict:
    """safetensors file -> {name: tensor} backed by a private file mapping, no copy.

    The pages stay file-backed page cache (counted in MemAvailable, evictable) until
    written, and inference never writes weights. safetensors.load_file would copy every
    tensor into anonymous memory instead.
    """
    import mmap
    import struct

    import torch
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_COPY)
    base, out = 8 + n, {}
    for name, info in header.items():
        if name == "__metadata__":
            continue
        dt = getattr(torch, _ST_DTYPES[info["dtype"]])
        lo, hi = info["data_offsets"]
        count = (hi - lo) // torch.empty((), dtype=dt).element_size()
        t = torch.frombuffer(mm, dtype=dt, count=count, offset=base + lo) if count else torch.empty(0, dtype=dt)
        out[name] = t.view(info["shape"])
    return out


class params_on_meta:
    """Build a module with every Parameter on the meta device (no storage) while buffers
    stay real, so buffers computed in __init__ (ModernBERT's rotary inv_freq) survive.
    Same idea as accelerate.init_empty_weights(include_buffers=False). The caller must then
    load_state_dict(strict=True, assign=True) so no parameter is left on meta."""

    def __enter__(self):
        import torch
        from torch import nn
        self._orig = orig = nn.Module.register_parameter

        def reg(module, name, param):
            orig(module, name, param)
            if param is not None:
                p = module._parameters[name]
                module._parameters[name] = type(p)(p.to(torch.device("meta")), **p.__dict__)
        nn.Module.register_parameter = reg
        return self

    def __exit__(self, *exc):
        from torch import nn
        nn.Module.register_parameter = self._orig
        return False


def upcast_on_use(module) -> int:
    """Run every Linear/LayerNorm/Embedding under `module` in fp32 while its weights stay
    stored in their checkpoint dtype (bf16, memory-mapped): each weight is cast at the
    moment it is used and the fp32 copy dies with the op. bf16 -> fp32 is exact, so the
    arithmetic is the stock fp32 path's. Fails loud on any parameter it does not cover.
    Returns the number of patched modules."""
    import torch.nn.functional as F
    from torch import nn

    def f(t):
        return None if t is None else t.float()

    n = 0
    for m in module.modules():
        own = list(m.parameters(recurse=False))
        if not own:
            continue
        if isinstance(m, nn.Linear):
            m.forward = (lambda m: lambda x: F.linear(x, f(m.weight), f(m.bias)))(m)
        elif isinstance(m, nn.LayerNorm):
            m.forward = (lambda m: lambda x: F.layer_norm(
                x, m.normalized_shape, f(m.weight), f(m.bias), m.eps))(m)
        elif isinstance(m, nn.Embedding):
            m.forward = (lambda m: lambda ids: F.embedding(
                ids, m.weight, m.padding_idx, m.max_norm, m.norm_type, m.scale_grad_by_freq,
                m.sparse).float())(m)
        else:
            raise TypeError(f"upcast_on_use: no fp32 path for {type(m).__name__} parameters")
        n += 1
    return n


class Laya:
    """laya's own Router/Agent (laya 0.3.6), with the encoder's weights memory-mapped.

    Why not the stock load: the box sits at ~7.3 GiB MemAvailable with the fleet up and
    the host's memory guard kills below 6. The stock CPU load builds a random-init fp32 model (1.7 GB),
    then copies the checkpoint into it: a probe dropped MemAvailable to 6.18 GiB before the
    first answer. Here:
      1. parameters are built on the meta device, then the mmapped checkpoint is assigned
         (strict=True) in their place: no random-init copy, no anonymous weight copy
      2. the ModernBERT encoder (~0.84 GB, stored bf16 in the checkpoint) stays bf16 on
         disk-backed pages and is upcast per op (upcast_on_use); the small decision heads
         are converted to fp32 outright
    Activations and arithmetic are fp32 exactly as in laya's stock CPU path, so answers
    should match it up to summation order; parity is measured against the earlier stock
    run on the gate heldout (docs/benchmark.md). ARENA_LAYA_LOAD=stock runs laya unmodified.
    """

    def __init__(self, auto_task_detection: bool = False):
        import torch
        torch.set_num_threads(int(os.environ.get("ARENA_TORCH_THREADS", "8")))
        self.load_mode = os.environ.get("ARENA_LAYA_LOAD", "mmap")
        if self.load_mode == "mmap":
            import laya.agent as LA
            orig_build = LA.build_model

            def build(cfg, encoder_dir=None):
                with params_on_meta():
                    m = orig_build(cfg, encoder_dir)
                lsd = m.load_state_dict

                def load_assign(sd, strict=True):
                    r = lsd(sd, strict=strict, assign=True)
                    for name, child in m.named_children():
                        if name != "encoder":
                            child.float()
                    m.temperature = m.temperature.float()
                    upcast_on_use(m.encoder)
                    assert not any(p.is_meta for p in m.parameters())
                    return r
                m.load_state_dict = load_assign
                return m
            LA.build_model = build
            LA.load_file = mmap_safetensors
        elif self.load_mode != "stock":
            raise SystemExit(f"ARENA_LAYA_LOAD must be mmap or stock, not {self.load_mode!r}")
        from laya import Router
        self.router = Router(auto_task_detection=auto_task_detection, max_loaded=1)

    def ask(self, item: dict) -> tuple[dict, dict]:
        state = item["state"]
        if item["task"] == "typed_decisions":
            state = json.loads(state)  # laya's native input for structured state is the object
        out = self.router.predict(state, item["questions"])
        r = out.get("routing") or {}
        return out["answers"], {"routing": r.get("model"), "reason": r.get("reason"), "usage": out.get("usage"),
                                "load": self.load_mode}


def reduce_questions(questions: dict, lists: dict[str, list[str]]) -> dict:
    """Copy of `questions` where each choice question named in `lists` keeps only the
    shortlisted labels, in the question's ORIGINAL order (rank order would hand the
    judge the retriever's confidence as option position). Other questions pass through."""
    out = {}
    for qid, q in questions.items():
        keep = lists.get(qid)
        if keep is None or q["type"] != "choice":
            out[qid] = q
            continue
        missing = [k for k in keep if k not in q["criteria"]]
        if missing:
            raise ValueError(f"shortlist names labels the question does not have: {missing[:3]}")
        kept = set(keep)
        out[qid] = {**q, "criteria": {k: v for k, v in q["criteria"].items() if k in kept}}
    return out


class Shortlisted:
    """Any system, asked over the shared shortlist (shortlist.py). The run still
    normalizes against the FULL question, so a gold label the shortlist dropped
    scores as a miss with zero probability: the retriever's misses are charged."""

    def __init__(self, inner):
        self.inner = inner
        self._files: dict[str, dict] = {}

    def _lists(self, task: str) -> dict:
        if task not in self._files:
            from shortlist import shortlist_path
            p = shortlist_path(task)
            if not p.exists():
                raise SystemExit(f"no shortlist for {task}: run shortlist.py {task} first")
            self._files[task] = json.loads(p.read_text())
        return self._files[task]

    def ask(self, item: dict) -> tuple[dict, dict]:
        f = self._lists(item["task"])
        lists = f["lists"].get(item["id"], {})
        raw, meta = self.inner.ask({**item, "questions": reduce_questions(item["questions"], lists)})
        return raw, {**meta, "shortlist_k": f["k"], "shortlisted": sorted(lists)}


QUORUM_VARIANTS = {
    "quorum-direct": {},
    "quorum-cot": {"cot": True},
    "quorum-fanout": {"fanout": True},
    "quorum-fanout-cot": {"cot": True, "fanout": True},
    "quorum-prose": {"prose": True},
    "quorum-fanout-prose": {"fanout": True, "prose": True},
}


def make(system: str):
    """<system>[@<brain>][+sl]. "@<brain>" asks the server at $ARENA_QUORUM_UPSTREAM,
    which must report exactly that model id (launch it with --alias <brain>)."""
    if system.endswith("+sl"):
        return Shortlisted(make(system[: -len("+sl")]))
    base, _, brain = system.partition("@")
    if base in QUORUM_VARIANTS:
        if not brain:
            return Quorum(**QUORUM_VARIANTS[base])
        upstream = os.environ.get("ARENA_QUORUM_UPSTREAM")
        if not upstream:
            raise SystemExit(f"{system}: set ARENA_QUORUM_UPSTREAM to the server hosting {brain!r}")
        return Quorum(**QUORUM_VARIANTS[base], upstream=upstream, brain=brain)
    if system == "jev":
        return Jev()
    if system == "laya":
        return Laya(auto_task_detection=False)
    if system == "laya-td":
        return Laya(auto_task_detection=True)
    raise SystemExit(f"unknown system {system!r}")
