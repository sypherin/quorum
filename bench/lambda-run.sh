#!/usr/bin/env bash
# lambda-run.sh — run on the Lambda instance (not Strix).
# Job 1: judgment-gate v12 = SFT on approved corpus, then KTO on the
# cloud-Jev-labeled preference set (v12 dataset: single phase prefix,
# generic imperative hints, model-tag negatives).
# Job 2: quorum prob-head distill (Qwen3-0.6B -> cloud Jev noul probabilities).
# Job 3: eval base / SFT-only / KTO on the frozen heldout with FIXED scoring
# (v11's scorer counted a stray think-token line as a wrong verdict, which
#  turned a 35/63 baseline into 1/63 and made the KTO number meaningless).
#
# Checkpoints are scp'd back to Strix after every stage (the persistent FS
# is region-bound and H100 capacity is not, so this run skips the FS —
# per the 16-Aug lesson, mirror off-box as artifacts land).
set -euo pipefail

WORK=/root/work
RUN_TAG="${RUN_TAG:-v12-$(date +%Y%m%d-%H%M)}"
cd "$WORK"
echo "RUN_TAG=$RUN_TAG" > /root/work/RUN_TAG

# env pinned manually before launch (image ships torch cu130 vs driver 570/cu128):
# torch==2.11.0+cu128 torchvision==0.26.0+cu128, pillow>=12, numpy<2 (unsloth_zoo
# needs numpy.Inf), transformers 5.5 / trl 0.24 already present.
python3 -c "import torch, unsloth; assert torch.cuda.is_available(); print('env ok', torch.__version__, unsloth.__version__)"

echo "== GPU =="; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

sync_out() {  # stage to ubuntu's home (root scp is disabled on this image)
  local src="$1" base
  base=$(basename "$src")
  cp -r "$src" "/root/work/staging/$base"
  echo "[stage] $src -> /root/work/staging/$base"
}
mkdir -p /root/work/staging

# ---------- Job 1a: SFT (same recipe as local train.py, 4-bit on NVIDIA) ----
python3 - <<'PY'
import json, os
from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer

MAXLEN = 2048
model, tok = FastLanguageModel.from_pretrained(
    "unsloth/Qwen3-4B", max_seq_length=MAXLEN, load_in_4bit=True)
model = FastLanguageModel.get_peft_model(
    model, r=16, lora_alpha=16, lora_dropout=0.0,
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    use_gradient_checkpointing="unsloth", random_state=42)

ds = load_dataset("json", data_files="sft_train.jsonl", split="train")
def fmt(ex):
    return {"text": tok.apply_chat_template(ex["messages"], tokenize=False)}
ds = ds.map(fmt, remove_columns=list(ds.features))
print("SFT examples:", len(ds), flush=True)

trainer = SFTTrainer(model=model, tokenizer=tok, train_dataset=ds, args=SFTConfig(
    dataset_text_field="text", max_seq_length=MAXLEN, packing=False,
    remove_unused_columns=False, per_device_train_batch_size=4,
    gradient_accumulation_steps=2, warmup_steps=5, num_train_epochs=3,
    learning_rate=2e-4, logging_steps=5, optim="adamw_torch", weight_decay=0.01,
    lr_scheduler_type="linear", seed=42, output_dir="/root/work/out-sft",
    report_to="none"))
print(trainer.train().metrics)
trainer.save_model("/root/work/out-sft/lora")
tok.save_pretrained("/root/work/out-sft/lora")
PY
sync_out /root/work/out-sft/lora

# ---------- Job 1b: KTO on top of the SFT adapter ---------------------------
python3 - <<'PY'
import json, torch
from unsloth import FastLanguageModel
from datasets import Dataset
from peft import PeftModel
from trl import KTOConfig, KTOTrainer

MAXLEN = 2048
model, tok = FastLanguageModel.from_pretrained(
    "unsloth/Qwen3-4B", max_seq_length=MAXLEN, load_in_4bit=True)
model = PeftModel.from_pretrained(model, "/root/work/out-sft/lora", is_trainable=True)

rows = [json.loads(l) for l in open("/root/work/kto_train.jsonl")]

def render(msgs):
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

ds = Dataset.from_dict({
    "prompt": [render(r["prompt"]) for r in rows],
    "completion": [r["completion"] for r in rows],
    "label": [bool(r["label"]) for r in rows],
})
print("KTO examples:", len(ds), "pos:", sum(ds["label"]), flush=True)

trainer = KTOTrainer(model=model, ref_model=None, train_dataset=ds, tokenizer=tok, args=KTOConfig(
    max_length=MAXLEN, per_device_train_batch_size=2, gradient_accumulation_steps=8,
    num_train_epochs=1, learning_rate=5e-6, warmup_steps=20,
    optim="adamw_torch", seed=42, bf16=True,
    output_dir="/root/work/out-kto", report_to="none"))
print(trainer.train().metrics)
trainer.save_model("/root/work/out-kto/lora")
tok.save_pretrained("/root/work/out-kto/lora")
PY
sync_out /root/work/out-kto/lora

# ---------- Job 2: quorum prob-head distill (Qwen3-0.6B -> cloud probs) ----
python3 - <<'PY'
import json, random
import torch, torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

SEED = 20260919
random.seed(SEED); torch.manual_seed(SEED)
rows = [json.loads(l) for l in open("/root/work/distill_items.jsonl")]
random.shuffle(rows)
n = len(rows); nval = max(50, n // 10)
train_rows, val_rows = rows[nval:], rows[:nval]
print(f"distill train {len(train_rows)} val {len(val_rows)}", flush=True)

NAME = "unsloth/Qwen3-0.6B"
tok = AutoTokenizer.from_pretrained(NAME)
if tok.pad_token is None: tok.pad_token = tok.eos_token
base = AutoModelForCausalLM.from_pretrained(NAME, torch_dtype=torch.bfloat16).cuda().eval()
D = base.get_input_embeddings().weight.shape[1]
yes_id = tok.convert_tokens_to_ids("yes")
no_id = tok.convert_tokens_to_ids("no")
print("hidden", D, "yes/no token ids:", yes_id, no_id, flush=True)

class Head(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.h = nn.Sequential(nn.Linear(d, 512), nn.ReLU(),
                               nn.Linear(512, 2), nn.LogSoftmax(-1))
    def forward(self, x): return self.h(x)

def feats(rows_):
    X, Y = [], []
    with torch.no_grad():
        for r in rows_:
            txt = f"{r['state'][:1200]}\nQuestion: {r['question'][:120]}\nAnswer (yes/no):"
            e = tok(txt, truncation=True, max_length=512, return_tensors="pt").to("cuda")
            hs = base.get_input_embeddings()(e["input_ids"])
            am = e["attention_mask"].unsqueeze(-1).to(hs.device)
            pooled = (hs[0] * am[0]).sum(0) / am[0].sum(0)
            X.append(pooled.float().cpu())
            Y.append([1 - float(r["gold"]), float(r["gold"])])  # col1 = P(yes)
    return torch.stack(X).cuda(), torch.tensor(Y).cuda()

Xtr, Ytr = feats(train_rows)
Xva, Yva = feats(val_rows)

head = Head(D).cuda()
opt = torch.optim.AdamW(head.parameters(), lr=1e-3)
lossf = nn.KLDivLoss(reduction="batchmean")
best = (1e9, 1e9)
for ep in range(30):
    head.train()
    perm = torch.randperm(len(Xtr))
    tot = 0.0
    for i in range(0, len(perm), 128):
        idx = perm[i:i+128]
        opt.zero_grad()
        loss = lossf(head(Xtr[idx]), Ytr[idx])
        loss.backward(); opt.step(); tot += loss.item() * len(idx)
    head.eval()
    with torch.no_grad():
        pv = head(Xva).exp()
        val = lossf(pv.clamp_min(1e-8).log(), Yva).item()
        err = (pv[:, 1] - Yva[:, 1]).abs().mean().item()
    print(f"ep{ep:02d} train_kl {tot/len(Xtr):.4f} val_kl {val:.4f} val_mae {err:.4f}", flush=True)
    if val < best[0]:
        best = (val, err)
        torch.save({"head": head.state_dict(), "yes_id": yes_id, "no_id": no_id,
                    "base": NAME, "val_kl": val, "val_mae": err, "hidden": D},
                   "/root/work/prob_head.pt")

print("best (val_kl, val_mae)", best)
PY
sync_out /root/work/prob_head.pt

# ---------- Job 3: eval base / SFT / KTO on the frozen heldout --------------
# FIXED scoring: extract the last "VERDICT: <TAG>" occurrence with a regex.
# The v11 scorer compared gold.split("|")[0] against the raw prediction, so a
# stray think-token line before the verdict scored a correct tag as wrong
# (baseline 35/63 was reported as 1/63 locally; the KTO 17/63 is suspect too).
python3 - <<'PY'
import json, re
import torch
from unsloth import FastLanguageModel
from peft import PeftModel

TAGRE = re.compile(r"VERDICT:\s*([A-Z][A-Z-]*)")

def tag_of(text):
    m = TAGRE.findall(text)
    return m[-1] if m else ""

eval_rows = [json.loads(l)["messages"] for l in open("/root/work/sft_eval.jsonl")]

def run_eval(model, tok, label):
    ok = tot = okn = totn = 0
    misses = []
    for m in eval_rows:
        gold = m[-1]["content"]
        inp = tok.apply_chat_template(m[:2], tokenize=False, add_generation_prompt=True)
        ids = tok(inp, return_tensors="pt").to("cuda")
        out = model.generate(**ids, max_new_tokens=48, do_sample=False)
        pred = tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)
        gt, pt = tag_of(gold), tag_of(pred)
        tot += 1; ok += gt == pt
        if gt != "OK":
            totn += 1; okn += gt == pt
            if gt != pt:
                misses.append((gt, pt))
        torch.cuda.empty_cache()
    print(f"EVAL {label}: all {ok}/{tot}  non-OK {okn}/{totn}", flush=True)
    for gt, pt in misses[:8]:
        print(f"  gold {gt:<16} pred {pt}")

def load(adapter=None):
    model, tok = FastLanguageModel.from_pretrained(
        "unsloth/Qwen3-4B", max_seq_length=2048, load_in_4bit=True, full_finetuning=False)
    if adapter:
        model = PeftModel.from_pretrained(model, adapter)
    FastLanguageModel.for_inference(model)
    return model, tok

for label, adapter in [("base", None),
                       ("sft-only", "/root/work/out-sft/lora"),
                       ("sft+kto", "/root/work/out-kto/lora")]:
    model, tok = load(adapter)
    run_eval(model, tok, label)
    del model
    torch.cuda.empty_cache()
PY

echo "== ALL DONE =="
