#!/usr/bin/env python3
"""A System One model plays Super Mario Bros in the EmulatorJS page at smbgames.be.

Loop: pause the emulator -> read NES RAM -> describe the situation -> ask typed questions ->
press the buttons the answers pick -> repeat. The game only advances between decisions, so
model latency never costs Mario a frame (this is not real-time play).

  --backend cloud   TypeSafe's hosted Jev, one 5-way `choice` question (the original demo)
  --backend local   the quorum shim on 127.0.0.1:8017, nothing leaves the machine
  --policy atomic   for a small local model: atomic yes/no questions over a state written in
                    words, thresholds fitted on the teacher's logs (policy4b.py). No overrides:
                    the model's answers alone pick the action.

  python3 play.py --backend local --policy atomic     # one attempt, headless, video in runs/<ts>/
  python3 play.py --attempts 5                        # fresh games until one clears the level
  python3 play.py --headed                            # watch it
"""
import argparse
import asyncio
import json
import subprocess
import sys
import time
from pathlib import Path

from playwright.async_api import async_playwright

import policy4b as P4
import smb_state as S
from jev_client import Jev

HERE = Path(__file__).resolve().parent
URL = "https://www.smbgames.be/original-smb.php"
UA = ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
      "Chrome/126.0 Safari/537.36")
B, START, LEFT, RIGHT, A = 0, 3, 6, 7, 8
HUD_WIDTH = 460

RULES = (
    "Super Mario Bros on the NES. Mario must stay alive and reach the flagpole far to the right. "
    "Distances are in tiles, from Mario's front edge to the near edge of the thing; an enemy with a "
    "negative distance is behind him. Terrain labels: floor, drop:N (floor N tiles lower, safe), "
    "pit (bottomless, fatal), wall:H (a block or pipe H tiles high that stops him). "
    "Touching a hostile enemy from the side or from below kills Mario; landing on top of it kills the "
    "enemy. A pit kills Mario. A wall only stops him. Enemies marked friendly are power-ups. ")

GROUND_Q = {
    "type": "choice",
    "instructions": RULES + (
        "Mario is on the ground; terrain_ahead lists the next 8 tiles in order. At running speed he "
        "advances about 1 tile per decision. A hop rises 1.5 tiles and lands about 4 tiles ahead. "
        "A full jump rises 4 tiles and lands about 7 tiles ahead; it clears walls up to 4 tiles high "
        "even from right next to them. Once he runs off an edge he can no longer jump. "
        "if_hop_right and if_jump_right are physics predictions of where each jump would come down if "
        "taken right now: lands_on floor (tiles_ahead says how far) or lands_on pit, which is fatal. "
        "The predictions ignore enemies. Pick the controller action for this instant."),
    "criteria": {
        "run_right": "Run right along the ground. Right when nothing needs a jump yet: no hostile enemy "
                     "at his level within 3 tiles ahead, no pit within 1.5 tiles, no wall within 1 tile.",
        "hop_right": "Short low jump to the right. Right for a single hostile enemy at his level 1 to 3 "
                     "tiles ahead, or a wall only 1 tile high within 1 tile. Never when if_hop_right "
                     "lands_on pit.",
        "jump_right": "Full high jump to the right. Right when a pit starts within 1.5 tiles, when a wall "
                      "2 or more tiles high is within 1.5 tiles, when several hostile enemies are close "
                      "together ahead, or when he is stuck against something. Never when if_jump_right "
                      "lands_on pit; if a jump is needed and only one prediction is floor, take that one.",
        "wait": "Release the controls for a moment. Right only when any move now would put Mario into a "
                "hazard, such as an enemy arriving exactly where he would land.",
        "run_left": "Move left. Right only to get away from an enemy about to touch him when a jump "
                    "would not save him, or to back off for a run-up.",
    },
}
AIR_Q = {
    "type": "choice",
    "instructions": RULES + (
        "Mario is in the air and cannot jump again until he lands. below_and_ahead lists the column under "
        "him first, then the next 8 tiles: floor:N means solid ground N tiles below his feet, pit means "
        "nothing to land on. He keeps his forward momentum; pulling left only shortens the jump. "
        "if_keep_right and if_pull_left are physics predictions of where he comes down under each "
        "control: lands_on floor (tiles_ahead says how far) or lands_on pit, which is fatal. The "
        "predictions ignore enemies. Pick the air control for this instant."),
    "criteria": {
        "keep_right": "Hold right. Right whenever if_keep_right lands_on floor, and also when both "
                      "predictions are pit, so that he carries as far as he can. Landing on top of an "
                      "enemy is good.",
        "pull_left": "Pull left to shorten the jump. Right when if_keep_right lands_on pit while "
                     "if_pull_left lands_on floor, or when holding right would bring him down right "
                     "beside a hostile enemy at landing height instead of on top of it.",
    },
}
DANGER_Q = {
    "type": "noul",
    "instructions": "If Mario simply kept running right with no jump, would he die within about one second?",
    "criteria": {"yes": "an enemy, pit or other hazard would kill him within about a second",
                 "no": "the path right ahead is safe for now"},
}

WINDOW = 6   # game frames per ordinary decision
ACTIONS = {
    "run_right": ([RIGHT, B], WINDOW), "hop_right": ([RIGHT, B, A], 8), "jump_right": ([RIGHT, B, A], 30),
    "wait": ([], WINDOW), "run_left": ([LEFT, B], WINDOW),
    "keep_right": ([RIGHT, B], WINDOW), "pull_left": ([LEFT, B], WINDOW),
}


def state_text(state):
    """the JSON we send, one top-level key per line so it fits the HUD"""
    rows = [f' "{k}": {json.dumps(v, separators=(", ", ": "))}' for k, v in state.items()]
    return "{\n" + ",\n".join(rows) + "\n}"


class Attempt:
    def __init__(self, page, jev, num, outdir, capture, cap_every, max_decisions, policy=None):
        self.page, self.jev, self.num = page, jev, num
        self.policy = policy          # None = cloud-style 5-way choice; dict = decomposed small-model policy
        self.dir = outdir / f"attempt_{num:02d}"
        self.frames_dir = self.dir / "frames"
        self.capture, self.cap_every, self.max_decisions = capture, cap_every, max_decisions
        self.clip = None
        self.manifest = []            # (jpg name, game frames it stands for)
        self.game_frames = 0
        self.overshoot = 0
        self.last_buttons = []
        self.log = None

    async def shot(self, frames):
        name = f"f{len(self.manifest):06d}.jpg"
        await self.page.screenshot(path=str(self.frames_dir / name), clip=self.clip, type="jpeg", quality=92)
        self.manifest.append((name, frames))

    async def advance(self, buttons, frames):
        """hold `buttons` for `frames` game frames; returns the RAM afterwards"""
        if A in buttons and A in self.last_buttons:      # a jump needs a fresh A press
            await self._act([b for b in buttons if b != A], 1)
        done, ram = 0, None
        chunk = self.cap_every if self.capture else frames
        while done < frames:
            got, ram = await self._act(buttons, min(chunk, frames - done))
            done += got
        return ram

    async def _act(self, buttons, n):
        r = await self.page.evaluate("([b, n]) => __smb.act(b, n)", [buttons, n])
        self.last_buttons = buttons
        self.game_frames += r["frames"]
        self.overshoot += r["frames"] - n
        if self.capture:
            await self.shot(r["frames"])
        return r["frames"], bytes(r["ram"])

    async def footer(self, s):
        t = self.jev.totals()
        await self.page.evaluate("f => __hud.footer(f)", {
            "attempt": self.num, "world": f'{s["world"]}-{s["level"]}', "x": s["x"],
            "gameTime": round(self.game_frames / 60, 1), "calls": t["calls"],
            "tokens": t["input_tokens"] + t["output_tokens"], "avgLatency": t["avg_latency_ms"]})

    async def boot(self):
        page = self.page
        await page.goto(URL, wait_until="domcontentloaded", timeout=60000)
        await page.wait_for_function("() => !!window.EJS_emulator", timeout=60000)
        await page.locator(".ejs_start_button").click()
        await page.wait_for_function(
            "() => window.EJS_emulator.gameManager && window.EJS_emulator.started", timeout=120000)
        await page.evaluate((HERE / "hud.js").read_text())
        await page.wait_for_timeout(2500)                 # let the title screen settle
        await page.evaluate("() => __smb.run(false)")
        await page.evaluate("() => window.scrollTo(0, 0)")
        clip = await page.evaluate("w => __hud.mount(w)", HUD_WIDTH)
        if getattr(self.jev, "host", None):         # local quorum client: label the HUD truthfully
            await page.evaluate("b => __hud.brand(b[0], b[1], b[2])", [
                "QUORUM · LOCAL SYSTEM ONE", f"POST {self.jev.host}:{self.jev.port}{self.jev.path}", "QUORUM"])
        vw = await page.evaluate("() => document.documentElement.clientWidth")
        if clip["x"] + clip["width"] > vw:
            raise RuntimeError(f"HUD does not fit the viewport ({clip} vs {vw}px): the video would be cropped")
        clip["width"] -= clip["width"] % 2
        clip["height"] -= clip["height"] % 2
        self.clip = clip
        self.frames_dir.mkdir(parents=True, exist_ok=True)
        self.log = open(self.dir / "decisions.jsonl", "w")

    async def decide_atomic(self, state, kind, n):
        """small-model policy: atomic nouls from the model, composition in code (policy4b)"""
        pol = self.policy
        noul = lambda a: a.get("noul") if a.get("noul") is not None else float(a.get("answer") == "yes")
        if kind == "air":
            payload, questions = {"situation": P4.air_situation(state)}, {"pull": P4.Q_PULL_W}
            await self.page.evaluate("p => __hud.thinking(p)", {
                "stateText": state_text(payload), "kind": "1 noul · air", "n": n})
            r = await asyncio.to_thread(self.jev.ask, payload, questions)
            p = {"pull": noul(r["answers"]["pull"])}
            choice, why = P4.air_action(p["pull"])
            r["request_id"] = why
            probs = {"keep_right": 1 - p["pull"], "pull_left": p["pull"]}
        else:
            payload, questions = P4.DESIGNS[pol["design"]]["build"](state)
            await self.page.evaluate("p => __hud.thinking(p)", {
                "stateText": state_text(payload), "kind": f"{len(questions)} nouls · {kind}", "n": n})
            r = await asyncio.to_thread(self.jev.ask, payload, questions)
            p = {q: noul(a) for q, a in r["answers"].items()}
            design = P4.DESIGNS[pol["design"]]
            if "second" in design and (design["jump_score"](p) >= pol["t_jump"]):
                for step in ("second", "third"):                    # only asked when a jump is coming,
                    if step not in design:                          # one atomic question per call
                        continue
                    payload2, questions2 = design[step](state)
                    r2 = await asyncio.to_thread(self.jev.ask, payload2, questions2)
                    p.update({q: noul(a) for q, a in r2["answers"].items()})
                    r["latency_ms"] += r2["latency_ms"]
                    for k in ("input_tokens", "output_tokens"):
                        r["usage"][k] = (r["usage"].get(k) or 0) + (r2["usage"].get(k) or 0)
            choice, why = P4.compose_ground(p, state, pol["t_jump"], pol["t_full"])
            probs = P4.pseudo_probs(p, pol["t_full"])
            r["request_id"] = why
        ans = {"choice": choice, "probabilities": probs, "confidence": probs.get(choice), "nouls": p}
        await self.page.evaluate("p => __hud.decided(p)", {
            "model": r["model"], "choice": choice, "confidence": ans["confidence"], "danger": None,
            "latency": r["latency_ms"], "usage": r["usage"], "requestId": r["request_id"],
            "options": [{"name": k, "p": float(v), "chosen": k == choice} for k, v in probs.items()]})
        return state, kind, choice, ans, None, r

    async def decide(self, ram, n, no_progress):
        state = S.describe(ram, no_progress)
        kind = "air" if not state["mario"]["on_ground"] else "ground"
        if self.policy:
            return await self.decide_atomic(state, kind, n)
        question = AIR_Q if kind == "air" else GROUND_Q
        await self.page.evaluate("p => __hud.thinking(p)", {"stateText": state_text(state), "kind": kind, "n": n})
        r = await asyncio.to_thread(self.jev.ask, state, {"action": question, "danger": DANGER_Q})
        ans = r["answers"]["action"]
        choice = ans["choice"]
        if choice not in question["criteria"]:
            raise RuntimeError(f"Jev returned an option that was not offered: {choice!r}")
        probs = ans.get("probabilities") or {}   # local backend may return null probs
        danger = (r["answers"].get("danger") or {}).get("noul")
        await self.page.evaluate("p => __hud.decided(p)", {
            "model": r["model"], "choice": choice, "confidence": ans.get("confidence"), "danger": danger,
            "latency": r["latency_ms"], "usage": r["usage"], "requestId": r["request_id"],
            "options": [{"name": k, "p": float(probs.get(k, 0)), "chosen": k == choice} for k in question["criteria"]]})
        return state, kind, choice, ans, danger, r

    async def run(self):
        await self.boot()
        ram = await self.advance([START], 6)
        s = S.snapshot(ram)
        result = {"attempt": self.num, "outcome": "max decisions reached", "decisions": 0}
        start_level = None
        n = best_x = no_progress = 0
        t0 = time.time()
        while n < self.max_decisions:
            s = S.snapshot(ram)
            if start_level is None and s["mode"] == 1:
                start_level = (s["world"], s["level"])
            if S.cleared(s):
                result["outcome"] = "cleared"
                await self.page.evaluate("t => __hud.banner(t, false)", f'LEVEL {s["world"]}-{s["level"]} CLEARED · decisions {n}')
                for _ in range(1500 // 4):                  # flag slide, castle walk, score countdown
                    ram = await self.advance([], 4)
                    s2 = S.snapshot(ram)
                    if (s2["world"], s2["level"]) != start_level or s2["mode"] != 1:
                        ram = await self.advance([], 90)
                        break
                break
            if S.dying(s):
                result["outcome"] = f'died at x={s["x"]}' + (" (fell)" if s["yview"] > 1 else "")
                await self.page.evaluate("t => __hud.banner(t, true)", f'MARIO DIED · x={s["x"]} · decision {n}')
                ram = await self.advance([], 120)
                break
            if s["mode"] == 3 or (s["mode"] == 0 and n > 0):
                result["outcome"] = "game over"
                break
            if not S.controllable(s):
                ram = await self.advance([START] if s["mode"] == 0 and self.game_frames % 40 < 6 else [], 4)
                continue
            n += 1
            state, kind, choice, ans, danger, r = await self.decide(ram, n, no_progress)
            buttons, frames = ACTIONS[choice]
            await self.footer(s)
            self.log.write(json.dumps({
                "n": n, "frame": self.game_frames, "x": s["x"], "y": s["y"], "kind": kind, "state": state,
                "choice": choice, "probabilities": ans.get("probabilities"), "confidence": ans.get("confidence"),
                "nouls": ans.get("nouls"),
                "danger": danger, "latency_ms": r["latency_ms"], "usage": r["usage"],
                "request_id": r["request_id"], "model": r["model"]}) + "\n")
            self.log.flush()
            hazards = [f'{k}={json.dumps(state[k])}' for k in ("nearest_wall", "nearest_pit") if state.get(k)]
            hazards += [f'{e["type"]}@{e["distance"]}' for e in state["enemies"][:2]]
            print(f'#{n:<3} x={s["x"]:<5} {kind:<6} -> {choice:<10} p={(ans.get("probabilities") or {}).get(choice, 0):.2f} '
                  f'conf={ans.get("confidence") or 0:.2f} {r["latency_ms"]:>4}ms  {" ".join(hazards)}', flush=True)
            ram = await self.advance(buttons, frames)
            x = S.snapshot(ram)["x"]
            if x > best_x + 4:
                best_x, no_progress = x, 0
            else:
                no_progress += 1
        s = S.snapshot(ram)
        await self.footer(s)
        if self.capture:
            await self.shot(1)
        self.log.close()
        result.update({"decisions": n, "max_x": max(best_x, s["x"]), "game_seconds": round(self.game_frames / 60, 1),
                       "wall_seconds": round(time.time() - t0, 1), "frames_captured": len(self.manifest),
                       "step_overshoot_frames": self.overshoot})
        return result

    def encode(self, out):
        """frames -> mp4 at true game speed: each jpg is shown for the game frames it covers"""
        lines = []
        for name, frames in self.manifest:
            lines += [f"file 'frames/{name}'", f"duration {max(frames, 1) / 60:.5f}"]
        lines += [f"file 'frames/{self.manifest[-1][0]}'", "duration 2.5", f"file 'frames/{self.manifest[-1][0]}'"]
        (self.dir / "frames.txt").write_text("\n".join(lines) + "\n")
        cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", "frames.txt",
               "-vf", "fps=30,format=yuv420p", "-c:v", "libx264", "-preset", "slow", "-crf", "17",
               "-movflags", "+faststart", str(out)]
        subprocess.run(cmd, cwd=self.dir, check=True)


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--attempts", type=int, default=1)
    ap.add_argument("--backend", choices=("cloud", "local"), default="cloud",
                    help="cloud = api.typesafe.ai (Jev); local = quorum shim on :8017")
    ap.add_argument("--quorum-url", default=None, help="local backend URL (default :8017)")
    ap.add_argument("--no-reasoning", action="store_true",
                    help="local backend: turn OFF quorum chain-of-thought (faster, lower quality)")
    ap.add_argument("--policy", choices=("choice", "atomic"), default="choice",
                    help="choice = cloud Jev's single 5-way question; atomic = decomposed nouls + code (small models)")
    ap.add_argument("--design", default="words", choices=sorted(P4.DESIGNS))
    ap.add_argument("--t-jump", type=float, default=0.42)
    ap.add_argument("--t-full", type=float, default=0.5)
    ap.add_argument("--headed", action="store_true")
    ap.add_argument("--no-capture", action="store_true")
    ap.add_argument("--cap-every", type=int, default=2, help="game frames per captured video frame")
    ap.add_argument("--max-decisions", type=int, default=900)
    ap.add_argument("--scale", type=float, default=2.0, help="device scale factor for the capture")
    args = ap.parse_args()

    outdir = HERE / "runs" / time.strftime("%Y%m%d-%H%M%S")
    outdir.mkdir(parents=True)
    if args.backend == "local":
        from quorum_client import Quorum
        jev = Quorum(url=args.quorum_url, reasoning=not args.no_reasoning)
        print(f"[backend] local quorum reasoning={not args.no_reasoning}", flush=True)
    else:
        jev = Jev()
        print("[backend] cloud Jev (api.typesafe.ai)", flush=True)
    results, video = [], None
    async with async_playwright() as p:
        flags = ["--autoplay-policy=no-user-gesture-required"]
        if not args.headed:
            flags += ["--enable-unsafe-swiftshader", "--use-gl=angle", "--use-angle=swiftshader"]
        browser = await p.chromium.launch(headless=not args.headed, args=flags)
        for num in range(1, args.attempts + 1):
            ctx = await browser.new_context(user_agent=UA, viewport={"width": 2200, "height": 900},
                                            device_scale_factor=args.scale)
            page = await ctx.new_page()
            policy = ({"design": args.design, "t_jump": args.t_jump, "t_full": args.t_full}
                      if args.policy == "atomic" else None)
            att = Attempt(page, jev, num, outdir, not args.no_capture, args.cap_every, args.max_decisions, policy)
            print(f"=== attempt {num} ===", flush=True)
            res = await att.run()
            await ctx.close()
            if att.capture and att.manifest:
                out = outdir / f"attempt_{num:02d}_{'CLEARED' if res['outcome'] == 'cleared' else 'failed'}.mp4"
                att.encode(out)
                res["video"] = str(out)
            results.append(res)
            print(json.dumps(res), flush=True)
            if res["outcome"] == "cleared":
                video = res.get("video")
                break
        await browser.close()
    summary = {"results": results, "jev": jev.totals(), "cleared_video": video}
    (outdir / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))
    sys.exit(0 if video or (args.no_capture and results[-1]["outcome"] == "cleared") else 2)


if __name__ == "__main__":
    asyncio.run(main())
