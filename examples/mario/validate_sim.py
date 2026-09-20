"""Check smb_state.simulate against the real emulator: predict a landing, do the jump, compare.
No Jev calls."""
import asyncio, json
from pathlib import Path
from playwright.async_api import async_playwright
import smb_state as S
from play import Attempt, UA, RIGHT, LEFT, B, A, START


async def land(att, buttons):
    for _ in range(200):
        ram = await att.advance(buttons, 1)
        if S.snapshot(ram)["on_ground"]:
            return ram
    raise RuntimeError("never landed")


async def main():
    async with async_playwright() as p:
        br = await p.chromium.launch(headless=True, args=["--enable-unsafe-swiftshader", "--use-gl=angle", "--use-angle=swiftshader"])
        ctx = await br.new_context(user_agent=UA, viewport={"width": 2200, "height": 900})
        att = Attempt(await ctx.new_page(), None, 0, Path("runs/_validate"), False, 2, 0)
        await att.boot()
        ram = await att.advance([START], 6)
        while not S.controllable(S.snapshot(ram)):
            ram = await att.advance([], 4)
        rows = []
        # One case only: the flat stretch before the first goomba has room for a single clean jump.
        # Hop and pull-left accuracy are checked from real runs instead (decisions.jsonl).
        for label, run_frames, a_frames, air in (("full jump, running", 50, 30, [RIGHT, B]),):
            ram = await att.advance([RIGHT, B], run_frames)
            x0 = S.snapshot(ram)["x"]
            pred = S.simulate(ram, a_frames=a_frames, steer=1)
            ram = await att.advance([RIGHT, B, A], a_frames)
            pred_air = S.simulate(ram, steer=1 if air[0] == RIGHT else -1)
            ram = await land(att, air)
            s = S.snapshot(ram)
            rows.append({"case": label, "x0": x0, "predicted_from_ground": None if air[0] == LEFT else pred["x"],
                         "predicted_from_air": pred_air["x"], "actual": s["x"], "died": S.dying(s)})
            print(json.dumps(rows[-1]), flush=True)
        await br.close()

asyncio.run(main())
