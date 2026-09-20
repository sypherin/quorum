"""Pure decoding of Super Mario Bros (NES) RAM into a compact state for Jev.

No I/O, no browser, no network: takes the 2048-byte NES RAM and returns dicts.
Addresses follow the public SMB RAM map (the same ones MarI/O reads).
"""

TILE = 16
ROWS = 13                 # block-buffer rows per page
PAGE_TILES = ROWS * 16    # 208 bytes per page, two pages at 0x500
LOOKAHEAD = 8             # tiles ahead; stays inside the loaded block buffer

# block-buffer ids Mario passes through: empty, coins, flagpole, vine, hidden blocks, axe
NONSOLID = {0x00, 0xC2, 0xC3, 0x24, 0x25, 0x26, 0x5F, 0x60, 0xC5}

ENEMY_NAMES = {
    0x00: "koopa", 0x01: "koopa", 0x02: "buzzy beetle", 0x03: "koopa", 0x04: "koopa",
    0x05: "hammer bro", 0x06: "goomba", 0x07: "blooper", 0x08: "bullet bill",
    0x0A: "cheep cheep", 0x0B: "cheep cheep", 0x0D: "piranha plant", 0x0E: "paratroopa",
    0x0F: "paratroopa", 0x10: "paratroopa", 0x11: "lakitu", 0x12: "spiny", 0x14: "cheep cheep",
    0x15: "bowser flame", 0x2D: "bowser",
}
POWERUP = 0x2E

# GameEngineSubroutine (0x0E)
SUB_CONTROL = 0x08
SUB_DYING = {0x06, 0x0B}
SUB_FLAGPOLE = {0x04, 0x05}


def _s8(b):
    return b - 256 if b > 127 else b


def _half(x):
    """round to the nearest 0.5 tile"""
    return round(x * 2) / 2


def tile(ram, px, row):
    """block-buffer id at level pixel x `px`, buffer row `row` (0 = top, 12 = bottom)"""
    if row < 0 or row >= ROWS:
        return 0
    page = (px // 256) % 2
    col = (px % 256) // TILE
    return ram[0x500 + page * PAGE_TILES + row * 16 + col]


def solid(ram, px, row):
    return tile(ram, px, row) not in NONSOLID


def snapshot(ram):
    """raw game facts the driver needs for lifecycle decisions"""
    return {
        "mode": ram[0x770],          # 0 title/demo, 1 playing, 2 victory, 3 game over
        "sub": ram[0x0E],
        "x": ram[0x6D] * 256 + ram[0x86],
        "y": ram[0xCE],
        "yview": ram[0xB5],          # >1 = fell below the screen
        "on_ground": ram[0x1D] == 0,
        "float": ram[0x1D],          # 3 = sliding down the flagpole
        "xspd": _s8(ram[0x57]),
        "yspd": _s8(ram[0x9F]),
        "lives": ram[0x75A],
        "world": ram[0x75F] + 1,
        "level": ram[0x75C] + 1,
        "power": ram[0x756],         # 0 small, 1 big, 2 fire
        "timer": ram[0x7F8] * 100 + ram[0x7F9] * 10 + ram[0x7FA],
    }


def controllable(s):
    return s["mode"] == 1 and s["sub"] == SUB_CONTROL and s["yview"] <= 1


def dying(s):
    return s["mode"] == 1 and (s["sub"] in SUB_DYING or s["yview"] > 1)


def cleared(s):
    return s["mode"] == 1 and (s["sub"] in SUB_FLAGPOLE or s["float"] == 3)


def feet_row(y):
    """buffer row of the tile under Mario's feet. His box is 32px tall from `y`,
    the buffer starts 32px down the screen, so feet_px - 32 == y."""
    return (y + 8) // TILE


def column(ram, px, frow):
    """what one tile column means for a Mario whose feet are at buffer row `frow`:
    ("wall", height) | ("floor", drop) | ("pit", None)"""
    body = frow - 1
    if 0 <= body < ROWS and solid(ram, px, body):
        h, r = 0, body
        while r >= 0 and solid(ram, px, r):
            h, r = h + 1, r - 1
        return ("wall", h)
    for r in range(max(frow, 0), ROWS):
        if solid(ram, px, r):
            return ("floor", r - frow)
    return ("pit", None)


def _label(kind, n, airborne):
    if kind == "wall":
        return f"wall:{n}"
    if kind == "pit":
        return "pit"
    if airborne:
        return f"floor:{n}"
    return "floor" if n == 0 else f"drop:{n}"


def terrain(ram, mx, frow, airborne):
    """labels for Mario's own column (index 0) and LOOKAHEAD columns ahead, plus
    the pixel x of each column's left edge"""
    base = ((mx + 8) // TILE) * TILE
    cols = []
    for c in range(LOOKAHEAD + 1):
        px = base + c * TILE
        kind, n = column(ram, px, frow)
        cols.append({"px": px, "kind": kind, "n": n, "label": _label(kind, n, airborne)})
    return cols


def _dist(px, mx):
    """tiles from Mario's front edge to a column's near edge, never negative"""
    return max(0.0, _half((px - (mx + TILE)) / TILE))


def nearest_wall(cols, mx):
    for c in cols[1:]:
        if c["kind"] == "wall":
            return {"distance": _dist(c["px"], mx), "height": c["n"]}
    return None


def nearest_pit(cols, mx):
    for i, c in enumerate(cols):
        if c["kind"] == "pit":
            width = 0
            for d in cols[i:]:
                if d["kind"] != "pit":
                    break
                width += 1
            open_ended = i + width == len(cols)
            return {"distance": _dist(c["px"], mx), "width": f"{width}+" if open_ended else width}
    return None


def headroom(ram, mx, frow):
    """empty tiles above Mario's head in the column he is about to enter (capped at 4)"""
    px = ((mx + 8) // TILE) * TILE + TILE
    n, r = 0, frow - 2
    while r >= 0 and n < 4 and not solid(ram, px, r):
        n, r = n + 1, r - 1
    return n


def enemies(ram, mx, my):
    out = []
    mario_feet = my + 32
    for i in range(5):
        if ram[0x0F + i] == 0:
            continue
        if ram[0x1E + i] & 0x20:          # defeated, falling off screen
            continue
        etype = ram[0x16 + i]
        if etype == POWERUP:
            name, hostile = "powerup", False
        elif etype in ENEMY_NAMES:
            name, hostile = ENEMY_NAMES[etype], True
        else:
            continue                      # lifts, flags, springboards: not decisions
        ex = ram[0x6E + i] * 256 + ram[0x87 + i]
        ey = ram[0xCF + i]
        gap_px = ex - (mx + TILE) if ex >= mx else (ex + TILE) - mx
        if abs(gap_px) > (LOOKAHEAD + 2) * TILE:
            continue
        rise = (mario_feet - (ey + 24)) / TILE   # + = enemy higher than Mario's feet
        if abs(rise) < 0.75:
            level = "same level"
        elif rise > 0:
            level = f"{_half(rise):g} tiles above"
        else:
            level = f"{_half(-rise):g} tiles below"
        xs = _s8(ram[0x58 + i])
        if xs == 0:
            moving = "still"
        elif (xs < 0) == (ex >= mx):
            moving = "toward mario"
        else:
            moving = "away from mario"
        e = {"type": name, "distance": _half(gap_px / TILE), "height": level, "moving": moving}
        if not hostile:
            e["friendly"] = True
        out.append(e)
    out.sort(key=lambda e: abs(e["distance"]))
    return out


# ---- landing prediction -------------------------------------------------------
# SMB jump physics by horizontal speed class (|xspd| in 1/16 px per frame):
#   (max |xspd|, initial vy px/frame, gravity while A held and rising, gravity otherwise), gravity in 1/256 px/f^2
JUMP_CLASSES = ((15, -4.0, 0x20, 0x70), (24, -4.0, 0x1E, 0x60), (999, -5.0, 0x28, 0x90))
AIR_ACCEL = 0xE4 / 256 / 16     # px/f^2 while a direction is held
RUN_MAX, BACK_MAX = 2.5, 1.5625
MAX_FALL = 4.0


def simulate(ram, a_frames=0, steer=1, max_frames=200):
    """where Mario comes down if he holds `steer` (+1 right, -1 left, 0 nothing) from now on,
    pressing A for the first `a_frames` frames when he is on the ground.
    Returns {"on": "floor"|"pit"|"unknown", "x": landing x in level px}."""
    s = snapshot(ram)
    x, y = float(s["x"]), float(s["y"])
    vx = s["xspd"] / 16
    body_h = 16 if s["power"] == 0 else 32
    if s["on_ground"] and a_frames:
        cls = next(c for c in JUMP_CLASSES if abs(s["xspd"]) <= c[0])
        vy, g_up, g_down = cls[1], cls[2] / 256, cls[3] / 256
    else:
        a_frames = 0
        vy = s["yspd"] + ram[0x433] / 256
        g_up = g_down = (ram[0x70A] or 0x70) / 256
    for f in range(max_frames):
        vy = min(vy + (g_up if f < a_frames and vy < 0 else g_down), MAX_FALL)
        if steer > 0:
            vx = min(vx + AIR_ACCEL, RUN_MAX)
        elif steer < 0:
            vx = max(vx - AIR_ACCEL, -BACK_MAX)
        nx, ny = x + vx, y + vy
        body_row = (int(ny) + 24 - 32) // TILE
        front = int(nx) + (13 if vx > 0 else 2)
        if vx and solid(ram, front, body_row):
            nx, vx = x, 0.0                       # a wall stops him, he keeps falling
        if vy < 0 and solid(ram, int(nx) + 8, (int(ny) + 32 - body_h - 32) // TILE):
            vy, ny = 0.0, y                       # bonked a block overhead
        if vy > 0:
            row = int(ny) // TILE
            if row >= ROWS:
                return {"on": "pit", "x": int(nx), "frames": f + 1}
            if y <= row * TILE and (solid(ram, int(nx) + 4, row) or solid(ram, int(nx) + 11, row)):
                return {"on": "floor", "x": int(nx), "frames": f + 1}
        x, y = nx, ny
    return {"on": "unknown", "x": int(x), "frames": max_frames}


def _hostile_near_landing(ram, land_x, land_frames, band_tiles=1.25):
    """True if a hostile enemy is predicted within `band_tiles` of `land_x` at the
    frame Mario comes down. Enemies keep their current horizontal speed (walls,
    ledges and turns are ignored — good enough over a jump's ~30-60 frames). This
    is the fact the landing predictor was missing: it says WHERE Mario lands but
    not whether something deadly will be standing there. Coming straight down onto
    a ground enemy stomps it; drifting beside one is fatal — the model is told to
    prefer a clear landing and only descend onto an enemy from directly above."""
    for i in range(5):
        if ram[0x0F + i] == 0:
            continue
        if ram[0x1E + i] & 0x20:              # defeated, falling off screen
            continue
        if ram[0x16 + i] not in ENEMY_NAMES:  # hostile enemies only (skip powerups/props)
            continue
        ex = ram[0x6E + i] * 256 + ram[0x87 + i]
        xs = _s8(ram[0x58 + i])
        ex_land = ex + (xs / 16.0) * land_frames
        if abs(ex_land - land_x) <= band_tiles * TILE:
            return True
    return False


def _landing(ram, mx, **kw):
    r = simulate(ram, **kw)
    out = {"lands_on": r["on"]}
    if r["on"] == "floor":
        out["tiles_ahead"] = _half((r["x"] - mx) / TILE)
        if _hostile_near_landing(ram, r["x"], r["frames"]):
            out["hostile_near_landing"] = True      # key only present when true: enemy-free states are unchanged
    return out


def _speed(xspd):
    a = abs(xspd)
    word = "stopped" if a < 4 else "walking" if a <= 24 else "running"
    if word != "stopped" and xspd < 0:
        word += " left"
    return word


def describe(ram, no_progress=0):
    """the state sent to Jev. Distances are tiles from Mario's front edge;
    enemy distance is negative when the enemy is behind him."""
    s = snapshot(ram)
    mx, my = s["x"], s["y"]
    frow = feet_row(my)
    airborne = not s["on_ground"]
    cols = terrain(ram, mx, frow, airborne)
    mario = {"on_ground": s["on_ground"], "speed": _speed(s["xspd"]),
             "size": ("small", "big", "fire")[min(s["power"], 2)]}
    state = {"mario": mario}
    if airborne:
        mario["vertical"] = "rising" if s["yspd"] < 0 else "falling"
        state["below_and_ahead"] = [c["label"] for c in cols]
        state["if_keep_right"] = _landing(ram, mx, steer=1)
        state["if_pull_left"] = _landing(ram, mx, steer=-1)
    else:
        state["terrain_ahead"] = [c["label"] for c in cols[1:]]
        state["nearest_wall"] = nearest_wall(cols, mx)
        state["if_hop_right"] = _landing(ram, mx, a_frames=8, steer=1)
        state["if_jump_right"] = _landing(ram, mx, a_frames=30, steer=1)
        hr = headroom(ram, mx, frow)
        if hr < 4:
            state["headroom_tiles"] = hr
    state["nearest_pit"] = nearest_pit(cols, mx)
    state["enemies"] = enemies(ram, mx, my)
    if no_progress >= 5:
        state["stuck"] = f"no progress to the right for {5 if no_progress < 15 else 15}+ decisions"
    return state
