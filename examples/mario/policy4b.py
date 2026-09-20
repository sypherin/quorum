"""Judgment designs for a SMALL local model (the 4B behind quorum) playing SMB.

Cloud Jev answers one overloaded 5-way `choice` over the whole state. A 4B cannot: it
reads the terrain, ignores the enemies and answers run_right forever. So the judgment is
decomposed the way the TypeSafe docs recommend - atomic yes/no questions over a pruned
state - and code composes the answers into a controller action, with thresholds fitted
against cloud Jev's logged decisions (the teacher). Pure module: no I/O, unit-tested.
"""

# ---------- state pruning / rendering ----------

def hostile_ahead(state):
    return [e for e in state.get("enemies", []) if not e.get("friendly") and e["distance"] >= -0.5]


def pruned(state):
    """only what the ground judgment needs, threat first (a 4B reads the top and drifts)"""
    out = {"enemies_ahead": [{"type": e["type"], "tiles_ahead": max(e["distance"], 0.0),
                              "height": e["height"], "moving": e["moving"]} for e in hostile_ahead(state)[:3]],
           "nearest_pit": state.get("nearest_pit"), "nearest_wall": state.get("nearest_wall"),
           "if_hop_right": state.get("if_hop_right"), "if_jump_right": state.get("if_jump_right")}
    if state.get("stuck"):
        out["stuck"] = state["stuck"]
    return out


def _land(name, p):
    if not p:
        return f"{name}: unknown."
    if p.get("lands_on") == "floor":
        s = f"{name} would land on floor {p.get('tiles_ahead', '?')} tiles ahead"
        return s + (" next to a hostile enemy." if p.get("hostile_near_landing") else ".")
    return f"{name} would land in a {p.get('lands_on')}."


def prose(state):
    en = hostile_ahead(state)[:3]
    parts = []
    if en:
        parts.append("Enemies ahead: " + "; ".join(
            f"a {e['type']} {max(e['distance'], 0.0):g} tiles ahead, {e['height']}, {e['moving']}" for e in en) + ".")
    else:
        parts.append("Enemies ahead: none.")
    pit, wall = state.get("nearest_pit"), state.get("nearest_wall")
    parts.append(f"Pit: {pit['distance']:g} tiles ahead, {pit['width']} tiles wide." if pit else "Pit: none ahead.")
    parts.append(f"Wall: {wall['distance']:g} tiles ahead, {wall['height']} tiles high." if wall else "Wall: none ahead.")
    parts.append(_land("A short hop", state.get("if_hop_right")))
    parts.append(_land("A full jump", state.get("if_jump_right")))
    if state.get("stuck"):
        parts.append("Mario is stuck: " + state["stuck"] + ".")
    return " ".join(parts)


# ---------- atomic questions ----------

Q_ENEMY = {"type": "noul",
           "instructions": "Is a hostile enemy within 3 tiles ahead of Mario at the same level as him?",
           "criteria": {"yes": "an enemy is 3 tiles ahead or closer and at the same level",
                        "no": "no enemy ahead, or it is farther than 3 tiles, or it is above or below him"}}
Q_PIT = {"type": "noul",
         "instructions": "Is a pit within 1.5 tiles ahead of Mario?",
         "criteria": {"yes": "a pit is 1.5 tiles ahead or closer", "no": "no pit, or it is farther than 1.5 tiles"}}
Q_WALL = {"type": "noul",
          "instructions": "Is a wall within 1.5 tiles ahead of Mario?",
          "criteria": {"yes": "a wall is 1.5 tiles ahead or closer", "no": "no wall, or it is farther than 1.5 tiles"}}
Q_FULL = {"type": "noul",
          "instructions": "If Mario jumps now, does he need a full high jump instead of a short hop?",
          "criteria": {"yes": "a pit is near, or a wall 2 or more tiles high is near, or several enemies are close "
                              "together, or a short hop would land in a pit or next to an enemy",
                       "no": "a single enemy or a wall 1 tile high: a short hop is enough"}}
Q_JUMPNOW = {"type": "noul",
             "instructions": "Should Mario jump right now?",
             "criteria": {"yes": "a hostile enemy at his level is within 3 tiles ahead, or a pit is within 1.5 tiles, "
                                 "or a wall is within 1.5 tiles, or he is stuck",
                          "no": "nothing ahead needs a jump yet"}}

# what each atomic question's answer SHOULD be, computed from the state (bench ground truth)
TRUTH = {
    "enemy": lambda s: any(0 <= e["distance"] <= 3 and e["height"] == "same level" for e in hostile_ahead(s)),
    "pit": lambda s: bool(s.get("nearest_pit")) and s["nearest_pit"]["distance"] <= 1.5,
    "wall": lambda s: bool(s.get("nearest_wall")) and s["nearest_wall"]["distance"] <= 1.5,
    "ledge": lambda s: bool(ledge_enemy(s)),
    "group": lambda s: "More enemies follow" in situation(s),
    "big": lambda s: bool(ledge_enemy(s)) or (bool(s.get("nearest_pit")) and s["nearest_pit"]["distance"] <= 1.5) or
                     (bool(s.get("nearest_wall")) and s["nearest_wall"]["distance"] <= 1.5 and s["nearest_wall"]["height"] >= 2),
}

ATOMIC = {"enemy": Q_ENEMY, "pit": Q_PIT, "wall": Q_WALL, "full": Q_FULL}


def _atomic_jump(p):
    return max(p["enemy"], p["pit"], p["wall"])


def threat(state):
    """nearest hostile enemy AHEAD AT MARIO'S LEVEL (code filters by level; the model judges proximity)"""
    t = [e for e in hostile_ahead(state) if e["height"] == "same level" and e["distance"] >= 0]
    return t[0] if t else None


def threat_sentence(state):
    t = threat(state)
    if not t:
        return "No enemy is ahead of Mario at his level."
    return (f"The nearest enemy ahead at Mario's level is a {t['type']}, {t['distance']:g} tiles away, "
            f"{t['moving']}.")


Q_ENEMY_SOLO = {"type": "noul",
                "instructions": "Mario runs right. Is there a hostile enemy within 3 tiles directly ahead of Mario, at his "
                                "own level, that he will collide with if he keeps running right?",
                "criteria": {"yes": "a hostile enemy is within 3 tiles ahead at his level",
                             "no": "no hostile enemy that close ahead at his level"}}
Q_ENEMY_NEAR = {"type": "noul",
                "instructions": "Is the enemy described 3 tiles away or closer?",
                "criteria": {"yes": "the enemy is 3 tiles away or closer",
                             "no": "the enemy is more than 3 tiles away, or there is no enemy"}}

# ---------- situation in words: a 4B cannot compare decimals, it can read categories ----------

def _near(d, close, mid):
    return "right in front of him" if d <= 1 else "close ahead" if d <= close else \
           "still some way off" if d <= mid else "far ahead"


def ledge_enemy(state):
    """hostile enemies on lower ground just past a ledge Mario is about to run off"""
    ta = state.get("terrain_ahead") or []
    if not any(str(t).startswith("drop") for t in ta[:2]):
        return []
    return [e for e in hostile_ahead(state) if "below" in e["height"] and -0.5 <= e["distance"] <= 5]


def situation(state):
    """one short paragraph, proximity in words. Code turns numbers into categories (a HUD,
    not a decision); the model judges what the described situation calls for."""
    parts = []
    t = threat(state)
    level = [e for e in hostile_ahead(state) if e["height"] == "same level" and e["distance"] >= 0]
    if t:
        group = [e for e in level if e["distance"] - t["distance"] <= 3]
        # keep the nearest-enemy sentence identical to the single-enemy case: a 4B reads "A goomba is
        # close ahead" reliably (p~0.8) but "2 enemies ..., the nearest is close ahead" as p=0.09
        parts.append(f"A {t['type']} is {_near(t['distance'], 3, 5)} at Mario's level, {t['moving']}.")
        if len(group) > 1 and t["distance"] <= 3:       # said only when it matters: on a far enemy the
            parts.append("More enemies follow right behind it.")   # extra sentence reads as urgency (p 0.11 -> 0.62)
    else:
        parts.append("No enemy is ahead at Mario's level.")
    pit, wall = state.get("nearest_pit"), state.get("nearest_wall")
    parts.append(f"A pit is {_near(pit['distance'], 1.5, 4)}." if pit else "No pit ahead.")
    if wall:
        size = "low (1 tile)" if wall["height"] <= 1 else f"tall ({wall['height']} tiles)"
        parts.append(f"A {size} wall is {_near(wall['distance'], 1.5, 4)}.")
    else:
        parts.append("No wall ahead.")
    # Wording is part of the design: a 4B's reading shifts when a sentence is added or dropped, so the
    # paragraph is kept to the form verified on every teacher situation. The hop sentence reflects the
    # pit prediction only; an enemy near the landing stays a code rule (in the text it read as "jump now"
    # with the enemy still far ahead, and cost a life at x=2026).
    hop = state.get("if_hop_right") or {}
    parts.append("A short hop would land safely." if hop.get("lands_on") == "floor"
                 else "A short hop would NOT land safely.")
    below = ledge_enemy(state)
    if below:
        n = "An enemy is" if len(below) == 1 else f"{len(below)} enemies are"
        parts.append(f"Mario is on a ledge about to drop down, and {n.lower() if False else n} waiting on the lower ground just past the edge.")
    if state.get("stuck"):
        parts.append("Mario is stuck against something.")
    return " ".join(parts)


Q_JUMP_W = {"type": "noul",
            "instructions": "Does Mario need to jump right now?",
            "criteria": {"yes": "an enemy, a pit or a wall is close ahead or right in front of him, or enemies are "
                                "waiting on the lower ground just past a ledge he is about to leave, or he is stuck",
                         "no": "nothing is close yet: whatever is ahead is still some way off, far ahead, or absent"}}
Q_FULL_W = {"type": "noul",
            "instructions": "Would a short hop be too small here, so that Mario needs a full high jump?",
            "criteria": {"yes": "a pit or a tall wall is close, or several enemies are close together, or a short hop "
                                "would not land safely, or he is stuck",
                         "no": "only a single enemy or a low wall is close, and a short hop lands safely"}}

Q_HOP_W = {"type": "noul",
           "instructions": "Mario is about to jump. Is a short hop enough to get past what is ahead?",
           "criteria": {"yes": "only a single enemy or a low wall is close, and a short hop lands safely",
                        "no": "a pit or a tall wall is close, or several enemies are close together, or a short hop "
                              "would not land safely, or he is stuck: he needs a full high jump"}}

Q_BIG_W = {"type": "noul",
           "instructions": "Is a pit or a tall wall close ahead or right in front of Mario, or are enemies waiting "
                           "below a ledge he is about to leave?",
           "criteria": {"yes": "a pit, or a wall described as tall, is close ahead or right in front of him, or "
                               "enemies are waiting below a ledge he is about to leave",
                        "no": "no pit and no tall wall is close: only enemies, a low wall, or things still some way off"}}

# a fourth "or" inside Q_BIG_W broke readings that were right before (ledge 0.58 -> 0.08): one atomic
# presence question per call, so the enemy group gets its own
Q_GROUP_W = {"type": "noul",
             "instructions": "Do more enemies follow right behind the nearest enemy?",
             "criteria": {"yes": "more enemies follow right behind the nearest one",
                          "no": "there is a single enemy, or no enemy ahead"}}

DESIGNS = {
    # the live small-model policy: one judgment per call, the second only when a jump is coming
    "words": {"build": lambda s: ({"situation": situation(s)}, {"jump": Q_JUMP_W}),
              "second": lambda s: ({"situation": situation(s)}, {"big": Q_BIG_W}),
              "third": lambda s: ({"situation": situation(s)}, {"group": Q_GROUP_W}),
              "jump_score": lambda p: p["jump"], "full_score": lambda p: p.get("big", 0.0)},
    "words-big": {"build": lambda s: ({"situation": situation(s)}, {"big": Q_BIG_W}),
                  "jump_score": lambda p: p["big"], "full_score": lambda p: p["big"]},
    "words-hop": {"build": lambda s: ({"situation": situation(s)}, {"hop": Q_HOP_W}),
                  "jump_score": lambda p: p["hop"], "full_score": lambda p: 1 - p["hop"]},
    "words-jump": {"build": lambda s: ({"situation": situation(s)}, {"jump": Q_JUMP_W}),
                   "jump_score": lambda p: p["jump"]},
    "words-full": {"build": lambda s: ({"situation": situation(s)}, {"full": Q_FULL_W}),
                   "jump_score": lambda p: p["full"], "full_score": lambda p: p["full"]},
    "enemy-solo": {"build": lambda s: ({"enemies": [{"type": e["type"], "distance": e["distance"], "height": e["height"],
                                                     "moving": e["moving"]} for e in hostile_ahead(s)[:3]]},
                                       {"enemy": Q_ENEMY_SOLO}),
                   "jump_score": lambda p: p["enemy"]},
    "enemy-sentence": {"build": lambda s: ({"situation": threat_sentence(s)}, {"enemy": Q_ENEMY_NEAR}),
                       "jump_score": lambda p: p["enemy"]},
    "atomic-json": {"build": lambda s: (pruned(s), ATOMIC), "jump_score": _atomic_jump,
                    "full_score": lambda p: p["full"]},
    "atomic-prose": {"build": lambda s: ({"situation": prose(s)}, ATOMIC), "jump_score": _atomic_jump,
                     "full_score": lambda p: p["full"]},
    "jumpnow-json": {"build": lambda s: (pruned(s), {"jump": Q_JUMPNOW, "full": Q_FULL}),
                     "jump_score": lambda p: p["jump"], "full_score": lambda p: p["full"]},
    "jumpnow-prose": {"build": lambda s: ({"situation": prose(s)}, {"jump": Q_JUMPNOW, "full": Q_FULL}),
                      "jump_score": lambda p: p["jump"], "full_score": lambda p: p["full"]},
}


# ---------- composition: judgments -> controller action (pure) ----------

def _floor(pred):
    return bool(pred) and pred.get("lands_on") == "floor"


def compose_ground(p, state, t_jump, t_full):
    """p = {question id: P(yes)}; returns (action, why). No overrides: the model's two judgments
    alone pick the action, code only maps answers to buttons (the way the cloud harness presses
    whatever Jev chose). `state` is unused on purpose and kept for the call signature."""
    score = p["jump"] if "jump" in p else max(p["enemy"], p["pit"], p["wall"])
    if score < t_jump:
        return "run_right", f"jump {score:.2f} < {t_jump:.2f}"
    full = p["full"] if "full" in p else max(p.get("big", 0.0), p.get("group", 0.0))
    return ("jump_right" if full >= t_full else "hop_right"), \
        f"jump {score:.2f}, big {p.get('big', 0.0):.2f}, group {p.get('group', 0.0):.2f}"


def _lands(pred):
    on = (pred or {}).get("lands_on")
    return "on floor" if on == "floor" else "in a pit" if on == "pit" else "somewhere he cannot see yet"


def air_situation(state):
    """facts from the landing predictor, with the relation between the two controls spelled out:
    a 4B reads "pulling left would save him" but cannot combine two separate landing sentences"""
    keep, pull = (state.get("if_keep_right") or {}).get("lands_on"), (state.get("if_pull_left") or {}).get("lands_on")
    if keep != "pit":
        return f"Mario is in the air. Holding right he lands {_lands(state.get('if_keep_right'))}. No pit under his path."
    saves = "Pulling left would save him: he lands on floor." if pull == "floor" else \
            "Pulling left ends in a pit too, so pulling left is useless."      # no negation: a 4B misreads "would not"
    return f"Mario is in the air. Holding right he lands in a pit. {saves}"


Q_PULL_W = {"type": "noul",
            "instructions": "Would pulling left save Mario from a pit?",
            "criteria": {"yes": "holding right lands him in a pit and pulling left would save him",
                         "no": "there is no pit under his path, or pulling left would not save him"}}

AIR_TRUTH = lambda s: (s.get("if_keep_right") or {}).get("lands_on") == "pit" and \
                      (s.get("if_pull_left") or {}).get("lands_on") == "floor"


T_PULL = 0.70      # fitted in the gap: the one "pull left" situation scores 0.77, every other one <= 0.63


def air_action(p_pull, t_pull=T_PULL):
    """the air judgment is the model's too: pull left only if it says so"""
    return ("pull_left" if p_pull >= t_pull else "keep_right"), f"pull {p_pull:.2f}"


def pseudo_probs(p, t_full):
    """action distribution for the HUD from the two judgments"""
    j = p["jump"] if "jump" in p else max(p["enemy"], p["pit"], p["wall"])
    f = p["full"] if "full" in p else max(p.get("big", 0.0), p.get("group", 0.0))
    return {"run_right": 1 - j, "hop_right": j * (1 - f), "jump_right": j * f}
