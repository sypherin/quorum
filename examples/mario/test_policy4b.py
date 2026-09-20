import policy4b as P

ST = {"enemies": [{"type": "goomba", "distance": 1.5, "height": "same level", "moving": "toward mario"},
                  {"type": "goomba", "distance": -3.0, "height": "same level", "moving": "away from mario"},
                  {"type": "powerup", "distance": 2.0, "height": "same level", "moving": "still", "friendly": True}],
      "nearest_pit": None, "nearest_wall": {"distance": 4.0, "height": 2},
      "if_hop_right": {"lands_on": "floor", "tiles_ahead": 4.0}, "if_jump_right": {"lands_on": "pit"}}


def test_pruned_puts_threat_first_and_drops_friendly_and_behind():
    pr = P.pruned(ST)
    assert list(pr)[0] == "enemies_ahead"
    assert [e["type"] for e in pr["enemies_ahead"]] == ["goomba"]


def test_prose_mentions_enemy_wall_and_landings():
    s = P.prose(ST)
    assert "goomba 1.5 tiles ahead" in s and "Wall: 4 tiles ahead, 2 tiles high" in s
    assert "A full jump would land in a pit" in s and "Pit: none" in s
    near = dict(ST, if_hop_right={"lands_on": "floor", "tiles_ahead": 4.0, "hostile_near_landing": True})
    assert "A short hop would land safely." in P.situation(near)   # enemy-at-landing is a code rule, never text


def test_compose_runs_below_threshold_and_jumps_above():
    lo = {"enemy": 0.1, "pit": 0.0, "wall": 0.2, "full": 0.9}
    assert P.compose_ground(lo, ST, 0.5, 0.5)[0] == "run_right"
    hi = {"enemy": 0.8, "pit": 0.0, "wall": 0.2, "full": 0.1}
    assert P.compose_ground(hi, ST, 0.5, 0.5)[0] == "hop_right"


def test_enemy_group_keeps_the_single_enemy_sentence_and_adds_one():
    st = dict(ST, enemies=[{"type": "goomba", "distance": 2.0, "height": "same level", "moving": "toward mario"},
                           {"type": "goomba", "distance": 3.5, "height": "same level", "moving": "toward mario"}])
    s = P.situation(st)
    assert s.startswith("A goomba is close ahead at Mario's level, toward mario. More enemies follow right behind it.")


def test_situation_uses_words_not_numbers_for_proximity():
    s = P.situation(ST)
    assert "close ahead at Mario's level" in s and "1.5" not in s
    assert "A tall (2 tiles) wall is still some way off." in s


def test_policy_is_pure_no_code_override_of_the_models_judgment():
    # a full jump predicted into a pit, an enemy at the hop landing, a stuck flag: none of it may
    # change what the model's two answers say
    st = dict(ST, stuck="no progress", if_hop_right={"lands_on": "floor", "hostile_near_landing": True})
    assert P.compose_ground({"jump": 0.9, "big": 0.9}, st, 0.42, 0.5)[0] == "jump_right"
    assert P.compose_ground({"jump": 0.9, "big": 0.1}, st, 0.42, 0.5)[0] == "hop_right"
    assert P.compose_ground({"jump": 0.1, "big": 0.9}, st, 0.42, 0.5)[0] == "run_right"
    assert P.compose_ground({"jump": 0.9, "big": 0.1, "group": 0.8}, st, 0.42, 0.5)[0] == "jump_right"


def test_air_is_a_model_judgment_too():
    assert P.air_action(0.8)[0] == "pull_left" and P.air_action(0.2)[0] == "keep_right"
    s = P.air_situation({"if_keep_right": {"lands_on": "pit"}, "if_pull_left": {"lands_on": "floor"}})
    assert "Holding right he lands in a pit" in s and "Pulling left would save him" in s
