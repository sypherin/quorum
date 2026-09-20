"""Unit tests for smb_state on synthetic RAM. Run: pytest -q"""
import smb_state as S

GROUND_Y = 176          # Mario's y when standing on the normal floor (rows 11-12)
BRICK = 0x54


def make_ram(mx=40, my=GROUND_Y, on_ground=True):
    ram = bytearray(2048)
    for page in (0, 1):
        for row in (11, 12):
            for col in range(16):
                ram[0x500 + page * 208 + row * 16 + col] = BRICK
    ram[0x6D], ram[0x86] = mx // 256, mx % 256
    ram[0xCE] = my
    ram[0xB5] = 1
    ram[0x1D] = 0 if on_ground else 1
    ram[0x770], ram[0x0E] = 1, 8
    return ram


def put(ram, level_col, row, tid=BRICK):
    px = level_col * 16
    ram[0x500 + ((px // 256) % 2) * 208 + row * 16 + (px % 256) // 16] = tid


def add_enemy(ram, slot, ex, ey=184, etype=0x06, xspd=-8):
    ram[0x0F + slot] = 1
    ram[0x16 + slot] = etype
    ram[0x6E + slot], ram[0x87 + slot] = ex // 256, ex % 256
    ram[0xCF + slot] = ey
    ram[0x58 + slot] = xspd & 0xFF


def test_feet_row_on_normal_floor():
    assert S.feet_row(GROUND_Y) == 11


def test_flat_ground_is_all_floor():
    st = S.describe(make_ram())
    assert st["terrain_ahead"] == ["floor"] * 8
    assert st["nearest_wall"] is None and st["nearest_pit"] is None
    assert st["enemies"] == []


def test_pipe_is_a_wall_with_height_and_distance():
    ram = make_ram(mx=40)                 # centre 48 -> column 3, front edge at 56
    for row in (9, 10):
        put(ram, 6, row)                  # 2-high pipe, left edge at px 96
    st = S.describe(ram)
    assert st["terrain_ahead"][2] == "wall:2"
    assert st["nearest_wall"] == {"distance": 2.5, "height": 2}


def test_pit_distance_and_width():
    ram = make_ram(mx=40)
    for col in (5, 6):
        put(ram, col, 11, 0), put(ram, col, 12, 0)
    st = S.describe(ram)
    assert st["terrain_ahead"][1:3] == ["pit", "pit"]
    assert st["nearest_pit"] == {"distance": 1.5, "width": 2}


def test_pit_running_off_the_lookahead_is_open_ended():
    ram = make_ram(mx=40)
    for col in range(9, 16):
        put(ram, col, 11, 0), put(ram, col, 12, 0)
    assert S.describe(ram)["nearest_pit"]["width"] == "3+"


def test_coins_and_hidden_blocks_are_not_walls():
    ram = make_ram(mx=40)
    put(ram, 5, 10, 0xC2)
    put(ram, 6, 10, 0x5F)
    assert S.describe(ram)["nearest_wall"] is None


def test_floating_blocks_overhead_are_not_walls():
    ram = make_ram(mx=40)
    put(ram, 5, 7)
    st = S.describe(ram)
    assert st["nearest_wall"] is None
    assert st["terrain_ahead"][1] == "floor"


def test_standing_on_a_ledge_sees_a_drop_not_a_pit():
    ram = make_ram(mx=40, my=GROUND_Y - 32)       # on top of a 2-high platform
    for col in (2, 3):
        for row in (9, 10):
            put(ram, col, row)
    st = S.describe(ram)
    assert st["terrain_ahead"][0] == "drop:2"
    assert st["nearest_pit"] is None


def test_goomba_ahead_at_same_level_moving_toward_mario():
    ram = make_ram(mx=40)
    add_enemy(ram, 0, ex=40 + 16 + 40)            # 2.5 tiles of clear space
    e = S.describe(ram)["enemies"]
    assert e == [{"type": "goomba", "distance": 2.5, "height": "same level", "moving": "toward mario"}]


def test_enemy_behind_has_negative_distance_and_defeated_is_dropped():
    ram = make_ram(mx=200)
    add_enemy(ram, 0, ex=200 - 16 - 32, xspd=8)   # 2 tiles behind, walking right = toward
    add_enemy(ram, 1, ex=260)
    ram[0x1E + 1] = 0x20
    e = S.describe(ram)["enemies"]
    assert len(e) == 1
    assert e[0]["distance"] == -2.0 and e[0]["moving"] == "toward mario"


def test_powerup_is_marked_friendly_and_lifts_are_skipped():
    ram = make_ram(mx=40)
    add_enemy(ram, 0, ex=120, etype=0x2E, xspd=8)
    add_enemy(ram, 1, ex=130, etype=0x24)
    e = S.describe(ram)["enemies"]
    assert [x["type"] for x in e] == ["powerup"] and e[0]["friendly"] is True


def test_airborne_state_reports_floor_depth_and_pits():
    ram = make_ram(mx=40, my=GROUND_Y - 48, on_ground=False)
    ram[0x9F] = 0xFC                              # rising
    put(ram, 4, 11, 0), put(ram, 4, 12, 0)
    st = S.describe(ram)
    assert st["mario"]["vertical"] == "rising" and "terrain_ahead" not in st
    assert st["below_and_ahead"][0] == "floor:3"
    assert st["below_and_ahead"][1] == "pit"


def test_block_buffer_wraps_across_pages():
    ram = make_ram(mx=500)                        # centre 508 -> page 1, column 15
    put(ram, 32, 10)                              # level col 32 -> page 0, col 0
    assert S.describe(ram)["terrain_ahead"][0] == "wall:1"


def test_stuck_flag_only_after_five_decisions():
    assert "stuck" not in S.describe(make_ram(), no_progress=4)
    assert "5+" in S.describe(make_ram(), no_progress=5)["stuck"]
    assert "15+" in S.describe(make_ram(), no_progress=20)["stuck"]


def test_lifecycle_predicates():
    ram = make_ram()
    s = S.snapshot(ram)
    assert S.controllable(s) and not S.dying(s) and not S.cleared(s)
    ram[0x0E] = 0x0B
    assert S.dying(S.snapshot(ram)) and not S.controllable(S.snapshot(ram))
    ram[0x0E], ram[0xB5] = 8, 2
    assert S.dying(S.snapshot(ram))
    ram[0xB5], ram[0x0E] = 1, 4
    assert S.cleared(S.snapshot(ram))


def _running(ram, xspd=40):
    ram[0x57] = xspd & 0xFF
    return ram


def test_full_jump_on_flat_ground_lands_on_floor_further_than_a_hop():
    ram = _running(make_ram(mx=40))
    st = S.describe(ram)
    hop, full = st["if_hop_right"], st["if_jump_right"]
    assert hop["lands_on"] == full["lands_on"] == "floor"
    assert 2.5 <= hop["tiles_ahead"] < full["tiles_ahead"] <= 10


def test_jump_into_a_wide_pit_is_predicted_as_pit():
    ram = _running(make_ram(mx=40))
    for page in (0, 1):
        for col in range(16):
            if page * 16 + col >= 5:
                for row in (11, 12):
                    ram[0x500 + page * 208 + row * 16 + col] = 0
    st = S.describe(ram)
    assert st["if_jump_right"] == {"lands_on": "pit"}
    assert st["if_hop_right"] == {"lands_on": "pit"}


def test_full_jump_clears_a_three_wide_pit_but_a_hop_from_far_back_does_not():
    ram = _running(make_ram(mx=40))               # front edge at px 56
    for col in (6, 7, 8):                         # pit px 96..144, 2.5 tiles ahead
        put(ram, col, 11, 0), put(ram, col, 12, 0)
    st = S.describe(ram)
    assert st["if_jump_right"]["lands_on"] == "floor"
    assert st["if_jump_right"]["tiles_ahead"] * 16 + 40 >= 144


def test_airborne_pull_left_lands_shorter_than_keep_right():
    ram = _running(make_ram(mx=40, my=GROUND_Y - 60, on_ground=False))
    ram[0x70A] = 0x90
    st = S.describe(ram)
    assert st["if_keep_right"]["lands_on"] == st["if_pull_left"]["lands_on"] == "floor"
    assert st["if_pull_left"]["tiles_ahead"] < st["if_keep_right"]["tiles_ahead"]


def test_jump_toward_a_tall_wide_platform_lands_on_top_of_it():
    ram = _running(make_ram(mx=40))
    for col in range(6, 16):
        for row in (8, 9, 10):
            put(ram, col, row)                    # 3-high platform starting 2.5 tiles ahead
    full = S.simulate(ram, a_frames=30, steer=1)
    assert full["on"] == "floor" and 96 - 12 <= full["x"] < 256
    hop = S.simulate(ram, a_frames=8, steer=1)     # a hop is too low: he hits the side and drops back
    assert hop["on"] == "floor" and hop["x"] < 96


def test_landing_prediction_flags_an_enemy_walking_into_the_landing_spot():
    ram = _running(make_ram(mx=40))
    full = S.simulate(ram, a_frames=30, steer=1)
    assert full["on"] == "floor" and full["frames"] > 20
    # a goomba walking left that will be at the landing x when Mario comes down
    add_enemy(ram, 0, ex=full["x"] + int(8 / 16 * full["frames"]), xspd=-8)
    st = S.describe(ram)
    assert st["if_jump_right"].get("hostile_near_landing") is True
    assert "hostile_near_landing" not in st["if_hop_right"]          # the hop comes down well short of it


def test_landing_prediction_ignores_powerups_and_far_enemies():
    ram = _running(make_ram(mx=40))
    full = S.simulate(ram, a_frames=30, steer=1)
    add_enemy(ram, 0, ex=full["x"], etype=S.POWERUP, xspd=0)          # friendly
    add_enemy(ram, 1, ex=full["x"] + 96, xspd=0)                      # hostile but 6 tiles past the landing
    assert "hostile_near_landing" not in S.describe(ram)["if_jump_right"]
