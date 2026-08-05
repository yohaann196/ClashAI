# Sim-engine test suite

Run from `icebow/`:

```powershell
.\.venv\Scripts\python.exe -m pytest
```

Needs only `pytest`, `PyYAML` and `numpy` — **no torch, no OpenCV**, so it runs anywhere and in CI
without the heavy training dependencies.

## What this covers

`src/clashrl/sim/engine.py` — the headless match engine every sim-trained policy learns inside. If the
engine's model of Clash Royale is wrong, everything downstream optimises the wrong game, silently.

| file | mechanics pinned |
|---|---|
| `test_elixir.py` | regeneration, the 1x/2x/3x tiers and their exact boundaries, the 10-elixir cap, card costs, match end |
| `test_combat.py` | discrete hits, `hit_dmg == dps * hit_speed`, slow's effect on cadence, crown-tower damage reduction (`tower_hit_dmg` vs `hit_dmg`), tower firing |
| `test_towers.py` | destruction and overkill, crowns, king activation by damage and by tower loss, the overtime tiebreak, tower level scaling |
| `test_deploy.py` | the ~1s deploy delay, deploy immunity, swarm placement, building lifetime, arena clamping |
| `test_spells.py` | Rocket's point blast vs The Log's rolling corridor vs Tornado's pulling vortex — radius, direction, knockback, drag rate, DoT totals |
| `test_stat_scaling_and_defenses.py` | `1.1^(level-11)` scaling, shield pools, Evo Knight damage reduction |

## Conventions

**Every test names the Clash Royale mechanic it pins**, in its docstring, starting `CR mechanic:`.
A test that cannot name its mechanic is testing an implementation detail and should be reconsidered.
Two tests are honestly labelled `GOLDEN DATA (not a mechanic):` instead — they guard the stat *file*
(its reference level, and `damage` vs `dps * hit_speed` agreeing to rounding) rather than a game rule.

**Golden values come from `config/cards_stats.json`** (the level-11 Fandom stat dump) two ways:

- *recomputed* from the published fields — catches drift in `build_spec`
- *pinned as literals* (Knight 1766 HP / 202 damage / 1.2s, Rocket 1484 / 371) — catches drift in the
  stat file itself, which the recompute tests cannot see

**The engine is made deterministic.** `SimEngine.reset()` rolls the opponent's tower troop and level,
so the `cfg` fixture pins that to princess/L15 — identical to the player's side.

**Two immobilise helpers, and they are not interchangeable:**

- `freeze(unit)` — huge deploy delay: cannot act, ignored by collision, and **not targetable by
  towers** (deploy immunity). For spell and vortex geometry.
- `park(unit)` — huge stun: cannot act, but **still a legal tower target**. For anything where a tower
  must actually shoot.

## Regression guards

Three tests exist because the behaviour was wrong before and the fix must not be undone:

- `test_rocket_is_a_point_blast_not_a_rolling_spell` — Rocket carries a `knockback` flag, and
  classifying rolls on that alone gave it The Log's narrow forward corridor instead of its blast (87db21c)
- `test_evo_knight_has_damage_reduction_not_a_shield` — Evo Knight was first modelled as an HP shield
  pool, a different mechanic with different counterplay (5440e7b)
- `test_multi_unit_card_spawns_a_cluster_at_the_placement_point` — swarm spawn offsets re-included the
  placement coordinate, so every multi-unit card resolved to `2x + offset` and clamped into the arena
  corner

The suite is mutation-checked: deliberately breaking each covered mechanic in `engine.py` (elixir
thresholds, level-scaling exponent, shield overflow, king wake, tornado drag rate and pro-rating,
crown-tower reduction, deploy delay, roll classification, swarm offsets) makes it fail.
