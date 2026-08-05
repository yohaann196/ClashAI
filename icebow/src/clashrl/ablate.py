"""REWARD ABLATION HARNESS -- which shaping terms actually earn their place?

`rewards:` has grown one term at a time (87db21c alone added four tornado terms) and no term has ever
been shown to help. Each one is a hyperparameter, an interaction surface with every other term, and a
farming opportunity for the policy. This measures them.

METHOD. For each shaping term (and for a BASELINE with nothing removed), train a fresh policy from
scratch with that term ZEROED, then score it on the FROZEN NON-ADAPTIVE eval pool. Repeat over seeds.
The reported number per term is

    delta = mean_winrate(term removed) - mean_winrate(baseline)

so a term that EARNS ITS PLACE has a NEGATIVE delta -- removing it made the policy worse. A term with a
POSITIVE delta is actively harmful. A term whose delta is inside the seed noise is doing nothing
measurable at this training budget, which for a reward term is itself a verdict: it is complexity with
no evidence behind it.

FOUR THINGS THIS GETS RIGHT, because each is easy to get wrong and would silently invalidate the table:

1. The FINAL checkpoint is scored, never `_best.pt`. `_best.pt` is selected on this very benchmark, so
   scoring it would mean reporting a max-statistic -- biased upward by an amount that depends on how
   many evals each run happened to get. That is selection bias, not a measurement.
2. Every source of randomness is seeded per run (`random`, `numpy`, `torch`), because train_sim's
   exploration draws from the GLOBAL `random` module. Without this, two runs at the "same seed" differ
   and the variance column measures nothing in particular.
3. The eval pool is frozen and non-adaptive (`opponent_provider = None`, `rng.seed(777 + j)`, canonical
   render), exactly as train_sim's own benchmark -- the comparison across terms is against one fixed
   opponent distribution.
4. Deltas are reported with a Welch confidence interval and an explicit verdict. With a handful of
   seeds the intervals are WIDE, and the honest reading of most rows will be "inconclusive". A table
   that reports a ranking without its noise band invites exactly the over-reading this exists to stop.

COST. Runs = (terms + 1) x seeds, each a full from-scratch train-sim. This is the dominant cost of the
whole exercise -- use `--dry-run` to see the plan and the projected wall time before committing, and
`--matches` to pick a budget. RESULTS ARE ONLY VALID AT THE BUDGET THEY WERE MEASURED AT: a term that
matters mainly for late-training refinement can look dead at 2k matches. Treat a short sweep as a
SCREEN for terms to delete, not as proof that a term is useless.

Results append to a JSON file and the sweep is RESUMABLE (`--resume`), because a full sweep runs for
hours and losing it to one crash is unacceptable. `--terms` shards the work across machines.

Usage, from icebow/:
    python run.py ablate-rewards --dry-run --matches 4000 --seeds 3
    python run.py ablate-rewards --matches 4000 --seeds 3 --envs 16 --resume
    python run.py ablate-rewards --terms nado_group,spell_waste --matches 4000 --seeds 3
"""
from __future__ import annotations

import contextlib
import copy
import io
import json
import math
import random
import time
from pathlib import Path
from functools import partial
from typing import Dict, List, Optional

import numpy as np

# A sweep runs for hours behind a pipe/tee, where stdout is block-buffered by default -- progress
# that only appears at the end is useless for a job you need to babysit.
_p = partial(print, flush=True)

# --- what gets ablated -------------------------------------------------------------------------
# name -> {config key: value to use when this ablation is active}. Zeroing is the usual ablation, but
# not always the RIGHT one -- see rocket_combo_mult.
ABLATIONS: Dict[str, Dict[str, float]] = {
    # (1) threat response
    "threat_response": {"threat_response": 0.0},
    "threat_miss": {"threat_miss": 0.0},
    "threat_group": {"threat_response": 0.0, "threat_miss": 0.0},
    # (2) elixir trade
    "elixir_trade": {"elixir_trade": 0.0},
    # (3) win-condition execution
    "wincon_exec": {"wincon_exec": 0.0},
    "wincon_misplace": {"wincon_misplace": 0.0},
    "wincon_group": {"wincon_exec": 0.0, "wincon_misplace": 0.0},
    # A MULTIPLIER, not an additive term: wincon_exec * rocket_combo_mult. Zeroing it would make a
    # rocket 2-for-1 score 0 -- LESS than the 0.6x an ordinary defensive rocket chip earns -- i.e. it
    # would punish the combo rather than remove its bonus. 1.0 is the real ablation: combo == base.
    "rocket_combo_mult": {"rocket_combo_mult": 1.0},
    # Soft partial credit for a too-deep defensive X-Bow. 0.0 removes the partial credit (the misplace
    # penalty then applies), which is exactly the thing being tested.
    "xbow_deep_frac": {"xbow_deep_frac": 0.0},
    # (4) cycle planning
    "cycle_plan": {"cycle_plan": 0.0},
    "cycle_waste": {"cycle_waste": 0.0},
    "cycle_group": {"cycle_plan": 0.0, "cycle_waste": 0.0},
    # (5) tempo / economy
    "leak_penalty": {"leak_penalty": 0.0},
    "spell_waste": {"spell_waste": 0.0},
    # tornado execution credit (87db21c). Each is small and fires rarely, so the INDIVIDUAL deltas are
    # likely to sit inside the noise -- nado_group is the test that actually has a chance of resolving.
    "nado_clump": {"nado_clump": 0.0},
    "nado_combo": {"nado_combo": 0.0},
    "nado_king_activate": {"nado_king_activate": 0.0},
    "nado_retarget": {"nado_retarget": 0.0},
    "nado_group": {"nado_clump": 0.0, "nado_combo": 0.0,
                   "nado_king_activate": 0.0, "nado_retarget": 0.0},
    # dense tower-chip proxy (the crown jump in take/lose_own_tower is NOT ablated -- it is part of the
    # sparse outcome signal this whole exercise is trying to fall back to)
    "tower_chip_scale": {"tower_chip_scale": 0.0},
    # ALL shaping off: the sparse-reward control. If this is not much worse than baseline, the entire
    # shaping apparatus is not paying for itself and the table below is rearranging deck chairs.
    "all_shaping": {"threat_response": 0.0, "threat_miss": 0.0, "elixir_trade": 0.0,
                    "wincon_exec": 0.0, "wincon_misplace": 0.0, "cycle_plan": 0.0,
                    "cycle_waste": 0.0, "leak_penalty": 0.0, "spell_waste": 0.0,
                    "nado_clump": 0.0, "nado_combo": 0.0, "nado_king_activate": 0.0,
                    "nado_retarget": 0.0, "tower_chip_scale": 0.0},
}

# Two-sided t critical values at 95%, by degrees of freedom. Small-sample sweeps land at df 2-8, where
# the normal approximation (1.96) is badly optimistic -- df=2 needs 4.30.
_T95 = {1: 12.71, 2: 4.30, 3: 3.18, 4: 2.78, 5: 2.57, 6: 2.45, 7: 2.36, 8: 2.31,
        9: 2.26, 10: 2.23, 12: 2.18, 15: 2.13, 20: 2.09, 30: 2.04}


def _t_crit(df: float) -> float:
    if df <= 0:
        return float("inf")
    keys = sorted(_T95)
    for k in keys:
        if df <= k:
            return _T95[k]
    return 1.96


def _seed_everything(seed: int) -> None:
    """train_sim's exploration draws from the GLOBAL `random` module, so a `seed=` argument alone does
    NOT make a run reproducible. Without this the variance column is measuring uncontrolled noise."""
    import torch
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    torch.manual_seed(seed)


def benchmark(cfg, ckpt_path: Path, matches: int = 48, envs: int = 4) -> Optional[float]:
    """Win-rate (%) of a checkpoint against the FROZEN NON-ADAPTIVE eval pool.

    Deliberately a standalone re-implementation of train_sim's `evaluate(fair=False)` protocol rather
    than a hook into its training-time eval history: the sweep needs every run scored on the same fixed
    opponent distribution with the same sample size, decoupled from how many evals a run happened to
    trigger. `opponent_provider = None` keeps the bots NON-ADAPTIVE (87db21c's adaptive behaviours are
    training-only), `rng.seed(777 + j)` fixes the deck sequence, and the render is canonical.
    """
    import torch

    from .sim.env import SimMatchEnv
    from .train_rl import _build_net, _pick_device

    if not ckpt_path.exists():
        return None
    device = _pick_device(cfg)
    pool = [SimMatchEnv(cfg, seed=100000 + i) for i in range(max(1, envs))]
    for e in pool:
        e.domain_rand.enabled = False        # canonical render: comparable + noise-free
        e.domain_rand.resample()
        e.opponent_provider = None           # frozen scripted meta bots, NON-adaptive
    e0 = pool[0]
    net = _build_net(cfg, device, e0.n_cards, e0.n_cells, e0.threat_dim)
    ck = torch.load(ckpt_path, map_location="cpu")
    net.policy.load_state_dict(ck["model"])
    if "gate" in ck:
        net.gate.load_state_dict(ck["gate"])
    net.eval()

    anywhere = set(e0.anywhere_ids)
    yourhalf = torch.tensor(e0.actions.deployable_mask(False), dtype=torch.bool, device=device)
    allcells = torch.ones(e0.n_cells, dtype=torch.bool, device=device)
    costs = torch.tensor([float(s.elixir) for s in e0.specs], dtype=torch.float32, device=device)

    def to_t(o):
        return torch.from_numpy(o).float().permute(2, 0, 1).to(device) / 255.0

    def vec(v):
        return torch.from_numpy(np.asarray(v, np.float32)).to(device)

    for j, e in enumerate(pool):
        e.rng.seed(777 + j)                  # SAME benchmark decks + engine rolls every run
    obs = [e.reset() for e in pool]
    wins = played = 0
    while played < matches:
        with torch.no_grad():
            cq, ceq, gq = net(torch.stack([to_t(o) for o in obs]),
                              torch.stack([vec(e.hand_vec) for e in pool]),
                              torch.stack([vec(e.next_vec) for e in pool]),
                              torch.stack([vec(e.elixir_vec) for e in pool]),
                              torch.stack([vec(e.threat_vec) for e in pool]))
        cq = cq.masked_fill(torch.stack([vec(e.hand_vec) for e in pool]) < 0.5, float("-inf"))
        for i, e in enumerate(pool):
            cq_i = cq[i].masked_fill(costs > e.elixir + 1e-6, float("-inf"))
            if not torch.isfinite(cq_i).any():
                act = (0, 0, 0)
            else:
                ci = int(cq_i.argmax())
                ceq_i = ceq[i].masked_fill(~(allcells if ci in anywhere else yourhalf), float("-inf"))
                act = ((0, 0, 0) if gq[i, 0] >= gq[i, 1] + cq_i.max() + ceq_i.max()
                       else (1, ci, int(ceq_i.argmax())))
            nobs, _r, done, info = e.step(act)
            if done:
                wins += info.get("outcome") == "win"
                played += 1
                obs[i] = e.reset()
            else:
                obs[i] = nobs
    return 100.0 * wins / max(1, played)


def _run_one(cfg, name: str, overrides: Dict[str, float], seed: int, matches: int, envs: int,
             eval_matches: int, workdir: Path, quiet: bool = True) -> float:
    """Train one policy from scratch with `overrides` applied to rewards, then benchmark it."""
    from .train_sim import train_sim

    run_cfg = copy.deepcopy(cfg)
    run_cfg.data.setdefault("rewards", {}).update(overrides)
    ckpt = workdir / f"{name}_s{seed}.pt"
    run_cfg.data.setdefault("train", {})["sim_checkpoint"] = str(ckpt)
    # No mid-training benchmark: it costs wall time and, more importantly, is what writes `_best.pt`.
    # The sweep scores the FINAL checkpoint, so the keep-best machinery is pure overhead here.
    run_cfg.data.setdefault("sim", {})["eval_every_matches"] = 0

    _seed_everything(seed)
    buf = io.StringIO()
    ctx = contextlib.redirect_stdout(buf) if quiet else contextlib.nullcontext()
    with ctx:
        train_sim(run_cfg, matches=matches, resume=False, seed=seed, envs=envs)
    _seed_everything(seed + 99991)           # decouple eval draws from the training stream
    wr = benchmark(run_cfg, ckpt, matches=eval_matches, envs=min(envs, 4))
    try:                                     # a full sweep would otherwise leave GBs of checkpoints
        ckpt.unlink(missing_ok=True)
        ckpt.with_name(ckpt.stem + "_best" + ckpt.suffix).unlink(missing_ok=True)
    except OSError:
        pass
    try:                                     # 60+ sequential runs otherwise creep on GPU memory
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001
        pass
    return wr


def _stats(xs: List[float]) -> Dict[str, float]:
    n = len(xs)
    if n == 0:
        return {"n": 0, "mean": float("nan"), "sd": float("nan")}
    m = sum(xs) / n
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1)) if n > 1 else 0.0
    return {"n": n, "mean": m, "sd": sd}


def _welch(a: List[float], b: List[float]):
    """(delta, half-width of the 95% CI, df) for mean(a) - mean(b). Returns inf width when either arm
    has <2 samples -- with one seed there is no variance estimate and no honest interval."""
    sa, sb = _stats(a), _stats(b)
    delta = sa["mean"] - sb["mean"]
    if sa["n"] < 2 or sb["n"] < 2:
        return delta, float("inf"), 0.0
    va, vb = sa["sd"] ** 2 / sa["n"], sb["sd"] ** 2 / sb["n"]
    se = math.sqrt(va + vb)
    if se == 0.0:
        return delta, 0.0, float(sa["n"] + sb["n"] - 2)
    df = (va + vb) ** 2 / ((va ** 2 / (sa["n"] - 1)) + (vb ** 2 / (sb["n"] - 1)))
    return delta, _t_crit(df) * se, df


def ablate_rewards(cfg, matches: int = 4000, seeds: int = 3, envs: int = 8, terms: Optional[str] = None,
                   eval_matches: int = 48, out: Optional[str] = None, resume: bool = False,
                   dry_run: bool = False, verbose: bool = False) -> None:
    try:
        import torch
        _ = torch.__version__          # availability probe: fail early, not 40 minutes in
    except ImportError as exc:  # noqa: BLE001
        _p(f"[ablate] PyTorch required ({exc}). Install the CUDA build (see README).")
        return

    selected = list(ABLATIONS) if not terms else [t.strip() for t in terms.split(",") if t.strip()]
    unknown = [t for t in selected if t not in ABLATIONS]
    if unknown:
        _p(f"[ablate] unknown term(s): {', '.join(unknown)}")
        _p(f"[ablate] available: {', '.join(sorted(ABLATIONS))}")
        return
    arms = ["baseline"] + selected
    seed_list = list(range(seeds))
    total = len(arms) * len(seed_list)

    out_path = Path(cfg.path(out or "runs/ablate/rewards.json"))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results: Dict[str, Dict[str, float]] = {}
    if resume and out_path.exists():
        try:
            results = json.loads(out_path.read_text()).get("runs", {})
            _p(f"[ablate] resuming: {len(results)} run(s) already recorded in {out_path}")
        except (OSError, ValueError):
            _p(f"[ablate] could not read {out_path}; starting fresh")

    _p(f"[ablate] {len(arms)} arm(s) x {len(seed_list)} seed(s) = {total} from-scratch train-sim runs")
    _p(f"[ablate] {matches} matches/run, {envs} env(s), scored on {eval_matches} frozen eval matches")
    _p(f"[ablate] arms: {', '.join(arms)}")
    if dry_run:
        _p("[ablate] --dry-run: nothing executed. Time one run first to project the sweep:")
        _p(f"[ablate]   python run.py ablate-rewards --terms spell_waste --seeds 1 --matches {matches}")
        return

    workdir = Path(cfg.path("runs/ablate/ckpt"))
    workdir.mkdir(parents=True, exist_ok=True)
    done = 0
    t_start = time.time()
    for arm in arms:
        overrides = {} if arm == "baseline" else ABLATIONS[arm]
        for seed in seed_list:
            key = f"{arm}|{seed}"
            done += 1
            if key in results:
                _p(f"[ablate] ({done}/{total}) {key} -- already done, skipping")
                continue
            t0 = time.time()
            try:
                wr = _run_one(cfg, arm, overrides, seed, matches, envs, eval_matches, workdir,
                              quiet=not verbose)
            except KeyboardInterrupt:
                _p("\n[ablate] interrupted -- partial results kept; re-run with --resume")
                _report(results, arms, matches, eval_matches, out_path)
                return
            dt = time.time() - t0
            results[key] = {"arm": arm, "seed": seed, "winrate": wr, "seconds": dt,
                            "matches": matches, "overrides": overrides}
            eta = (time.time() - t_start) / done * (total - done)
            _p(f"[ablate] ({done}/{total}) {arm:<20} seed {seed}  winrate {wr:5.1f}%  "
                  f"[{dt / 60:.1f} min, ETA {eta / 60:.0f} min]", flush=True)
            out_path.write_text(json.dumps({"config": {"matches": matches, "seeds": seeds,
                                                       "envs": envs, "eval_matches": eval_matches},
                                            "runs": results}, indent=2))
    _report(results, arms, matches, eval_matches, out_path)


def _report(results, arms, matches, eval_matches, out_path) -> None:
    by_arm: Dict[str, List[float]] = {}
    for r in results.values():
        if r.get("winrate") is not None:
            by_arm.setdefault(r["arm"], []).append(float(r["winrate"]))
    base = by_arm.get("baseline", [])
    if not base:
        _p("[ablate] no baseline runs recorded -- cannot report deltas")
        return

    rows = []
    for arm in arms:
        if arm == "baseline" or arm not in by_arm:
            continue
        delta, ci, df = _welch(by_arm[arm], base)
        s = _stats(by_arm[arm])
        if not math.isfinite(ci):
            verdict = "NO CI (need >=2 seeds)"
        elif delta + ci < 0:
            verdict = "TERM HELPS"          # removing it significantly HURT -> the term earns its place
        elif delta - ci > 0:
            verdict = "TERM HURTS"          # removing it significantly HELPED -> the term is harmful
        else:
            verdict = "inconclusive"
        rows.append((delta, ci, s, arm, verdict))
    rows.sort(key=lambda r: r[0])            # most-negative delta first = most valuable term first

    bs = _stats(base)
    _p()
    _p("=" * 100)
    _p(f"REWARD ABLATION -- {matches} matches/run, {bs['n']} seed(s), "
          f"{eval_matches} frozen non-adaptive eval matches per run")
    _p(f"BASELINE (nothing removed): {bs['mean']:.1f}% +- {bs['sd']:.1f} (sd over {bs['n']} seed(s))")
    _p("=" * 100)
    _p(f"{'term removed':<22}{'winrate':>10}{'sd':>7}{'delta':>9}{'95% CI':>18}   verdict")
    _p("-" * 100)
    for delta, ci, s, arm, verdict in rows:
        ci_s = "(need >=2 seeds)" if not math.isfinite(ci) else f"[{delta - ci:+6.1f},{delta + ci:+6.1f}]"
        _p(f"{arm:<22}{s['mean']:>9.1f}%{s['sd']:>7.1f}{delta:>+9.1f}{ci_s:>18}   {verdict}")
    _p("-" * 100)
    _p("delta = winrate(term removed) - winrate(baseline).")
    _p("  NEGATIVE delta -> removing the term HURT  -> the term earns its place.")
    _p("  POSITIVE delta -> removing the term HELPED -> the term is actively harmful.")
    _p("  CI crossing zero -> no measurable effect AT THIS BUDGET. For a reward term that is a")
    _p("  verdict in itself: it is complexity with no evidence behind it. It is NOT proof of")
    _p("  uselessness -- a term that matters late in training can look dead in a short sweep.")
    _p(f"raw runs -> {out_path}")
