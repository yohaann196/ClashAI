"""Verify NUMERICALLY that clashrl/shaping.py is policy-invariant in the Ng et al. sense.

"Potential-based" is a mathematical claim, not a naming convention -- the codebase already called two
terms potential-based while omitting the gamma factor and clipping one of them, which silently breaks
the property. This script tests the property itself rather than trusting the label.

THE PROPERTY. For F(s,a,s') = gamma*Phi(s') - Phi(s) with Phi(absorbing) = 0, the DISCOUNTED SUM of F
over an episode telescopes:

    sum_t gamma^t * F_t  =  sum_t [gamma^(t+1) Phi(s_t+1) - gamma^t Phi(s_t)]  =  -Phi(s_0)

Every intermediate term cancels. So from a FIXED start state the discounted shaping return is the SAME
CONSTANT no matter what actions are taken -- which is exactly why shaping of this form cannot change
which policy is optimal. That is the testable statement, and it is what this checks: run wildly
different action sequences from one identical start state and confirm they all return -Phi(s_0).

Test 2 is the control: the CLASSIC (non-potential) reward is run the same way and must FAIL the same
check. If both passed, the test would be measuring nothing.

Run from icebow/:
    .venv/Scripts/python.exe tools/potential_invariance.py
"""
from __future__ import annotations

import pathlib
import random
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

from clashrl.config import Config  # noqa: E402
from clashrl.sim.env import SimMatchEnv  # noqa: E402

CFG = str(pathlib.Path(__file__).resolve().parent.parent / "config" / "config.yaml")


def _cfg(potential: bool) -> Config:
    cfg = Config.load(CFG)
    cfg.data.setdefault("rewards", {}).setdefault("potential_based", {})["enabled"] = potential
    cfg.data["observation"]["obs_mode"] = "rgb"     # the raster is irrelevant here; rgb is cheapest
    return cfg


def _rollout(cfg, policy_seed: int, max_steps: int = 400):
    """One episode from a FIXED start state (env seed pinned), acting per `policy_seed`.

    Returns (discounted shaping return, discounted total return, phi_0). Only the SHAPING part is
    subject to the invariance claim -- the sparse outcome terms legitimately differ between policies,
    which is the entire point of keeping them.
    """
    env = SimMatchEnv(cfg, seed=4242)               # same env seed => same s_0 for every policy
    env.reset()
    gamma = float(cfg.get("train", "gamma", default=0.99))
    phi0 = env.shaper.phi() if env.shaper is not None else 0.0
    rng = random.Random(policy_seed)
    disc_shape = disc_total = 0.0
    g = 1.0
    for _ in range(max_steps):
        hand = [c for c in env._hand_ids() if env.specs[c].elixir <= env.eng.elixir[0]]
        # deliberately different behaviour per policy_seed: play rate and placement both vary
        if hand and rng.random() < (0.15 + 0.7 * ((policy_seed % 5) / 4.0)):
            act = (1, rng.choice(hand), rng.randrange(env.n_cells))
        else:
            act = (0, 0, 0)
        _o, r, done, _i = env.step(act)
        disc_total += g * r
        if env.shaper is not None:
            disc_shape += g * env._shaping_last
        g *= gamma
        if done:
            break
    return disc_shape, disc_total, phi0


def main() -> int:
    ok = True

    print("TEST 1 -- POTENTIAL shaping: discounted shaping return must equal -Phi(s_0) for EVERY policy")
    cfg = _cfg(True)
    vals = []
    for ps in range(6):
        disc_shape, disc_total, phi0 = _rollout(cfg, ps)
        err = abs(disc_shape - (-phi0))
        vals.append(disc_shape)
        flag = "ok  " if err < 1e-6 else "FAIL"
        if err >= 1e-6:
            ok = False
        print(f"  {flag} policy {ps}: shaping return {disc_shape:+.9f}  -Phi(s_0) {-phi0:+.9f}  "
              f"|err| {err:.2e}   (total return {disc_total:+8.3f})")
    spread = max(vals) - min(vals)
    print(f"  spread across policies: {spread:.2e}  (must be ~0: the shaping return is policy-INVARIANT)")
    if spread >= 1e-6:
        ok = False

    print()
    print("TEST 2 -- CONTROL: the CLASSIC reward must NOT have this property (else test 1 proves nothing)")
    cfg_c = _cfg(False)
    totals = [_rollout(cfg_c, ps)[1] for ps in range(6)]
    c_spread = max(totals) - min(totals)
    print(f"  classic total return across the same 6 policies: "
          f"{', '.join(f'{t:+.2f}' for t in totals)}")
    print(f"  spread: {c_spread:.3f}  (must be LARGE -- classic shaping is policy-dependent/farmable)")
    if c_spread < 1.0:
        print("  FAIL: the control did not vary; the test is not discriminating")
        ok = False

    print()
    print("TEST 3 -- Phi components are finite and state-only")
    env = SimMatchEnv(_cfg(True), seed=7)
    env.reset()
    for _ in range(40):
        hand = [c for c in env._hand_ids() if env.specs[c].elixir <= env.eng.elixir[0]]
        env.step((1, hand[0], 200) if hand else (0, 0, 0))
    bd = env.shaper.breakdown()
    for k, v in bd.items():
        finite = abs(v) < 1e6 and v == v
        ok &= finite
        print(f"  {'ok  ' if finite else 'FAIL'} Phi[{k:<7}] = {v:+.4f}")
    # calling phi() twice with no state change must give the same value (no hidden accumulation)
    a, b = env.shaper.phi(), env.shaper.phi()
    same = abs(a - b) < 1e-12
    ok &= same
    print(f"  {'ok  ' if same else 'FAIL'} Phi is a pure function of state (repeat call: {a:.9f} vs {b:.9f})")

    print()
    print("INVARIANCE VERIFIED" if ok else "INVARIANCE CHECK FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
