# Review / fixes

Verified against scipy `expm`, finite differences and Pinocchio. Run `python tests.py`.

## Real bugs
1. **SE2 & SE3 `jacobian_minus_right_1/_2` were swapped** (d/dM1 must be `Jr^-1`, d/dM2 must be `-Jl^-1`; SO3/S3/S1/SO2 were already right).
2. **T `minus_right/minus_left` had the wrong sign** (`t2 - t1` instead of `t1 - t2`) and returned an (n,1) column; they also crashed for some inputs.
3. **T `plus_right/plus_left` returned a bare translation column, not a group element** (4x4 matrix). The old test `right_plus == left_plus` passed trivially.
4. **SO3 `Log/log` had no `arccos` clip -> NaN for `R == I` (+rounding)**; precision loss (~1e-8) for small angles; garbage at theta near pi (division by sin(theta) ~ 0). SE3 `Log` inherited all of this. Now: atan2 + symmetric-part axis extraction near pi.
5. **SO3 `Log` small-angle branch returned `theta * [0,0,1]`** (axis forced to z) instead of the real rotation vector.
6. **S3 `Log/log` returned theta > pi for `w < 0`** (q and -q are the same rotation). Now flips to the shortest rotation.
7. **`SO3.qR_quat` was wrong at theta = pi** for axes with mixed signs (e.g. (1,-1,0)); replaced by Shepperd's method.
8. **`hat`, `vee`, `algebra_element`, `exp`, `Exp`, `group_element` silently dropped small inputs** (`|theta| < 1e-7 -> 0`). `SO3.vee` also returned shape (3,1) there; SE3 `vee` returned `theta*[0,0,1]`. They now keep the first-order term.
9. **Catastrophic cancellation for small angles** in `(1-cos t)/t`, `(t-sin t)/t^3`, SE2 `V/Vinv` (error up to 1e-3 at theta = 2e-7), and especially the SE3 `Q` matrix (error up to ~1e-2 for t in [1e-7, 1e-3]). Replaced by Taylor series below 0.5 rad (see `lie_utils.py`); also made `Jl_inv` stable at theta = pi (cot(t/2) form).

## Smaller issues
- Docstrings with `\in` -> `SyntaxWarning` (future `SyntaxError`): made them raw strings.
- `S1.group_identity()` was real `[1.]` (now `1+0j`); `S1.exp` used a needless complex-theta round trip.
- Dead commented-out code removed; S3 docstring said `tau_hat = theta*u_hat` (it is `tau/2`).
- `tests.py`: only self-consistency checks, Jacobians never verified, failures swallowed by `except AssertionError: pass`, `small_case_tol` defined only under `__main__`, random `tau` with norm > pi (can't round-trip through `Log`) -> the "94.3 %" was noise. Replaced with a strict suite.
- `pinocchio_testing.py`: unused/uninstalled `manifpy` import crashed the script; compared numpy arrays with Pinocchio objects; the speed message was inverted.

## Notes (not changed)
- `group_action(M, x)` for T/SE2/SE3 expects homogeneous `x`, while `jacobian_*_action_*` take a plain point `p`.
- `tau` with `|theta| > pi` is not preserved by `Log(Exp(tau))` (inherent to the representation).
