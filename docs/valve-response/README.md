# Experimental MPC V2 valve-response learning

This opt-in extension learns **delivered room heating versus valve opening**, in K/min. It does not measure hydraulic flow or promise to identify a unique physical valve characteristic from room temperature alone.

## Enable

Use MPC V2 and direct valve control with **one radiator per Better Thermostat**. In advanced options enable **MPC V2: learn valve heating response (experimental)**. Multiple-radiator groups and target-temperature calibration keep their existing controller behavior.

The configured maximum opening remains an upper bound. You can set it to 100% to allow observations over the full range. Learning does not force heating experiments: a room that never needs high openings cannot reveal that part of the curve. With mild weather or short cycles, learning may remain inconclusive for a long time. A higher configured cap can deliver more heat before learning has enough evidence, so it is not equivalent to a pre-calibrated valve.

This fork declares `daqp==0.9.1` so the optimizer is installed with the integration. Disabling response learning returns to the original MPC model; retained observations are not erased.

## What changes in control

Initially the existing MPC remains in use. After three consistent, independent settled heating holds at an opening, the response model can influence control. It models room temperature and retained radiator heating rate, plans in effective heat, and maps the chosen heat back to the smallest valve opening that delivers it. The configured cap is included in the optimizer's constraints, not just clipped afterward. The Sonoff closing workaround now also respects that cap.

The curve is monotone and uses a conservative upper envelope when observations disagree. Unknown portions retain assumptions for control; diagnostics never display those assumptions as measured points. Beyond the highest supported opening, the control curve continues increasing until real observations show saturation. The radiator lag remains a 15-minute prior; it is not identified independently from sensor lag in this version.

For an active response model, `mpc_v2_heat_rate_hat_K_min` replaces the fictitious radiator-temperature estimate. `mpc_v2_T_rad_hat` is omitted, because the second state is now a heating rate, not a temperature. Existing temperature and valve diagnostics remain available.

## Data acceptance

- Every accepted point requires a fresh timestamped room report. Repeated control calls with the same report add no evidence.
- Commands must stay within one percentage point for at least 30 minutes before a heating measurement window starts. Off baselines require 45 minutes for retained radiator heat to decay.
- A slope window spans at least 15 minutes, contains at least four reports and uses a robust median of pairwise slopes.
- Gaps or report ages over ten minutes break the interval. Missing, non-finite, out-of-order, unavailable and degraded inputs cannot train it.
- Window/heating interruptions discard unfinished evidence. Live learning waits at least 45 minutes after a contact closes. Large temperature jumps, inconsistent slopes and changing weather are rejected.
- A heating hold needs a recent off baseline under similar outdoor conditions. This subtracts local room cooling rather than attributing the full observed temperature rise to the radiator.
- Each independent valve hold contributes **one** observation. Longer holds improve that observation, not its confidence count.
- Evidence is bounded to 32 holds per 5-percentage-point bin and expires after 14 days. Conflicting heat observations stop that bin influencing control.

These gates intentionally favor incomplete evidence over a confident wrong model. Sunlight, occupants, changing boiler water temperatures and unmeasured heat sources can still confound observations. The model estimates the room/radiator system under recently observed operating conditions, not a permanent factory calibration. Quiet Aqara sensors can prevent training even while ordinary heating remains functional.

## Uncertainty and estimated saturation

After six independent holds in a bin, the diagnostic includes a distribution-free, pointwise confidence interval for the median heating response. The order-statistic interval has at least 95% nominal coverage for independent, identically distributed holds, with an additional 0.003 K/min noise floor on either side. Before that, confidence bounds are null. These intervals do not cover systematic sensor bias or unobserved boiler/solar changes and are not simultaneous confidence bands over the entire curve.

An effective saturation estimate requires supported observations reaching **100%**, at least three supported points in the candidate tail and no gap wider than 20 percentage points. The upper bound on additional heating must be within 15% of the estimated 100%-opening heat. The resulting percentage is an approximate useful-opening knee, not the exact physical point of full flow. Wide overlapping intervals do not count as proof of saturation. The estimate is advisory and never silently lowers the configured cap.

## Diagnostic plots

1. Copy `www/mpc-response-card.js` to `/config/www/mpc-response-card.js`.
2. Add `/local/mpc-response-card.js` as a **JavaScript module** in dashboard resources.
3. Add the card below, replacing the example entity with your Better Thermostat entity:

```yaml
type: custom:mpc-response-card
entity: climate.my_room
title: Room · learned heating response
```

The card plots the estimated response with pointwise confidence intervals, the actual control curve, independent-hold coverage, learning status and estimated saturation. Unobserved points stay empty. The dashed control curve visibly distinguishes assumed/interpolated response from observations. A second Plotly example in `history-card.yaml` charts measured/target temperature, estimated retained heat and valve commands over time.

## Historical replay and seeding

`scripts/replay_valve_response.py` reads an exported Home Assistant history response. Export climate attributes, raw room temperature, valve opening number, contact and weather with `significant_changes_only=0`. Use only periods with known MPC V2 valve commands and unchanged room/sensor/valve wiring.

```sh
.venv/bin/python scripts/replay_valve_response.py history.json replay.json \
  --climate climate.my_room --room sensor.room_temperature \
  --valve number.valve_opening --window binary_sensor.room_window \
  --weather weather.home
```

The result includes a source hash, accepted evidence, diagnostics and rejection counts. It does not write to Home Assistant or bypass acceptance gates. Evidence can be restored by `ValveResponseLearner.restore()` for offline simulations. There is deliberately no unvalidated live state-file import. Keep private exports outside the public repository.

A replay may legitimately accept zero independent holds. Do not seed a curve from rejected intervals or treat the existing aggregate heating-rate estimate as a measured valve curve. Existing Better Thermostat learning is preserved.

## Validation and rollout

Tests use an independent simulated room with a nonlinear valve reaching full response at 35%, radiator inertia, sensor lag and noise. They cover estimation, uncertainty, missing coverage, stale data, interruptions, outliers, bounded persistence, optimizer cap behavior and closed-loop overshoot. In the reference closed-loop test with a 100% configured cap, peak overshoot falls from 0.291°C to 0.118°C; the final temperature is 21.548°C for a 21.5°C target. Simulated performance is not a guarantee for a real room.

Start with observation and inspect accepted-hold coverage before expecting a trustworthy curve. Use the existing cap for initial deployment; increasing it to 100% is a separate operator choice. No running Home Assistant installation or dashboard was changed while developing this branch.
