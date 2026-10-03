# Experimental MPC V2 response learning

The response learner estimates delivered room heating versus valve command from
irregular temperature reports. It supports event-triggered battery sensors such
as Aqara: quiet periods do not create samples or reset learning after ten minutes.

**This version is observation-only. Candidate curves never affect heating.**
Existing MPC V2, configured valve limits, presets and schedules continue to control
the radiator. No automatic probes or cap increases are performed. The output is
room heating in K/min, not measured water flow.

## Enable and observe

Use MPC V2 with direct valve control and one physical radiator. Enable the
advanced option **MPC V2: learn valve heating response (experimental)**. Multiple
radiators and other calibration methods retain their existing behavior.

The option starts collecting reports and applied command history. It does not
activate a candidate curve. Disabling it stops collection after the normal options
reload. Accepted evidence is retained in the entry's unified state store.

Install `www/mpc-response-card.js` as a JavaScript module and configure:

```yaml
type: custom:mpc-response-card
entity: climate.my_room
```

The updated card shows candidate response, uncertainty ranges, episode coverage,
report age, heartbeat allowance and the current collection reason. Update the
resource URL's version parameter when replacing the file to avoid browser caching.
`history-card.yaml` remains compatible; the retained learned-heat trace stays empty
while observation-only operation is in use.

## What the estimator uses

- Raw room-temperature values paired with their actual report timestamps. BT's
  filtered/delayed control temperature is not paired with a newer raw timestamp.
- Both changed readings and unchanged-value heartbeats. Controller ticks never
  count as fresh measurements.
- Successful actuator commands at their write times, plus observed outdoor changes.
  These are commands, not independently measured physical valve positions.
- One shared physical-source learner across eco, comfort and manual targets.
  Target-specific MPC controller state remains separate.

Sonoff's closing workaround records its temporary opening bump and delayed final
command separately, after both opening/closing writes complete. A refused or
partially delivered write interrupts learning until a successful command restores
certainty. Delayed closes retry twice and stop when superseded or the thermostat
unloads. `valve_delivery` reports command certainty and runtime failure counts;
clear general controller errors do not establish reliable radio delivery.

Each episode spans at least six hours and twelve genuine reports. Between reports,
the model integrates every recorded valve command through radiator inertia and
room heat loss. Commands may vary throughout an episode. Episodes have disjoint
measurement intervals; adjacent episodes share only their boundary reading.

A small monotone curve uses at most three opening knots. The fit searches a bounded
room-loss coefficient, estimates initial retained radiator heat and a constant
background heat term, and assumes a 15-minute radiator lag. This is deliberately
less flexible than fitting a separate coefficient at every valve percentage.

The model requires cooling and heating exposure and a sufficiently independent
set of inputs. It fits the first three quarters of an episode's intervals and
checks predictions on the remainder against both error limits and a constant
last-temperature predictor. Fitting runs in Home Assistant's executor, outside the
main event loop.

## Silence, uncertainty and interruptions

A quiet sensor contributes no new measurement. The learner retains command history
and waits for a real report. The expected heartbeat allowance starts at 90 minutes
and can grow, from observed report gaps, to at most 150 minutes. This allowance is
for learning, not a change to Home Assistant's device-availability policy.

Unavailability, missing heartbeats, open/unknown contacts, maintenance, missing
outdoor data and large temperature jumps discard the unfinished episode. Window
recovery retains the existing 45-minute learning pause. Handling-like jumps also
start a 45-minute learning quarantine. There is no assumption that silence proves
the temperature stayed within a specific threshold: missed radio packets and
firmware-specific reporting rules can invalidate that inference.

Blue bands show sensitivity to plausible loss coefficients, finite measurement
information and differences between accepted episodes, with a small noise floor.
Points with excessively wide bounds are withheld. The bands are **not 95% confidence bounds**
and do not account for every systematic model error. Unvisited higher openings
stay unknown. Saturation remains unknown in this observation-only release.

`candidate_consistent` is a diagnostic comparison across episodes, not permission
to use the curve. Neither that field nor importing evidence activates control.
Boiler changes, sunlight, occupants and sensor placement can still confound room
heat estimation. Some homes or seasons may provide insufficient information.

## Persistence and migration

Shared response evidence lives under `response_learners` in the existing per-entry
Better Thermostat state store. It is keyed by room sensor, outdoor source and valve,
not target temperature. At most 24 accepted episode summaries are retained for
14 days. Pending reports/commands are deliberately not resumed after restart.

Old target-specific v1 hold evidence is preserved for rollback, but is not promoted
into new episode counts or confidence. Standalone v1 imports are bounded and
archived. Duplicate/overlapping, expired and malformed episode records are rejected.
No historical evidence is automatically injected into a running installation.

## Private replay

Export climate attributes, raw temperature, opening command, window and weather
history with `significant_changes_only=0` and an explicit end time. Then run:

```sh
.venv/bin/python scripts/replay_valve_response.py history.json replay.json \
  --climate climate.my_room --room sensor.room_temperature \
  --valve number.valve_opening --window binary_sensor.room_window \
  --weather weather.home
```

The replay is diagnostic only. Recorder does not preserve every unchanged-value
report; startup rows and asynchronous command acknowledgements limit its fidelity.
Zero accepted episodes can be the correct result. Keep household exports outside
public repositories.

## Validation and remaining work

Tests cover threshold-triggered reports, hourly heartbeats, timing jitter, noise,
sensor lag, changing valve commands, missing packets, actual outages, handling,
confounding heat, boiler variation, target changes and persistence. A separately
simulated room supplies the reference response. Controller comparison tests assert
identical valve commands with observation enabled or disabled.

A future control-enabled version needs field prediction validation and calibrated
uncertainty across independent episodes. This release intentionally makes no claim
of improved live overshoot or an identified full-flow opening. Replacing the old
learner does not by itself establish that a particular home has enough evidence.
