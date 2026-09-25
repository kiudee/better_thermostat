"""Replay a private HA Recorder export without changing Home Assistant.

Only data collected during MPC V2 operation are eligible. Output contains
portable learner evidence and a rejection summary; household history should
stay outside a public repository.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from custom_components.better_thermostat.utils.calibration.mpc_v2.response import (
    ResponseObservation,
    ValveResponseLearner,
)


def replay(history, *, climate, room, valve, window, weather):
    """Replay known commands and fresh reports through the live acceptance gates."""
    events = {}
    for rows in history:
        for row in rows:
            t = datetime.fromisoformat(
                row["last_updated"].replace("Z", "+00:00")
            ).timestamp()
            events.setdefault(t, []).append(row)
    learner = ValveResponseLearner()
    states = {}
    reports = Counter()
    for t, rows in sorted(events.items()):
        for row in rows:
            states[row["entity_id"]] = (t, row)
        if not all(k in states for k in (climate, room, valve, window, weather)):
            reports["missing_entity"] += 1
            continue
        try:
            rt, rs = states[room]
            wt, ws = states[weather]
            c = states[climate][1]["attributes"]
            temp = float(rs["state"])
            outdoor = float(ws["attributes"]["temperature"])
            if rs.get("attributes", {}).get("unit_of_measurement") == "°F":
                temp = (temp - 32) * 5 / 9
            if ws["attributes"].get("temperature_unit") == "°F":
                outdoor = (outdoor - 32) * 5 / 9
            opening = float(states[valve][1]["state"])
            command = float(c["mpc_v2_group_valve_pct"])
            methods = c.get("valve_method", {})
            known_source = isinstance(methods, dict) and len(methods) == 1
            if known_source:
                learner.bind_source("|".join((room, weather, next(iter(methods)))), t)
            contact_changed = datetime.fromisoformat(
                states[window][1]["last_changed"].replace("Z", "+00:00")
            ).timestamp()
            valid = (
                known_source
                and states[window][1]["state"] == "off"
                and t - contact_changed >= 45 * 60
                and not c.get("degraded_mode", False)
                and not c.get("window_open", False)
                and not c.get("door_open", False)
                and not c.get("unavailable_sensors", [])
                and t - wt <= 5400
                and abs(opening - command) <= 1
            )
            learner.observe(ResponseObservation(t, rt, temp, outdoor, opening, valid))
        except ValueError, KeyError, TypeError:
            learner.interrupt(t, "missing_or_invalid_state")
        reports[learner.status] += 1
    return {
        "evidence": learner.export(),
        "diagnostics": learner.diagnostics(),
        "status_counts": dict(reports),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("history", type=Path)
    parser.add_argument("output", type=Path)
    for key in ("climate", "room", "valve", "window", "weather"):
        parser.add_argument("--" + key, required=True)
    args = parser.parse_args()
    raw = args.history.read_bytes()
    result = replay(
        json.loads(raw),
        **{
            key: getattr(args, key)
            for key in ("climate", "room", "valve", "window", "weather")
        },
    )
    result["source_sha256"] = hashlib.sha256(raw).hexdigest()
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                "accepted_independent_holds": sum(
                    result["diagnostics"]["independent_holds"]
                ),
                "status_counts": result["status_counts"],
            },
            indent=2,
        )
    )
