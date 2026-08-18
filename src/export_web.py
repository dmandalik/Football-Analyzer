"""Export fitted trajectories to static JSON for the web viewer.

Reads every fit in data/fits/, simulates the best-fit flight and the
bootstrap bundle densely, and writes web/data/kicks.json. The frontend
does zero inference — everything it draws is precomputed here.

Run from repo root:  python -m src.export_web
"""

import glob
import json
import os

import numpy as np

from src.physics import simulate
from src.fitting import flight_quantities

# display metadata per clip; quantities come from the fit files
META = {
    "messi_live": {"player": "Lionel Messi", "match": "vs Liverpool (UCL 2019)",
                   "foot": "L"},
    "messi2": {"player": "Lionel Messi",
               "match": "vs Cruz Azul (Leagues Cup 2023)", "foot": "L"},
}


def flight_time(p0, theta):
    xyz = simulate(p0, theta[:3], theta[3:], np.linspace(0, 2.5, 500))
    below = xyz[:, 1] <= 0
    i = int(np.argmax(below)) if below.any() else len(xyz) - 1
    return max(0.35, 2.5 * i / 499)


def path(p0, theta, t_end, n=120):
    xyz = simulate(p0, theta[:3], theta[3:], np.linspace(0.0, t_end, n))
    return np.round(xyz, 3).tolist()


def main():
    kicks = []
    for fp in sorted(glob.glob("data/fits/*.json")):
        fit = json.load(open(fp))
        cid = fit["clip_id"]
        if cid not in META:
            print(f"skipping {cid} (no display metadata)")
            continue
        theta = np.array(fit["theta6"])
        p0 = np.array(fit["p0"])
        t_end = flight_time(p0, theta)
        q = fit["quantities"]
        samples = [np.array(s) for s in fit.get("samples", [])[:20]]
        kicks.append({
            "id": cid, **META[cid],
            "launch": p0.tolist(),
            "flight_s": round(t_end, 3),
            "speed_ms": round(q["speed [m/s]"], 1),
            "speed_kmh": round(q["speed [m/s]"] * 3.6, 1),
            "elevation_deg": round(q["elevation [deg]"], 1),
            "azimuth_deg": round(q["azimuth [deg]"], 1),
            "spin_rpm": round(q["w_perp [rpm]"], 0),
            "spin_interval": [round(v, 0) for v in fit["intervals"][3]],
            "distance_m": round(float(np.hypot(p0[0], p0[1])), 1),
            "crossing": fit["box"][:2],
            # keeper reachability at the crossing: reaction 0.30 s, then an
            # effective dive that extends reach at 3.2 m/s from a 0.85 m
            # standing envelope, capped at a full-stretch 3.4 m. Model
            # parameters are displayed in the UI, not hidden.
            "reach_m": round(min(3.4, 0.85 + 3.2 * max(0.0, t_end - 0.30)), 2),
            "path": path(p0, theta, t_end),
            "bundle": [path(p0, s, t_end, 60) for s in samples],
        })
        print(f"{cid}: {q['speed [m/s]']:.1f} m/s, flight {t_end:.2f}s, "
              f"{len(samples)} bundle paths")
    os.makedirs("web/data", exist_ok=True)
    json.dump({"kicks": kicks}, open("web/data/kicks.json", "w"))
    print(f"wrote web/data/kicks.json ({len(kicks)} kicks)")


if __name__ == "__main__":
    main()
