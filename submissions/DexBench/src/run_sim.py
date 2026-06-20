"""DexBench task orchestrator (headless, no GPU needed).

Runs one or more dexterous-manipulation tasks, prints quantitative metrics,
records imitation-learning datasets (via EpisodeLogger) and dumps a qpos/time
trajectory used by the renderers (``make_video.py`` / ``record_demo.py``).

Tasks
-----
reach        Per-finger Cartesian IK: all five fingertips track moving 3-D
             targets. Metric: mean / max Cartesian tracking error (mm).
grasp        Caging grasp on the cube, then resist a 3-second multi-axis
             perturbation. Metric: object retained (held) + min height.
choreography Scripted sequential-finger dexterity showcase. Metric: per-finger
             flexion range achieved.
all          Run the full sequence back-to-back into one trajectory (default).

Usage
-----
    python src/run_sim.py --task all --out data --record traj.npz
"""
from __future__ import annotations
import argparse
import os
import sys
import numpy as np

os.environ.setdefault("MUJOCO_GL", "disable")  # headless: physics only
import mujoco  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from controllers import (GraspController, FingertipIK, Choreography,  # noqa: E402
                         ALL_FINGERS, CTRL_LO, CTRL_HI)
from data_collection import EpisodeLogger  # noqa: E402

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCENE = os.path.join(HERE, "assets", "scene.xml")


def load():
    m = mujoco.MjModel.from_xml_path(SCENE)
    d = mujoco.MjData(m)
    return m, d


def quat_to_yaw(q):
    w, x, y, z = q
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


# --------------------------------------------------------------------------- #
# Tasks
# --------------------------------------------------------------------------- #
def task_reach(m, d, traj, dur=6.0, R=0.015):
    """All five fingertips track circular targets in their reachable arc."""
    ik = FingertipIK(m, d)
    log = EpisodeLogger(m, d, "reach")
    # pre-curl every finger to mid-workspace
    for f in ALL_FINGERS:
        for a in ik.finger_act(f):
            d.ctrl[a] = 0.6
    for _ in range(int(0.6 / m.opt.timestep)):
        mujoco.mj_step(m, d)
    centers = {f: ik.site_pos(f) for f in ALL_FINGERS}
    errs = []
    t0 = traj["t"][-1] if traj["t"] else 0.0
    n = int(dur / m.opt.timestep)
    for i in range(n):
        t = i * m.opt.timestep
        e_step = []
        for k, f in enumerate(ALL_FINGERS):
            ph = 1.5 * t + k * 0.6
            tgt = centers[f] + np.array([0.006 * np.sin(ph),
                                         R * np.cos(ph) - R,
                                         R * np.sin(ph)])
            ik.apply(f, tgt)
            e_step.append(np.linalg.norm(tgt - ik.site_pos(f)))
        mujoco.mj_step(m, d)
        errs.append(np.mean(e_step))
        log.record(t, reward=-np.mean(e_step))
        _push(traj, t0 + t, d)
    errs = np.array(errs)
    metrics = dict(mean_err_mm=float(errs[int(0.3 * n):].mean() * 1000),
                   max_err_mm=float(errs.max() * 1000))
    return log, metrics


def task_grasp(m, d, traj, perturb_N=0.3, dur=7.0):
    g = GraspController(m)
    log = EpisodeLogger(m, d, "grasp")
    mujoco.mj_resetDataKeyframe(m, d, 0)
    oid = m.body("object").id
    ctrl = np.array(d.ctrl)
    held, minz = True, 1.0
    t0 = traj["t"][-1] if traj["t"] else 0.0
    for i in range(int(dur / m.opt.timestep)):
        t = i * m.opt.timestep
        ctrl = g(ctrl, t)
        d.ctrl[:] = np.clip(ctrl, CTRL_LO, CTRL_HI)
        d.xfrc_applied[oid, :3] = 0.0
        if 2.5 < t < 5.5:
            d.xfrc_applied[oid, :3] = perturb_N * np.array(
                [np.sin(2.1 * t), np.cos(1.7 * t), 0.5 * np.sin(3 * t)])
        mujoco.mj_step(m, d)
        z = d.body("object").xpos[2]
        minz = min(minz, z)
        held = held and z > 0.04
        log.record(t, reward=1.0 if z > 0.04 else 0.0)
        _push(traj, t0 + t, d)
    d.xfrc_applied[:] = 0.0
    ncon = int(d.ncon)
    metrics = dict(held=bool(held), min_height_m=float(minz),
                   perturb_N=perturb_N, contacts=ncon)
    return log, metrics


def task_choreography(m, d, traj, dur=4.5):
    ch = Choreography(m)
    log = EpisodeLogger(m, d, "choreography")
    mujoco.mj_resetData(m, d)
    ctrl = np.zeros(m.nu)
    qmax = np.zeros(m.nq)
    t0 = traj["t"][-1] if traj["t"] else 0.0
    for i in range(int(dur / m.opt.timestep)):
        t = i * m.opt.timestep
        ctrl = ch(ctrl, t)
        d.ctrl[:] = np.clip(ctrl, CTRL_LO, CTRL_HI)
        mujoco.mj_step(m, d)
        qmax = np.maximum(qmax, np.abs(d.qpos))
        log.record(t)
        _push(traj, t0 + t, d)
    metrics = dict(max_flex_rad=float(qmax[:20].max()))
    return log, metrics


def _push(traj, t, d):
    traj["t"].append(float(t))
    traj["qpos"].append(d.qpos.copy())


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description="DexBench task orchestrator")
    ap.add_argument("--task", default="all",
                    choices=["reach", "grasp", "choreography", "all"])
    ap.add_argument("--out", default=os.path.join(HERE, "data"))
    ap.add_argument("--record", default=os.path.join(HERE, "data", "traj.npz"),
                    help="where to dump the (t, qpos) trajectory for rendering")
    args = ap.parse_args()

    m, d = load()
    traj = {"t": [], "qpos": []}
    order = (["reach", "grasp", "choreography"] if args.task == "all"
             else [args.task])
    runners = dict(reach=task_reach, grasp=task_grasp,
                   choreography=task_choreography)

    print(f"DexBench | DOF={m.nv}  actuators={m.nu}  sensors={m.nsensordata}")
    print("-" * 56)
    for name in order:
        log, metrics = runners[name](m, d, traj)
        log.save(args.out, name, meta=dict(metrics=metrics))
        print(f"[{name:12s}] " +
              "  ".join(f"{k}={v}" for k, v in metrics.items()))

    os.makedirs(os.path.dirname(args.record), exist_ok=True)
    np.savez_compressed(args.record,
                        t=np.array(traj["t"]),
                        qpos=np.array(traj["qpos"]))
    print("-" * 56)
    print(f"datasets -> {args.out}/  |  trajectory -> {args.record} "
          f"({len(traj['t'])} frames)")


if __name__ == "__main__":
    main()
