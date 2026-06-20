# DexBench — A 20-DOF Dexterous-Hand Manipulation & Data-Collection Benchmark (MuJoCo)

**FFAI Robothon 2026 submission.** A self-contained MuJoCo environment for a
20-DOF anthropomorphic hand, three quantitatively-scored manipulation tasks,
keyboard teleoperation, and an imitation-learning data-collection pipeline.

> **No external assets, no GPU.** The hand and scene are generated procedurally
> as a single MJCF — `pip install mujoco` and run. The 1–3 min demo video is
> produced by running the code: a no-GL **shaded-solid renderer** for headless
> machines, *or* MuJoCo's **native renderer** where a GL backend exists.

---

## 1. Quick start

```bash
python3 -m pip install -r requirements.txt   # mujoco, numpy, matplotlib, imageio[-ffmpeg]
python3 src/build_model.py                   # (re)generate assets/scene.xml
python3 src/run_sim.py --task all            # run the 3-task benchmark + log datasets
python3 src/make_video.py                    # render demo video (headless, no GPU)
python3 src/analyze.py                       # plot benchmark metrics -> media/metrics.png
```

Outputs land in `data/` (datasets + `traj.npz`) and `media/` (`demo.mp4`, `metrics.png`).

### High-quality video (machine with a display / GPU)
```bash
python3 src/record_demo.py --traj data/traj.npz --out media/demo_hq.mp4
```
Falls back gracefully with a hint if no OpenGL backend is present.

### Interactive teleoperation (needs a display)
```bash
python3 src/teleop.py
```
| Key | Action | Key | Action |
|---|---|---|---|
| `O` | open hand | `C` | caging grasp (close) |
| `P` | pinch (thumb+index) | `F` | full fist |
| `I` | point (index) | `H` | hold pose |
| `R` | reset | `SPACE` | toggle perturbation pushes |
| `ESC` | quit | | |

---

## 2. The robot

A right anthropomorphic hand, **20 actuated DOF**:

* 4 fingers × 4 DOF — `abduction, mcp, pip, dip`
* 1 thumb × 4 DOF — `cmc_abd, cmc_flex, mcp, ip`

Built entirely in MJCF (no meshes): hinge joints with limits, armature and
joint friction, capsule links, **fingertip touch sensors**, object pose /
linear-/angular-velocity sensors, `position` actuators (PD), an `elliptic`
friction cone with `multiccd` for stable box contacts, and a `ready` keyframe.

---

## 3. The three benchmark tasks

| # | Task | What it tests | Metric (validated) |
|---|---|---|---|
| 1 | **Fingertip IK tracking** | per-finger Cartesian control (differential IK) | mean **12.8 mm**, max 16.7 mm |
| 2 | **Caging grasp + hold** | multi-finger grasp, contact stability | **held** under 0.3 N multi-axis perturbation, 11 contacts |
| 3 | **Independent-finger showcase** | high-DOF coordination | per-finger flexion up to 1.2 rad |

`run_sim.py` runs them back-to-back, prints metrics, writes one combined
`traj.npz` for the renderers, and saves a labelled dataset per task.

---

## 4. Data-collection format

Each episode is logged by `src/data_collection.py` into `data/<task>.{npz,csv,json}`:

* `npz` — `time, qpos, qvel, ctrl, touch, object_pose, tip_xyz, reward`
* `csv` — per-step object pose + per-finger touch (spreadsheet-friendly)
* `json` — task, step count, duration, DOF, actuators, task metrics

The schema is generic enough for behaviour cloning / offline RL / plotting.

---

## 5. Repository layout

```
assets/scene.xml          generated MJCF (hand + cube + goal + sensors)
src/build_model.py        procedural MJCF generator (parameterised)
src/controllers.py        Grasp / FingertipIK / Choreography / Teleop
src/run_sim.py            task orchestrator + dataset writer
src/data_collection.py    EpisodeLogger (npz/csv/json)
src/make_video.py         headless shaded-solid demo renderer (no GPU)
src/record_demo.py        native MuJoCo HQ renderer (needs GL)
src/teleop.py             keyboard teleoperation (needs display)
src/analyze.py            metrics figures
data/                     datasets + traj.npz (generated)
media/                    demo.mp4 + metrics.png (generated)
```

## 6. Reproducibility notes

* Deterministic: fixed timestep `0.002 s`, `implicitfast` integrator.
* CPU-only: physics runs headless; rendering has a no-GPU fallback.
* Tested with `mujoco 3.9`, Python 3.12.

See `docs/PROJECT_WRITEUP.md` for the full project write-up.
