# Project Write-up — DexBench

**Project name:** DexBench — a 20-DOF dexterous-hand manipulation & data-collection benchmark
**Robot platform / embodiment:** anthropomorphic 5-finger dexterous hand, 20 actuated DOF, built procedurally in MJCF (no external meshes).
**Built with:** an AI coding agent (see PR description) + MuJoCo 3.9.

## Task goal
Provide a reproducible, GPU-free MuJoCo testbed for **fine multi-finger manipulation** and for **generating imitation-learning datasets**. It bundles three graded tasks that isolate the core skills of a dexterous hand: per-finger Cartesian control, grasp stability, and high-DOF coordination.

## Technical approach
* **Model.** A parameterised generator (`build_model.py`) emits a single self-contained `scene.xml`: a mounted palm with four 4-DOF fingers and a 4-DOF opposable thumb. Hinge joints carry realistic limits, armature, and joint friction; links are capsules; the manipulated cube has a free joint with colour-coded faces. Contacts use an **elliptic friction cone** with `multiccd` and tuned `solref/solimp` for stable box grasping. Sensing: fingertip **touch** sensors plus object pose / linear- / angular-velocity sensors.
* **Control.**
  * *FingertipIK* — differential (Jacobian) IK per finger via `mj_jacSite` and a damped pseudo-inverse; drives each fingertip to moving 3-D targets.
  * *GraspController* — a tuned caging synergy (fingers wrap, thumb opposes).
  * *Choreography* — scripted sequential-finger flexion (dexterity showcase).
  * *TeleopController* — discrete keyboard commands → smoothed actuator goals.
* **Position actuators** (PD) are tuned (`kp≈18–22`) for crisp tracking *and* a firm grasp.
* **Data pipeline.** `EpisodeLogger` records full observation–action pairs to `npz/csv/json` per task — ready for behaviour cloning or offline RL.
* **Rendering.** Two paths, both replaying the *same* recorded trajectory so the video is provably produced by running the submitted code. (1) A dependency-light **headless renderer** (`make_video.py`) that rebuilds the scene as depth-sorted, light-shaded 3-D solids (low-poly cylinders for the phalanges, shaded boxes for palm/cube) with a metric HUD, phase cards, fingertip markers and an orbiting camera — runs on any no-GPU / no-display machine. (2) MuJoCo's **native renderer** (`record_demo.py`) for photo-realistic output with contact-force visualisation where a GL backend exists.

## Core features
* 20-DOF hand authored 100% in MJCF — zero external assets, one-line install.
* Three quantitatively-scored tasks with logged metrics.
* Closed-loop Cartesian fingertip control (differential IK).
* Grasp that withstands multi-axis perturbation.
* Imitation-learning-ready data-collection format.
* Keyboard teleoperation.
* Headless-safe demo rendering (no GPU required).

## Validated results
| Task | Metric | Result |
|---|---|---|
| Fingertip IK tracking | mean / max Cartesian error | **12.8 mm** / 16.7 mm |
| Caging grasp + hold | object retained under 0.3 N perturbation | **yes** (11 contacts) |
| Independent fingers | max flexion achieved | 1.2 rad |

## Highlights
* Fully reproducible and CPU-only; runs identically on a laptop or headless CI.
* Clean, parameterised, well-documented code; tasks and data schema are reusable.
* Demonstrates real closed-loop control (Jacobian IK) and contact-rich grasping, not just open-loop playback.

## Current limitations
* **In-hand reorientation** (continuous object rotation by finger gaiting) was prototyped but open-loop gaiting achieves <3° net rotation — it realistically needs a learned policy (RL). It is intentionally excluded from the scored tasks.
* The headless renderer draws shaded 3-D solids (not a physically-based render); for photo-realistic frames with contact forces, run the native MuJoCo renderer (`record_demo.py`) on a machine with a GL backend.
* Grasp robustness was tuned for a light cube (~50 g); heavier/odd-shaped objects would need re-tuning.

## Future work
* Train an RL / model-based policy for true in-hand reorientation using the included data pipeline.
* Add tactile-servoing grasp adaptation using the fingertip touch sensors in the loop.
* Extend the task suite (button panel, tool use) and add a gym-style env wrapper.
