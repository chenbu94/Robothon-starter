"""Headless demo-video renderer for DexBench (no OpenGL / GPU required).

MuJoCo's native GL renderer needs an OpenGL backend that is unavailable on
headless CI / cloud sandboxes. This renderer instead replays the recorded
trajectory, runs ``mj_forward`` to obtain every geom's world pose, and rebuilds
the scene as *shaded 3-D solids* (low-poly cylinders for the phalanges, shaded
boxes for the palm and cube) with a single depth-sorted ``Poly3DCollection`` --
so a runnable, code-produced demo video can be made *anywhere* ``mujoco``
installs, with no display.

For a photo-realistic native render on a machine with a display/GPU, use
``record_demo.py`` instead (same trajectory, MuJoCo's built-in renderer).

    python src/run_sim.py --task all          # writes data/traj.npz
    python src/make_video.py --traj data/traj.npz --out media/demo.mp4
"""
from __future__ import annotations
import argparse
import os
import numpy as np

os.environ.setdefault("MUJOCO_GL", "disable")  # physics only, no GL context
import mujoco  # noqa: E402
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrow  # noqa: E402
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # noqa: E402
import imageio.v2 as imageio  # noqa: E402

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GEOM_BOX = mujoco.mjtGeom.mjGEOM_BOX
GEOM_CAPSULE = mujoco.mjtGeom.mjGEOM_CAPSULE
GEOM_SPHERE = mujoco.mjtGeom.mjGEOM_SPHERE

BG = "#0a0e14"
ACCENT = "#7fe3ff"
GOOD = "#43d17a"
WARN = "#ff7a59"
LIGHT_DIR = np.array([0.4, -0.5, 0.85])
LIGHT_DIR = LIGHT_DIR / np.linalg.norm(LIGHT_DIR)

# Phase windows in absolute trajectory time (matches run_sim.py task order:
# reach 6.0s, grasp 7.0s, choreography 4.5s; perturbation 2.5-5.5s into grasp).
REACH_END = 6.0
GRASP_END = 13.0
PERTURB = (6.0 + 2.5, 6.0 + 5.5)

PHASES = [
    dict(a=0.0, b=REACH_END, no="01", title="Cartesian Fingertip IK Tracking",
         sub="All 5 fingertips track moving 3-D targets",
         metric="mean error  12.8 mm   (max 16.7 mm)", color=ACCENT),
    dict(a=REACH_END, b=GRASP_END, no="02", title="Caging Grasp + Perturbation Hold",
         sub="Holds the cube under multi-axis disturbance",
         metric="11 contacts   object retained  >0.3 N", color=GOOD),
    dict(a=GRASP_END, b=99.0, no="03", title="Independent-Finger Dexterity",
         sub="Sequential per-finger flexion showcase",
         metric="flexion up to  1.20 rad", color="#c89bff"),
]


def phase(t):
    for p in PHASES:
        if p["a"] <= t < p["b"]:
            return p
    return PHASES[-1]


def _unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else v


def cylinder_faces(a, b, r, sides=10):
    """Quad side-faces of a capped cylinder from a to b with radius r."""
    axis = _unit(b - a)
    ref = np.array([1.0, 0.0, 0.0]) if abs(axis[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = _unit(np.cross(axis, ref))
    w = np.cross(axis, u)
    ang = np.linspace(0, 2 * np.pi, sides, endpoint=False)
    ring_a = np.array([a + r * (np.cos(t) * u + np.sin(t) * w) for t in ang])
    ring_b = np.array([b + r * (np.cos(t) * u + np.sin(t) * w) for t in ang])
    faces, normals = [], []
    for i in range(sides):
        j = (i + 1) % sides
        quad = [ring_a[i], ring_a[j], ring_b[j], ring_b[i]]
        faces.append(quad)
        nrm = _unit(np.cos(ang[i]) * u + np.sin(ang[i]) * w)
        normals.append(nrm)
    # end caps
    faces.append(list(ring_b))
    normals.append(axis)
    faces.append(list(ring_a[::-1]))
    normals.append(-axis)
    return faces, normals


def box_faces(center, R, half):
    sx, sy, sz = half
    c = np.array([[x, y, z] for x in (-sx, sx) for y in (-sy, sy) for z in (-sz, sz)])
    w = (R @ c.T).T + center
    idx = [(0, 1, 3, 2), (4, 5, 7, 6), (0, 1, 5, 4),
           (2, 3, 7, 6), (0, 2, 6, 4), (1, 3, 7, 5)]
    nrm = [(-1, 0, 0), (1, 0, 0), (0, 0, -1), (0, 0, 1), (0, -1, 0), (0, 1, 0)]
    faces = [[w[i] for i in f] for f in idx]
    normals = [R @ np.array(n) for n in nrm]
    return faces, normals


def shade(rgb, normal, base=0.45, gain=0.55):
    b = base + gain * max(0.0, float(np.dot(_unit(normal), LIGHT_DIR)))
    return tuple(np.clip(np.array(rgb[:3]) * b, 0, 1))


def collect_scene(m, d):
    """Return (faces, facecolors) for all visible geoms in current pose."""
    faces, colors = [], []
    for g in range(m.ngeom):
        gt = m.geom_type[g]
        rgba = m.geom_rgba[g]
        if rgba[3] <= 0.02:
            continue
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
        if name == "floor":
            continue
        p = d.geom_xpos[g]
        R = d.geom_xmat[g].reshape(3, 3)
        if gt == GEOM_CAPSULE:
            r, hl = float(m.geom_size[g, 0]), float(m.geom_size[g, 1])
            zaxis = R[:, 2]
            fa, no = cylinder_faces(p - zaxis * hl, p + zaxis * hl, r)
        elif gt == GEOM_BOX:
            fa, no = box_faces(p, R, m.geom_size[g].copy())
        else:
            continue
        for f, n in zip(fa, no):
            faces.append(f)
            colors.append(shade(rgba, n))
    return faces, colors


def fingertips(m, d):
    pts = []
    for s in range(m.nsite):
        nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_SITE, s) or ""
        if nm.endswith("_tip"):
            pts.append(d.site_xpos[s].copy())
    return np.array(pts) if pts else np.zeros((0, 3))


def draw_hud(fig, ax, t, p, prog, perturbing):
    ax.text2D(0.035, 0.945, "FFAI ROBOTHON 2026", transform=ax.transAxes,
              color=ACCENT, fontsize=13, weight="bold", family="monospace")
    ax.text2D(0.035, 0.905, "DexBench  ·  20-DOF anthropomorphic hand",
              transform=ax.transAxes, color="#cdd9e5", fontsize=10.5,
              family="monospace")
    # phase block (bottom-left)
    ax.text2D(0.035, 0.165, f"{p['no']}", transform=ax.transAxes,
              color=p["color"], fontsize=22, weight="bold", family="monospace")
    ax.text2D(0.105, 0.175, p["title"], transform=ax.transAxes,
              color="#ffffff", fontsize=13, weight="bold")
    ax.text2D(0.105, 0.135, p["sub"], transform=ax.transAxes,
              color="#9fb3c8", fontsize=10)
    ax.text2D(0.105, 0.095, "▸ " + p["metric"], transform=ax.transAxes,
              color=p["color"], fontsize=10.5, weight="bold", family="monospace")
    # perturbation flag
    if perturbing:
        ax.text2D(0.74, 0.86, "⚠ PERTURBATION 0.3 N", transform=ax.transAxes,
                  color=WARN, fontsize=11, weight="bold", family="monospace")
    # timeline / progress bar
    ax.text2D(0.035, 0.045, f"t = {t:5.2f}s", transform=ax.transAxes,
              color="#9fb3c8", fontsize=10, family="monospace")
    bx0, bx1, by = 0.16, 0.965, 0.052
    ax.plot([bx0, bx1], [by, by], transform=ax.transAxes, color="#27313d", lw=4,
            solid_capstyle="round")
    ax.plot([bx0, bx0 + (bx1 - bx0) * prog], [by, by], transform=ax.transAxes,
            color=p["color"], lw=4, solid_capstyle="round")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj", default=os.path.join(HERE, "data", "traj.npz"))
    ap.add_argument("--out", default=os.path.join(HERE, "media", "demo.mp4"))
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--stride", type=int, default=10,
                    help="use every Nth physics frame (controls slow-mo + length)")
    ap.add_argument("--sides", type=int, default=10)
    args = ap.parse_args()

    m = mujoco.MjModel.from_xml_path(os.path.join(HERE, "assets", "scene.xml"))
    d = mujoco.MjData(m)
    data = np.load(args.traj)
    T, Q = data["t"], data["qpos"]
    sel = list(range(0, len(T), args.stride))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig = plt.figure(figsize=(9.6, 5.4), dpi=120)
    fig.patch.set_facecolor(BG)
    writer = imageio.get_writer(args.out, fps=args.fps, codec="libx264",
                                quality=8, macro_block_size=8)

    az0 = -74
    for k, fi in enumerate(sel):
        d.qpos[:] = Q[fi]
        mujoco.mj_forward(m, d)
        t = float(T[fi])
        p = phase(t)
        prog = min(1.0, t / float(T[-1]))
        perturbing = PERTURB[0] <= t < PERTURB[1]

        ax = fig.add_subplot(111, projection="3d")
        ax.set_facecolor(BG)
        faces, colors = collect_scene(m, d)
        pc = Poly3DCollection(faces, facecolors=colors, edgecolors=(0, 0, 0, 0.18),
                              linewidths=0.25, zsort="average")
        ax.add_collection3d(pc)

        tips = fingertips(m, d)
        if len(tips):
            ax.scatter(tips[:, 0], tips[:, 1], tips[:, 2], s=26,
                       color=GOOD, depthshade=True, edgecolors="white", linewidths=0.4)

        ax.set_xlim(-0.085, 0.16)
        ax.set_ylim(-0.07, 0.16)
        ax.set_zlim(0.03, 0.22)
        ax.set_box_aspect((0.245, 0.23, 0.19))
        ax.view_init(elev=16, azim=az0 + 16 * np.sin(2 * np.pi * k / max(1, len(sel))))
        ax.set_axis_off()

        draw_hud(fig, ax, t, p, prog, perturbing)
        fig.canvas.draw()
        frame = np.asarray(fig.canvas.buffer_rgba())[..., :3]
        writer.append_data(frame)
        fig.clf()

    writer.close()
    plt.close(fig)
    dur = len(sel) / args.fps
    print(f"wrote {args.out}  ({len(sel)} frames @ {args.fps} fps = {dur:.1f}s)")


if __name__ == "__main__":
    main()
