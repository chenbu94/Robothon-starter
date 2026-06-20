"""Headless demo-video renderer (no OpenGL / GPU required).

MuJoCo's GL renderer needs an OpenGL backend that is unavailable on headless
CI / cloud sandboxes. This script instead replays the recorded trajectory,
runs ``mj_forward`` to obtain every geom's world pose, and draws the scene with
matplotlib 3-D primitives -- so a runnable, code-produced demo video can be made
*anywhere* ``mujoco`` installs.

For a photo-realistic render on a machine with a display/GPU, use
``record_demo.py`` instead (same trajectory, MuJoCo's native renderer).

    python src/make_video.py --traj data/traj.npz --out media/demo.mp4
"""
from __future__ import annotations
import argparse
import os
import numpy as np

os.environ.setdefault("MUJOCO_GL", "disable")
import mujoco  # noqa: E402
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection  # noqa: E402
import imageio.v2 as imageio  # noqa: E402

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GEOM_BOX, GEOM_CAPSULE = mujoco.mjtGeom.mjGEOM_BOX, mujoco.mjtGeom.mjGEOM_CAPSULE

PHASES = [(0.0, 6.0, "1 - Cartesian fingertip IK tracking (5 fingers)"),
          (6.0, 13.0, "2 - Caging grasp + perturbation hold"),
          (13.0, 99.0, "3 - Independent-finger dexterity showcase")]


def box_faces(center, R, half):
    sx, sy, sz = half
    c = np.array([[x, y, z] for x in (-sx, sx) for y in (-sy, sy) for z in (-sz, sz)])
    w = (R @ c.T).T + center
    idx = [(0, 1, 3, 2), (4, 5, 7, 6), (0, 1, 5, 4),
           (2, 3, 7, 6), (0, 2, 6, 4), (1, 3, 7, 5)]
    return [[w[i] for i in f] for f in idx]


def phase_label(t):
    for a, b, lab in PHASES:
        if a <= t < b:
            return lab
    return PHASES[-1][2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj", default=os.path.join(HERE, "data", "traj.npz"))
    ap.add_argument("--out", default=os.path.join(HERE, "media", "demo.mp4"))
    ap.add_argument("--fps", type=int, default=25)
    ap.add_argument("--stride", type=int, default=18)
    args = ap.parse_args()

    m = mujoco.MjModel.from_xml_path(os.path.join(HERE, "assets", "scene.xml"))
    d = mujoco.MjData(m)
    data = np.load(args.traj)
    T, Q = data["t"], data["qpos"]
    sel = range(0, len(T), args.stride)

    # classify geoms once
    caps, boxes = [], []
    for g in range(m.ngeom):
        gt = m.geom_type[g]
        if m.geom_group[g] == 3:   # skip helper sites if any
            continue
        rgba = m.geom_rgba[g]
        if gt == GEOM_CAPSULE:
            caps.append((g, m.geom_size[g, 0], m.geom_size[g, 1], rgba))
        elif gt == GEOM_BOX:
            boxes.append((g, m.geom_size[g].copy(), rgba))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig = plt.figure(figsize=(9, 6), dpi=110)
    fig.patch.set_facecolor("#0b0f14")
    writer = imageio.get_writer(args.out, fps=args.fps, codec="libx264",
                                quality=8, macro_block_size=8)

    for fi in sel:
        d.qpos[:] = Q[fi]
        mujoco.mj_forward(m, d)
        ax = fig.add_subplot(111, projection="3d")
        ax.set_facecolor("#0b0f14")

        segs, lws, cols = [], [], []
        for g, r, hl, rgba in caps:
            p = d.geom_xpos[g]; zaxis = d.geom_xmat[g].reshape(3, 3)[:, 2]
            a, b = p - zaxis_safe(zaxis) * hl, p + zaxis_safe(zaxis) * hl
            segs.append([a, b]); lws.append(max(2.0, r * 900)); cols.append(rgba)
        lc = Line3DCollection(segs, linewidths=lws,
                              colors=[(*c[:3], 1.0) for c in cols])
        ax.add_collection3d(lc)

        for g, half, rgba in boxes:
            p = d.geom_xpos[g]; R = d.geom_xmat[g].reshape(3, 3)
            alpha = float(rgba[3])
            if alpha <= 0.01:
                continue
            pc = Poly3DCollection(box_faces(p, R, half), alpha=min(alpha, 0.95))
            pc.set_facecolor((*rgba[:3],)); pc.set_edgecolor((0, 0, 0, 0.3))
            ax.add_collection3d(pc)

        ax.set_xlim(-0.08, 0.16); ax.set_ylim(-0.06, 0.16); ax.set_zlim(0.02, 0.22)
        ax.set_box_aspect((0.24, 0.22, 0.20))
        ax.view_init(elev=18, azim=-72)
        ax.set_axis_off()
        ax.text2D(0.03, 0.95, "FFAI Robothon 2026 - DexBench (20-DOF hand)",
                  transform=ax.transAxes, color="#7fe3ff", fontsize=12, weight="bold")
        ax.text2D(0.03, 0.90, phase_label(T[fi]), transform=ax.transAxes,
                  color="#e8e8e8", fontsize=11)
        ax.text2D(0.03, 0.04, f"t = {T[fi]:5.2f}s", transform=ax.transAxes,
                  color="#9fb3c8", fontsize=10)

        fig.canvas.draw()
        frame = np.asarray(fig.canvas.buffer_rgba())[..., :3]
        writer.append_data(frame)
        fig.clf()

    writer.close()
    plt.close(fig)
    print(f"wrote {args.out} ({len(list(sel))} frames @ {args.fps} fps)")


def zaxis_safe(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else v


if __name__ == "__main__":
    main()
