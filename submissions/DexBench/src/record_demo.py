"""Photo-realistic demo recorder using MuJoCo's native renderer.

Run this on any machine with a working OpenGL backend (a normal desktop, or a
GPU/EGL cloud box). It replays the trajectory recorded by ``run_sim.py`` and
writes a high-quality MP4 with MuJoCo's built-in renderer.

    python src/run_sim.py --task all          # produces data/traj.npz
    python src/record_demo.py --traj data/traj.npz --out media/demo_hq.mp4

If no display is available the script prints a clear hint to use
``make_video.py`` (the matplotlib fallback) instead.
"""
from __future__ import annotations
import argparse
import os
import sys
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument("--traj", default=os.path.join(here, "data", "traj.npz"))
    ap.add_argument("--out", default=os.path.join(here, "media", "demo_hq.mp4"))
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--stride", type=int, default=15)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    args = ap.parse_args()

    import mujoco
    try:
        import imageio.v2 as imageio
    except ImportError:
        sys.exit("pip install imageio imageio-ffmpeg")

    m = mujoco.MjModel.from_xml_path(os.path.join(here, "assets", "scene.xml"))
    d = mujoco.MjData(m)
    try:
        renderer = mujoco.Renderer(m, args.height, args.width)
    except Exception as e:  # no GL backend
        sys.exit(f"MuJoCo renderer unavailable ({e}).\n"
                 f"Set MUJOCO_GL=egl|glfw|osmesa, or use make_video.py "
                 f"(matplotlib fallback) on headless machines.")

    data = np.load(args.traj)
    T, Q = data["t"], data["qpos"]

    cam = mujoco.MjvCamera()
    cam.lookat[:] = [0.03, 0.04, 0.13]
    cam.distance, cam.azimuth, cam.elevation = 0.45, -70, -12

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    writer = imageio.get_writer(args.out, fps=args.fps, codec="libx264",
                                quality=9, macro_block_size=8)
    opt = mujoco.MjvOption()
    opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
    opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = True
    for fi in range(0, len(T), args.stride):
        d.qpos[:] = Q[fi]
        mujoco.mj_forward(m, d)
        renderer.update_scene(d, camera=cam, scene_option=opt)
        writer.append_data(renderer.render())
    writer.close()
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
