"""Native MuJoCo demo recorder WITH on-frame HUD overlay.

Renders the recorded trajectory with MuJoCo's built-in (photo-realistic)
renderer -- shadows, materials, contact points and contact forces -- then draws
an informative HUD on every frame (phase title, validated metric, perturbation
flag, timeline) with Pillow. This combines the high-fidelity native look with
the explanatory captions, so a viewer immediately understands each phase.

Run on any machine with a GL backend:

    python src/run_sim.py --task all                       # writes data/traj.npz
    MUJOCO_GL=glfw python src/record_demo.py --stride 5 --fps 20   # desktop
    MUJOCO_GL=egl  python src/record_demo.py --stride 5 --fps 20   # headless GPU
    MUJOCO_GL=osmesa python src/record_demo.py --stride 5 --fps 20 # headless CPU

If no GL backend is available, use make_video.py (no-GL shaded-solid renderer).
"""
from __future__ import annotations
import argparse
import os
import sys
import numpy as np

# ---- HUD config (kept in sync with make_video.py) ------------------------- #
ACCENT = (127, 227, 255)
GOOD = (67, 209, 122)
PURPLE = (200, 155, 255)
WARN = (255, 122, 89)
WHITE = (255, 255, 255)
MUTE = (159, 179, 200)

REACH_END, GRASP_END = 6.0, 13.0
PERTURB = (6.0 + 2.5, 6.0 + 5.5)
PHASES = [
    dict(a=0.0, b=REACH_END, no="01", title="Cartesian Fingertip IK Tracking",
         sub="All 5 fingertips track moving 3-D targets",
         metric="mean error 12.8 mm  (max 16.7 mm)", color=ACCENT),
    dict(a=REACH_END, b=GRASP_END, no="02", title="Caging Grasp + Perturbation Hold",
         sub="Holds the cube under multi-axis disturbance",
         metric="11 contacts   object retained >0.3 N", color=GOOD),
    dict(a=GRASP_END, b=99.0, no="03", title="Independent-Finger Dexterity",
         sub="Sequential per-finger flexion showcase",
         metric="flexion up to 1.20 rad", color=PURPLE),
]


def phase(t):
    for p in PHASES:
        if p["a"] <= t < p["b"]:
            return p
    return PHASES[-1]


def _load_fonts(W):
    from PIL import ImageFont
    s = W / 1280.0
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    path = next((c for c in candidates if os.path.exists(c)), None)

    def f(sz):
        if path:
            return ImageFont.truetype(path, int(sz * s))
        return ImageFont.load_default()
    return dict(brand=f(26), sub=f(17), no=f(46), title=f(27),
               small=f(19), metric=f(20), tiny=f(17))


def draw_hud(frame, t, t_end):
    """frame: HxWx3 uint8 -> returns HxWx3 uint8 with HUD drawn."""
    from PIL import Image, ImageDraw
    img = Image.fromarray(frame).convert("RGBA")
    W, H = img.size
    ft = _load_fonts(W)
    ov = Image.new("RGBA", img.size, (0, 0, 0, 0))
    dr = ImageDraw.Draw(ov)
    p = phase(t)
    pad = int(0.028 * W)

    # subtle gradient scrims for legibility (top + bottom)
    band = int(0.16 * H)
    for i in range(band):
        a = int(150 * (1 - i / band))
        dr.line([(0, i), (W, i)], fill=(6, 10, 20, a))
        dr.line([(0, H - 1 - i), (W, H - 1 - i)], fill=(6, 10, 20, a))

    # top-left brand
    dr.text((pad, int(0.045 * H)), "FFAI ROBOTHON 2026", font=ft["brand"], fill=ACCENT + (255,))
    dr.text((pad, int(0.045 * H) + int(0.045 * H)),
            "DexBench  -  20-DOF anthropomorphic hand", font=ft["sub"], fill=MUTE + (255,))

    # perturbation flag (top-right)
    if PERTURB[0] <= t < PERTURB[1]:
        msg = "! PERTURBATION 0.3 N"
        w = dr.textlength(msg, font=ft["small"])
        dr.text((W - pad - w, int(0.06 * H)), msg, font=ft["small"], fill=WARN + (255,))

    # bottom-left phase block
    by = int(0.74 * H)
    dr.text((pad, by), p["no"], font=ft["no"], fill=p["color"] + (255,))
    tx = pad + int(0.075 * W)
    dr.text((tx, by + int(0.005 * H)), p["title"], font=ft["title"], fill=WHITE + (255,))
    dr.text((tx, by + int(0.06 * H)), p["sub"], font=ft["small"], fill=MUTE + (255,))
    dr.text((tx, by + int(0.105 * H)), "> " + p["metric"], font=ft["metric"],
            fill=p["color"] + (255,))

    # timeline progress bar + clock
    prog = min(1.0, t / max(1e-6, t_end))
    bx0, bx1 = pad, W - pad
    yb = int(0.955 * H)
    dr.line([(bx0, yb), (bx1, yb)], fill=(39, 49, 61, 255), width=max(3, int(0.006 * H)))
    dr.line([(bx0, yb), (bx0 + (bx1 - bx0) * prog, yb)], fill=p["color"] + (255,),
            width=max(3, int(0.006 * H)))
    dr.text((pad, yb - int(0.05 * H)), f"t = {t:5.2f}s", font=ft["tiny"], fill=MUTE + (255,))

    out = Image.alpha_composite(img, ov).convert("RGB")
    return np.asarray(out)


def main():
    ap = argparse.ArgumentParser()
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument("--traj", default=os.path.join(here, "data", "traj.npz"))
    ap.add_argument("--out", default=os.path.join(here, "media", "demo.mp4"))
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--no-hud", action="store_true", help="render without the text overlay")
    args = ap.parse_args()

    import mujoco
    try:
        import imageio.v2 as imageio
    except ImportError:
        sys.exit("pip install imageio imageio-ffmpeg")
    try:
        from PIL import Image  # noqa: F401
    except ImportError:
        sys.exit("pip install pillow")

    m = mujoco.MjModel.from_xml_path(os.path.join(here, "assets", "scene.xml"))
    d = mujoco.MjData(m)
    try:
        renderer = mujoco.Renderer(m, args.height, args.width)
    except Exception as e:  # no GL backend
        sys.exit(f"MuJoCo renderer unavailable ({e}).\n"
                 f"Set MUJOCO_GL=egl|glfw|osmesa, or use make_video.py "
                 f"(no-GL renderer) on headless machines.")

    data = np.load(args.traj)
    T, Q = data["t"], data["qpos"]
    t_end = float(T[-1])

    cam = mujoco.MjvCamera()
    cam.lookat[:] = [0.03, 0.04, 0.13]
    cam.distance, cam.azimuth, cam.elevation = 0.45, -70, -12

    opt = mujoco.MjvOption()
    opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
    opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = True

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    writer = imageio.get_writer(args.out, fps=args.fps, codec="libx264",
                                quality=9, macro_block_size=8)
    sel = range(0, len(T), args.stride)
    n = 0
    for k, fi in enumerate(sel):
        d.qpos[:] = Q[fi]
        mujoco.mj_forward(m, d)
        # gentle orbit so the 3-D structure reads clearly
        cam.azimuth = -70 + 14 * np.sin(2 * np.pi * k / max(1, len(T) // args.stride))
        renderer.update_scene(d, camera=cam, scene_option=opt)
        frame = renderer.render()
        if not args.no_hud:
            frame = draw_hud(frame, float(T[fi]), t_end)
        writer.append_data(frame)
        n += 1
    writer.close()
    print(f"wrote {args.out}  ({n} frames @ {args.fps} fps = {n / args.fps:.1f}s)")


if __name__ == "__main__":
    main()
