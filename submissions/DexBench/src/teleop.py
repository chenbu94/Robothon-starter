"""Interactive keyboard teleoperation of the dexterous hand.

Opens the MuJoCo passive viewer and maps keys to high-level hand commands via
``TeleopController``. Requires a display (run locally, not on a headless box).

Controls
--------
    O : open hand            C : caging grasp (close)
    P : pinch (thumb+index)  F : full fist
    I : point (index out)    H : hold current pose
    R : reset to keyframe     SPACE : toggle perturbation pushes on the object
    ESC : quit

    python src/teleop.py
"""
from __future__ import annotations
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from controllers import TeleopController, CTRL_LO, CTRL_HI  # noqa: E402

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    import mujoco
    import mujoco.viewer
    m = mujoco.MjModel.from_xml_path(os.path.join(HERE, "assets", "scene.xml"))
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    teleop = TeleopController(m)
    state = {"cmd": "hold", "perturb": False}
    oid = m.body("object").id

    key_map = {ord("O"): "open", ord("C"): "close", ord("P"): "pinch",
               ord("F"): "fist", ord("I"): "point", ord("H"): "hold"}

    def on_key(keycode):
        if keycode in key_map:
            state["cmd"] = key_map[keycode]
            print("cmd ->", state["cmd"])
        elif keycode == ord("R"):
            mujoco.mj_resetDataKeyframe(m, d, 0)
        elif keycode == ord(" "):
            state["perturb"] = not state["perturb"]
            print("perturbation:", state["perturb"])

    print(__doc__)
    with mujoco.viewer.launch_passive(m, d, key_callback=on_key) as v:
        ctrl = np.array(d.ctrl)
        while v.is_running():
            ctrl = teleop(ctrl, state["cmd"])
            d.ctrl[:] = np.clip(ctrl, CTRL_LO, CTRL_HI)
            d.xfrc_applied[oid, :3] = 0.0
            if state["perturb"]:
                d.xfrc_applied[oid, :3] = 0.25 * np.random.randn(3)
            mujoco.mj_step(m, d)
            v.sync()


if __name__ == "__main__":
    main()
