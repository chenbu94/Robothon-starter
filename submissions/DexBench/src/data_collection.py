"""Episode data logger for imitation-learning / analysis datasets.

Records, per simulation step, a flat record of the full observation-action
pair plus task metadata. At the end of an episode it writes:

* ``<name>.npz``  -- arrays (time, qpos, qvel, ctrl, touch, object_pose, tip_xyz)
* ``<name>.csv``  -- a human-readable per-step table (a subset of columns)
* ``<name>.json`` -- episode metadata (task, success, lengths, rubric notes)

The schema is intentionally generic so the same logs can feed behaviour
cloning, offline RL, or simple plotting.
"""
from __future__ import annotations
import json
import os
import numpy as np

TIP_SITES = ["index_tip", "middle_tip", "ring_tip", "pinky_tip", "thumb_tip"]
TOUCH = ["touch_index", "touch_middle", "touch_ring", "touch_pinky", "touch_thumb"]


class EpisodeLogger:
    def __init__(self, model, data, task: str):
        self.m, self.d, self.task = model, data, task
        self._t, self._qpos, self._qvel, self._ctrl = [], [], [], []
        self._touch, self._objpose, self._tips, self._reward = [], [], [], []
        self._tip_ids = [model.site(s).id for s in TIP_SITES]
        self._touch_ids = [model.sensor(s).id for s in TOUCH]
        self._touch_adr = [model.sensor_adr[i] for i in self._touch_ids]
        self._objpos_adr = model.sensor_adr[model.sensor("object_pos").id]
        self._objquat_adr = model.sensor_adr[model.sensor("object_quat").id]

    def record(self, t: float, reward: float = 0.0):
        d = self.d
        self._t.append(t)
        self._qpos.append(d.qpos.copy())
        self._qvel.append(d.qvel.copy())
        self._ctrl.append(d.ctrl.copy())
        self._touch.append(np.array([d.sensordata[a] for a in self._touch_adr]))
        op = d.sensordata[self._objpos_adr:self._objpos_adr + 3]
        oq = d.sensordata[self._objquat_adr:self._objquat_adr + 4]
        self._objpose.append(np.concatenate([op, oq]).copy())
        self._tips.append(np.array([d.site_xpos[i] for i in self._tip_ids]).copy())
        self._reward.append(reward)

    def save(self, out_dir: str, name: str, meta: dict | None = None):
        os.makedirs(out_dir, exist_ok=True)
        arrs = dict(
            time=np.array(self._t),
            qpos=np.array(self._qpos),
            qvel=np.array(self._qvel),
            ctrl=np.array(self._ctrl),
            touch=np.array(self._touch),
            object_pose=np.array(self._objpose),
            tip_xyz=np.array(self._tips),
            reward=np.array(self._reward),
        )
        np.savez_compressed(os.path.join(out_dir, name + ".npz"), **arrs)

        # CSV: time + object pose + per-finger touch (compact, viewer-friendly)
        import csv
        with open(os.path.join(out_dir, name + ".csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["t", "obj_x", "obj_y", "obj_z",
                        "obj_qw", "obj_qx", "obj_qy", "obj_qz"] + TOUCH)
            for i in range(len(self._t)):
                w.writerow([f"{self._t[i]:.4f}"]
                           + [f"{v:.5f}" for v in self._objpose[i]]
                           + [f"{v:.4f}" for v in self._touch[i]])

        info = dict(task=self.task, steps=len(self._t),
                    duration_s=float(self._t[-1]) if self._t else 0.0,
                    dof=int(self.m.nv), actuators=int(self.m.nu))
        if meta:
            info.update(meta)
        with open(os.path.join(out_dir, name + ".json"), "w") as f:
            json.dump(info, f, indent=2)
        return arrs, info
