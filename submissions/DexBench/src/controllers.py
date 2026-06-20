"""Controllers for the DexBench dexterous-hand suite.

All controllers act on the 20 position actuators of the hand
(4 fingers x [abd, mcp, pip, dip] + thumb [cmc_abd, cmc_flex, mcp, ip]).

Exposed controllers
-------------------
* ``GraspController``      -- closes into a firm caging grasp on the object.
* ``FingertipIK``          -- per-finger differential (Jacobian) inverse
                              kinematics: drive a fingertip to a 3-D Cartesian
                              target. The basis of the reaching / tracking task.
* ``Choreography``         -- scripted multi-finger synergy sequence (wave,
                              counting, pinch) used as a dexterity showcase.
* ``TeleopController``     -- maps discrete keyboard commands to actuator goals.

Design notes
------------
The fingers naturally curl up-and-back toward the palm, so reachable Cartesian
targets live in that swept arc. ``FingertipIK`` solves one Gauss-Newton step per
control tick using ``mj_jacSite`` and a damped pseudo-inverse, which keeps the
controller stable near kinematic singularities (fully extended / flexed).
"""
from __future__ import annotations
import numpy as np

FINGERS = ["index", "middle", "ring", "pinky"]
FINGER_DOF = {f: [f + s for s in ("_abd", "_mcp", "_pip", "_dip")] for f in FINGERS}
FINGER_DOF["thumb"] = ["thumb_cmc_abd", "thumb_cmc_flex", "thumb_mcp", "thumb_ip"]
ALL_FINGERS = FINGERS + ["thumb"]

CTRL_LO, CTRL_HI = -0.35, 1.7


class _NameMaps:
    """Resolve actuator / joint / site indices once per model."""

    def __init__(self, model):
        self.m = model
        self.act = {model.actuator(i).name.replace("act_", ""): i
                    for i in range(model.nu)}
        self.qadr = {model.joint(i).name: model.joint(i).qposadr[0]
                     for i in range(model.njnt)}
        self.dofadr = {model.joint(i).name: model.joint(i).dofadr[0]
                       for i in range(model.njnt)}

    def finger_act(self, f):
        return [self.act[j] for j in FINGER_DOF[f]]

    def finger_qadr(self, f):
        return [self.qadr[j] for j in FINGER_DOF[f]]

    def finger_dofadr(self, f):
        return [self.dofadr[j] for j in FINGER_DOF[f]]


class GraspController(_NameMaps):
    """A firm caging grasp: fingers wrap, thumb opposes."""

    def target(self) -> np.ndarray:
        t = np.zeros(self.m.nu)
        for f, abd in (("index", -0.05), ("middle", 0.0),
                       ("ring", 0.05), ("pinky", 0.12)):
            t[self.act[f + "_abd"]] = abd
            t[self.act[f + "_mcp"]] = 0.95
            t[self.act[f + "_pip"]] = 1.15
            t[self.act[f + "_dip"]] = 0.85
        t[self.act["thumb_cmc_abd"]] = 1.05
        t[self.act["thumb_cmc_flex"]] = 0.65
        t[self.act["thumb_mcp"]] = 0.85
        t[self.act["thumb_ip"]] = 0.75
        return t

    def __call__(self, ctrl, t, alpha=0.05):
        return ctrl + alpha * (self.target() - ctrl)


class FingertipIK(_NameMaps):
    """Differential IK: move ``finger`` fingertip toward a Cartesian target."""

    def __init__(self, model, data, damping: float = 1e-3):
        super().__init__(model)
        self.d = data
        self.damping = damping
        self._jacp = np.zeros((3, model.nv))

    def site_pos(self, f):
        return self.d.site_xpos[self.m.site(f + "_tip").id].copy()

    def step_targets(self, f, target_xyz) -> np.ndarray:
        """Return new joint position targets for finger ``f`` (length 4)."""
        sid = self.m.site(f + "_tip").id
        import mujoco
        mujoco.mj_jacSite(self.m, self.d, self._jacp, None, sid)
        J = self._jacp[:, self.finger_dofadr(f)]            # 3 x 4
        err = np.asarray(target_xyz) - self.d.site_xpos[sid]
        dq = np.linalg.pinv(J, rcond=self.damping) @ err
        q = self.d.qpos[self.finger_qadr(f)] + dq
        return np.clip(q, CTRL_LO, CTRL_HI)

    def apply(self, f, target_xyz):
        q = self.step_targets(f, target_xyz)
        for k, a in enumerate(self.finger_act(f)):
            self.d.ctrl[a] = q[k]
        return q


class Choreography(_NameMaps):
    """Scripted dexterity showcase: sequential finger flexion ('counting')."""

    def __call__(self, ctrl, t, alpha=0.12):
        target = np.zeros(self.m.nu)
        # each finger flexes in its own time window, then a final full fist
        order = ["index", "middle", "ring", "pinky", "thumb"]
        for k, f in enumerate(order):
            on = 0.5 * (1 + np.tanh(3 * (t - (0.6 + 0.5 * k))))
            for s, amp in zip(("_mcp", "_pip", "_dip"), (1.0, 1.2, 0.9)):
                key = f + s if f != "thumb" else None
            # explicit per-finger flexion
            if f == "thumb":
                target[self.act["thumb_cmc_flex"]] = 0.7 * on
                target[self.act["thumb_mcp"]] = 0.9 * on
                target[self.act["thumb_ip"]] = 0.8 * on
            else:
                target[self.act[f + "_mcp"]] = 1.0 * on
                target[self.act[f + "_pip"]] = 1.2 * on
                target[self.act[f + "_dip"]] = 0.9 * on
        return ctrl + alpha * (target - ctrl)


class TeleopController(_NameMaps):
    """Discrete keyboard command -> actuator goal (smoothed)."""

    CMDS = ("open", "close", "pinch", "point", "fist", "hold")

    def __init__(self, model):
        super().__init__(model)
        self._grasp = GraspController(model)

    def goal(self, cmd: str) -> np.ndarray:
        t = np.zeros(self.m.nu)
        if cmd == "open":
            return t
        if cmd in ("close", "fist"):
            return self._grasp.target() if cmd == "close" else np.clip(
                self._grasp.target() + 0.3, CTRL_LO, CTRL_HI)
        if cmd == "pinch":
            t[self.act["index_mcp"]] = 0.8; t[self.act["index_pip"]] = 1.0
            t[self.act["index_dip"]] = 0.7
            t[self.act["thumb_cmc_abd"]] = 1.1; t[self.act["thumb_cmc_flex"]] = 0.7
            t[self.act["thumb_mcp"]] = 0.9; t[self.act["thumb_ip"]] = 0.8
            return t
        if cmd == "point":
            g = self._grasp.target(); g[self.act["index_mcp"]] = 0.0
            g[self.act["index_pip"]] = 0.0; g[self.act["index_dip"]] = 0.0
            return g
        return None  # 'hold' -> keep current ctrl

    def __call__(self, ctrl, cmd: str, alpha=0.15):
        g = self.goal(cmd)
        if g is None:
            return ctrl
        return ctrl + alpha * (g - ctrl)
