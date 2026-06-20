"""Generate analysis figures from the recorded episode datasets.

    python src/analyze.py        # -> media/metrics.png
"""
from __future__ import annotations
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(HERE, "data")


def main():
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))

    # 1) reach: mean tracking error proxy via reward (=-err)
    r = np.load(os.path.join(DATA, "reach.npz"))
    err_mm = -r["reward"] * 1000.0
    ax[0].plot(r["time"], err_mm, color="#1f77b4")
    ax[0].axhline(err_mm[len(err_mm)//3:].mean(), ls="--", color="k",
                  label=f"mean {err_mm[len(err_mm)//3:].mean():.1f} mm")
    ax[0].set(title="Task 1 - Fingertip IK tracking error",
              xlabel="t (s)", ylabel="Cartesian error (mm)")
    ax[0].legend(); ax[0].grid(alpha=.3)

    # 2) grasp: object height + touch sum
    g = np.load(os.path.join(DATA, "grasp.npz"))
    ax[1].plot(g["time"], g["object_pose"][:, 2] * 100, color="#d62728",
               label="object height (cm)")
    ax[1].axvspan(2.5, 5.5, color="orange", alpha=.15, label="perturbation")
    ax[1].set(title="Task 2 - Grasp stability under perturbation",
              xlabel="t (s)", ylabel="height (cm)")
    ax[1].legend(); ax[1].grid(alpha=.3)

    # 3) choreography: per-finger fingertip height (independent control)
    c = np.load(os.path.join(DATA, "choreography.npz"))
    tips = c["tip_xyz"]  # (T,5,3)
    names = ["index", "middle", "ring", "pinky", "thumb"]
    for k, nm in enumerate(names):
        ax[2].plot(c["time"], tips[:, k, 2] * 100, label=nm)
    ax[2].set(title="Task 3 - Independent fingertip motion",
              xlabel="t (s)", ylabel="fingertip height (cm)")
    ax[2].legend(fontsize=8); ax[2].grid(alpha=.3)

    fig.tight_layout()
    out = os.path.join(HERE, "media", "metrics.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=120)
    print("wrote", out)


if __name__ == "__main__":
    main()
