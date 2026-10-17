"""Low-dim human-anchored RL parameterization (shared by trainer + CI eval).

The correction is an affine map on top of the human-trained nets:

    aim:  out'_i = g_i * out_i + o_i          i in {x, y}
    tap:  logit' = s * logit + t

x = [gx, gy, ox, oy, s, t]; identity is [1, 1, 0, 0, 1, 0] (== the seed nets),
so any candidate stays close to the human policy.
"""
from __future__ import annotations

import numpy as np

NUM_PARAMS = 6
X0 = np.array([1.0, 1.0, 0.0, 0.0, 1.0, 0.0])
STDS = np.array([0.06, 0.06, 2.0, 2.0, 0.06, 0.4])


def x_to_patch(x, aim, tap) -> dict:
    gx, gy, ox, oy, s, t = [float(v) for v in x]
    aw = aim.net[4].weight.detach()
    ab = aim.net[4].bias.detach()
    tw = tap.net[4].weight.detach()
    tb = tap.net[4].bias.detach()
    aim_w = aw.clone()
    aim_b = ab.clone()
    aim_w[0] = gx * aw[0]
    aim_w[1] = gy * aw[1]
    aim_b[0] = gx * ab[0] + ox
    aim_b[1] = gy * ab[1] + oy
    return {
        "aim_patch": {"net.4.weight": aim_w.tolist(), "net.4.bias": aim_b.tolist()},
        "tap_patch": {"net.4.weight": (s * tw).tolist(), "net.4.bias": (s * tb + t).tolist()},
    }
