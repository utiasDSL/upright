from functools import partial

import numpy as np
import jax.numpy as jnp
import jax

import util
from simulation import GRAVITY_VECTOR


@partial(jax.jit, static_argnums=(0,))
def object_balance_constraints(obj, C_we, ω_ew_w, a_ew_w, α_ew_w):
    C_ew = C_we.T
    Sω_ew_w = util.skew3(ω_ew_w)
    ddC_we = (util.skew3(α_ew_w) + Sω_ew_w @ Sω_ew_w) @ C_we

    α = obj.body.mass * C_ew @ (a_ew_w + ddC_we @ obj.body.com - GRAVITY_VECTOR)
    Iw = C_we @ obj.body.inertia @ C_we.T
    β = C_ew @ Sω_ew_w @ Iw @ ω_ew_w + obj.body.inertia @ C_ew @ α_ew_w

    h1 = (obj.mu * α[2]) ** 2 - α[0] ** 2 - α[1] ** 2 - (β[2] / obj.r_tau) ** 2
    h2 = α[2]  # α3 >= 0

    r_z = -obj.com_height
    S = np.array([[0, 1], [-1, 0]])
    zmp = (r_z * α[:2] - S @ β[:2]) / α[2]  # TODO numerical issues?
    # h3 = obj.support_area.zmp_constraints_scaled(zmp, α[2])
    h3 = obj.support_area.zmp_constraints(zmp)

    h12 = jnp.array([h1, h2])
    return jnp.concatenate((h12, h3))
