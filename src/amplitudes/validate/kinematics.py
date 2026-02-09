from __future__ import annotations
import numpy as np
import math

def cm_2to2_massless(sqrts: float, cos_theta: float):
    """Return momenta for 2->2 massless scattering in CM.

    Returns p_in1, p_in2, p_out1, p_out2 as (E,px,py,pz).
    Incoming are along +z and -z. Outgoing in x-z plane.
    """
    E = sqrts/2.0
    p = E
    st = math.sqrt(max(0.0, 1.0 - cos_theta*cos_theta))
    p1 = np.array([E, 0.0, 0.0, +p])
    p2 = np.array([E, 0.0, 0.0, -p])
    p3 = np.array([E, p*st, 0.0, p*cos_theta])
    p4 = np.array([E, -p*st, 0.0, -p*cos_theta])
    return p1,p2,p3,p4
