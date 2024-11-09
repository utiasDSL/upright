import numpy as np
import rigeo as rg


def body_gravity3(C_ew, g=9.81):
    """Compute body acceleration vector."""
    return C_ew @ [0, 0, -g]


def body_gravity6(C_ew, g=9.81):
    """Compute body acceleration twist."""
    return np.concatenate((body_gravity3(C_ew, g), np.zeros(3)))


def cone_span_to_face_form(S):
    """Convert the span form of a polyhedral convex cone to face form.

    Span form is { Sz | z  >= 0 }
    Face form is { x  | Ax <= 0 }

    Return A of the face form.
    """
    ff = rg.SpanForm(rays=S.T).to_face_form()
    assert np.allclose(ff.b, 0)
    return ff.A
