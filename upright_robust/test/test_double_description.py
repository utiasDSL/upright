import numpy as np
from scipy.spatial import ConvexHull

import upright_robust as rob


def _cone_span_to_face_form_qhull(S):
    points = np.vstack((np.zeros((1, S.shape[0])), S.T))
    hull = ConvexHull(points)
    mask = np.any(hull.simplices == 0, axis=1)
    F = hull.equations[mask, :]
    G = [F[0, :]]
    for i in range(1, F.shape[0]):
        if np.allclose(F[i, :], G[-1]):
            continue
        G.append(F[i, :])
    G = np.array(G)

    A = G[:, :-1]
    b = G[:, -1]
    return A, b


def test_single_contact():
    μ = 0.5
    S = np.array([[μ, 0, 1], [0, μ, 1], [-μ, 0, 1], [0, -μ, 1]]).T

    # compare to qhull-based implementation
    F1 = rob.cone_span_to_face_form(S)
    F2, b = _cone_span_to_face_form_qhull(S)
    assert np.allclose(b, 0)

    assert F1.shape == F2.shape

    # for each row of F1, check that there is one and only one row of F2 that
    # is a scalar multiple
    for i in range(F1.shape[0]):
        f = F1[i, :]
        X = F2 / f
        assert np.sum(np.isclose(X.ptp(axis=1), 0)) == 1
