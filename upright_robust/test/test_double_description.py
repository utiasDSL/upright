import numpy as np
import IPython

import upright_robust as rob


def test_single_contact():
    μ = 0.5
    S = np.array([[μ, 0, 1], [0, μ, 1], [-μ, 0, 1], [0, -μ, 1]]).T
    F1 = rob.span_to_face_form(S, library="cdd")
    F2 = rob.span_to_face_form(S, library="qhull")

    assert F1.shape == F2.shape

    # for each row of F1, check that there is one and only one row of F2 that
    # is a scalar multiple
    for i in range(F1.shape[0]):
        f = F1[i, :]
        X = F2 / f
        assert np.sum(np.isclose(X.ptp(axis=1), 0)) == 1
