from scipy.spatial import ConvexHull
import numpy as np
import IPython

import upright_robust as rob

# TODO
def test_single_contact():
    μ = 0.5
    # F = np.array([[1, 0, -μ], [0, 1, -μ], [-1, 0, -μ], [0, -1, -μ]])

    S = np.array([[μ, 0, 1], [0, μ, 1], [-μ, 0, 1], [0, -μ, 1]]).T
    # points = np.vstack((S.T, [[0, 0, 0]]))
    F1 = rob.span_to_face_form(S)

    # hull = ConvexHull(points)
    # F2 = hull.equations

    # IPython.embed()
