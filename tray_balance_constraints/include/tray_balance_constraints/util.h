#pragma once

#include <Eigen/Eigen>

#include "tray_balance_constraints/types.h"

template <typename Scalar>
Scalar squared(Scalar x) {
    return x * x;
}

// Compute skew-symmetric matrix from 3-dimensional vector.
template <typename Scalar>
Mat3<Scalar> skew3(const Vec3<Scalar>& x) {
    Mat3<Scalar> M;
    // clang-format off
    M << Scalar(0),     -x(2),      x(1),
         x(2),      Scalar(0),     -x(0),
        -x(1),           x(0), Scalar(0);
    // clang-format on
    return M;
}
