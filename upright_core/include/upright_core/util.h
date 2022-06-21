#pragma once

#include <Eigen/Eigen>

#include "upright_core/types.h"

namespace upright {

template <typename Scalar>
Scalar squared(Scalar x) {
    return x * x;
}

template <typename Scalar>
Scalar epsilon_norm(const Matrix<Scalar>& x, const Scalar eps) {
    Eigen::Map<const Vector<Scalar>> x_vec(x.data(), x.size(), 1);
    return sqrt(x_vec.dot(x_vec) + eps);

    // Unfortunately we cannot diff through this:
    // Eigen::JacobiSVD<Matrix<Scalar>> svd(x);
    // return svd.singularValues()(0);
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

// Generate a random scalar between 0 and 1
template <typename Scalar>
Scalar random_scalar() {
    Scalar x = Eigen::Matrix<Scalar, 1, 1>::Random()(0);
    return 0.5 * (x + 1.0);
}

// Test if a scalar is near zero. For a vector or matrix, use Eigen's
// Matrix::isZero method.
template <typename Scalar>
bool near_zero(Scalar x) {
    return abs(x) < Scalar(1e-6);
}

}  // namespace upright