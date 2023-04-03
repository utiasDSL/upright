#pragma once

#include <Eigen/Eigen>
#include <iostream>

#include "upright_core/types.h"

namespace upright {

template <typename Scalar>
Scalar squared(Scalar x) {
    return x * x;
}

template <typename Scalar>
Scalar epsilon_norm(const MatX<Scalar>& x, const Scalar eps) {
    Eigen::Map<const VecX<Scalar>> x_vec(x.data(), x.size(), 1);
    return sqrt(x_vec.dot(x_vec) + eps);

    // Unfortunately we cannot diff through this:
    // Eigen::JacobiSVD<MatX<Scalar>> svd(x);
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

// Second time-derivative of the rotation matrix.
template <typename Scalar>
Mat3<Scalar> dC_dtt(const Mat3<Scalar>& C_we, const Vec3<Scalar>& angular_vel,
                    const Vec3<Scalar>& angular_acc) {
    Mat3<Scalar> S_angular_vel = skew3<Scalar>(angular_vel);
    Mat3<Scalar> S_angular_acc = skew3<Scalar>(angular_acc);
    return (S_angular_acc + S_angular_vel * S_angular_vel) * C_we;
}

template <typename Scalar>
Mat3<Scalar> dC_dtt(const RigidBodyState<Scalar>& state) {
    return dC_dtt(state.pose.orientation, state.velocity.angular,
                  state.acceleration.angular);
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
    return abs(x) < Scalar(NEAR_ZERO);
}

// Compute the a basis for the nullspace of the vector v.
template <typename Scalar>
Eigen::Matrix<Scalar, 2, 3> null(const Vec3<Scalar>& v) {
    Eigen::FullPivLU<MatX<Scalar>> lu(v.transpose());
    MatX<Scalar> kernel = lu.kernel();

    if (kernel.rows() != 3 || kernel.cols() != 2) {
        std::cout << "kernel.shape = (" << kernel.rows() << ", " << kernel.cols() << ")" << std::endl;
        throw std::runtime_error("Kernel of vector is of wrong size!");
    }

    Eigen::Matrix<Scalar, 2, 3> S;
    for (size_t i = 0; i < 2; ++i) {
        S.row(i) = kernel.col(i).normalized().transpose();
    }
    return S;
}

}  // namespace upright
