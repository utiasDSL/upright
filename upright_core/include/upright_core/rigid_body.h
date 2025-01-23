#pragma once

#include <Eigen/Eigen>

#include "upright_core/types.h"
#include "upright_core/util.h"

namespace upright {

template <typename Scalar>
VecX<Scalar> vech(const Mat3<Scalar>& I) {
    // Half-vectorization of the inertia matrix (i.e., the upper-triangular part
    // is vectorized).
    VecX<Scalar> v(6);
    v << I(0, 0), I(0, 1), I(0, 2), I(1, 1), I(1, 2), I(2, 2);
    return v;
}

template <typename Scalar>
Mat3<Scalar> unvech(const VecX<Scalar>& v) {
    // Recover the inertia matrix from its half-vectorization.
    Mat3<Scalar> I;
    I << v(0), v(1), v(2), v(1), v(3), v(4), v(2), v(4), v(5);
    return I;
}

template <typename Scalar>
struct RigidBody {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    RigidBody(const Scalar mass, const Mat3<Scalar>& inertia,
              const Vec3<Scalar>& com)
        : mass(mass), inertia(inertia), com(com) {}

    // Create a RigidBody from a parameter vector
    static RigidBody<Scalar> from_parameters(const VecX<Scalar>& parameters,
                                             const size_t index = 0) {
        Scalar mass(parameters(index));
        Vec3<Scalar> com(parameters.template segment<3>(index + 1) / mass);
        VecX<Scalar> I_vec(parameters.template segment<6>(index + 4));
        Mat3<Scalar> inertia = unvech(I_vec);
        return RigidBody(mass, inertia, com);
    }

    static size_t num_parameters() { return 10; }

    VecX<Scalar> get_parameters() const {
        VecX<Scalar> p(num_parameters());
        p << mass, mass * com, vech(inertia);
        return p;
    }

    // Cast to a different underlying scalar type.
    template <typename T>
    RigidBody<T> cast() const {
        VecX<Scalar> parameters = get_parameters();
        VecX<T> parametersT = parameters.template cast<T>();
        return RigidBody<T>::from_parameters(parametersT);
    }

    Scalar mass;
    Mat3<Scalar> inertia;
    Vec3<Scalar> com;
};

}  // namespace upright
