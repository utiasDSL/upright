#pragma once

#include <Eigen/Eigen>

#include "upright_core/types.h"
#include "upright_core/util.h"

namespace upright {

template <typename Scalar>
Scalar circle_r_tau(Scalar radius);

template <typename Scalar>
Scalar rectangle_r_tau(Scalar w, Scalar h);

template <typename Scalar>
Mat3<Scalar> cylinder_inertia_matrix(Scalar mass, Scalar radius, Scalar height);

template <typename Scalar>
Mat3<Scalar> cuboid_inertia_matrix(Scalar mass,
                                   const Vec3<Scalar>& side_lengths);

template <typename Scalar>
struct RigidBody {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    RigidBody(const Scalar mass, const Mat3<Scalar>& inertia,
              const Vec3<Scalar>& com)
        : mass(mass), inertia(inertia), com(com) {}

    // Compose multiple rigid bodies into one.
    static RigidBody<Scalar> compose(
        const std::vector<RigidBody<Scalar>>& bodies);

    // Create a RigidBody from a parameter vector
    static RigidBody<Scalar> from_parameters(const VecX<Scalar>& parameters,
                                             const size_t index = 0);

    size_t num_parameters() const { return 1 + 3 + 9; }

    VecX<Scalar> get_parameters() const;

    Scalar mass;
    Mat3<Scalar> inertia;
    Vec3<Scalar> com;
};

}  // namespace upright

#include "impl/dynamics.tpp"
