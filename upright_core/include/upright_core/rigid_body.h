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

    // Compose multiple rigid bodies into one.
    // TODO is this actually used?
    static RigidBody<Scalar> compose(
        const std::vector<RigidBody<Scalar>>& bodies) {
        // Compute new mass and center of mass.
        Scalar mass = Scalar(0);
        Vec3<Scalar> com = Vec3<Scalar>::Zero();
        for (int i = 0; i < bodies.size(); ++i) {
            mass += bodies[i].mass;
            com += bodies[i].mass * bodies[i].com;
        }
        com = com / mass;

        // Parallel axis theorem to compute new moment of inertia.
        Mat3<Scalar> inertia = Mat3<Scalar>::Zero();
        for (int i = 0; i < bodies.size(); ++i) {
            Vec3<Scalar> r = bodies[i].com - com;
            Mat3<Scalar> R = skew3(r);
            inertia += bodies[i].inertia - bodies[i].mass * R * R;
        }

        return RigidBody<Scalar>(mass, inertia, com);
    }

    // Create a RigidBody from a parameter vector
    static RigidBody<Scalar> from_parameters(const VecX<Scalar>& parameters,
                                             const size_t index = 0) {
        Scalar mass(parameters(index));
        Vec3<Scalar> com(parameters.template segment<3>(index + 1));
        VecX<Scalar> I_vec(parameters.template segment<6>(index + 4));
        Mat3<Scalar> inertia = unvech(I_vec);
        return RigidBody(mass, inertia, com);
    }

    size_t num_parameters() const { return 10; }

    VecX<Scalar> get_parameters() const {
        // TODO probably should make this mass * com
        VecX<Scalar> p(num_parameters());
        p << mass, com, vech(inertia);
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
