#pragma once

#include <Eigen/Eigen>

#include "upright_core/types.h"

namespace upright {

// Number of constraints per contact. One constraint for the normal force
// to be non-negative; one for the friction cone.
const size_t NUM_CONSTRAINTS_PER_CONTACT = 5;

template <typename Scalar>
struct ContactPoint {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    // Names of the objects in contact
    std::string object1_name;
    std::string object2_name;

    // Coefficient of friction
    Scalar mu;

    // Position of the contact point in each object's local frame. The first
    // object is the one at whose base the forces are acting.
    // Used to compute the object dynamics.
    Vec3<Scalar> r_co_o1;
    Vec3<Scalar> r_co_o2;

    // Normal to the contact surface (points into the first object). Used to
    // compute the friction cone constraint.
    Vec3<Scalar> normal;

    // Cast to another underlying scalar type
    template <typename T>
    ContactPoint<T> cast() const {
        ContactPoint<T> point;
        point.object1_name = object1_name;
        point.object2_name = object2_name;
        point.mu = T(mu);
        point.r_co_o1 = r_co_o1.template cast<T>();
        point.r_co_o2 = r_co_o2.template cast<T>();
        point.normal = normal.template cast<T>();
        return point;
    }
};

}  // namespace upright
