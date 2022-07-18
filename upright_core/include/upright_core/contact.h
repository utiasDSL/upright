#pragma once

#include <Eigen/Eigen>

#include "upright_core/types.h"

namespace upright {

template <typename Scalar>
struct ContactPoint {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    // Names of the objects in contact
    std::string object1_name;
    std::string object2_name;

    // Position of the contact point in each object's local frame
    Vec3<Scalar> r_co_o1;
    Vec3<Scalar> r_co_o2;
};

}  // namespace upright
