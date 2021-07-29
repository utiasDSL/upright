#pragma once

#include <ocs2_mobile_manipulator_modified/definitions.h>
#include <ocs2_pinocchio_interface/PinocchioEndEffectorKinematicsCppAd.h>
#include <ocs2_robotic_tools/common/SkewSymmetricMatrix.h>
#include <ocs2_robotic_tools/end_effector/EndEffectorKinematics.h>

#include <ocs2_core/constraint/StateInputConstraintCppAd.h>
#include <ocs2_oc/synchronized_module/ReferenceManager.h>

namespace ocs2 {
namespace mobile_manipulator {

template <typename Scalar>
Eigen::Matrix<Scalar, 3, 3> cylinder_inertia_matrix(Scalar mass, Scalar radius,
                                                    Scalar height) {
    // diagonal elements
    Scalar xx =
        mass * (Scalar(3.0) * radius * radius + height * height) / Scalar(12.0);
    Scalar yy = xx;
    Scalar zz = Scalar(0.5) * mass * radius * radius;

    // construct the inertia matrix
    Eigen::Matrix<Scalar, 3, 3> I = Eigen::Matrix<Scalar, 3, 3>::Zero();
    I.diagonal() << xx, yy, zz;
    return I;
}

template <typename Scalar>
Scalar equilateral_triangle_inscribed_radius(Scalar side_length) {
    return side_length / Scalar(2 * std::sqrt(3));
}

}  // namespace mobile_manipulator
}  // namespace ocs2
