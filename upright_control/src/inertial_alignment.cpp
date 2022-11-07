#include <upright_control/types.h>

#include "upright_control/inertial_alignment.h"

namespace upright {

VecXad InertialAlignmentConstraint::constraintFunction(
    ocs2::ad_scalar_t time, const VecXad& state, const VecXad& input,
    const VecXad& parameters) const {
    Mat3ad C_we = kinematics_ptr_->getOrientationCppAd(state);
    Vec3ad linear_acc = kinematics_ptr_->getAccelerationCppAd(state, input);
    Vec3ad gravity = gravity_.cast<ocs2::ad_scalar_t>();
    Vec3ad n = settings_.contact_plane_normal.cast<ocs2::ad_scalar_t>();
    Mat23ad S = settings_.contact_plane_span.cast<ocs2::ad_scalar_t>();

    // In the EE frame
    Vec3ad a = C_we.transpose() * (linear_acc - gravity);

    // Take into account object center of mass, if available
    if (settings_.use_angular_acceleration) {
        Vec3ad angular_vel =
            kinematics_ptr_->getAngularVelocityCppAd(state, input);
        Vec3ad angular_acc =
            kinematics_ptr_->getAngularAccelerationCppAd(state, input);

        Mat3ad ddC_we =
            dC_dtt<ocs2::ad_scalar_t>(C_we, angular_vel, angular_acc);
        Vec3ad com = settings_.com.cast<ocs2::ad_scalar_t>();

        a += ddC_we * com;
    } else if (settings_.align_with_fixed_vector) {
        // In this case we just try to stay aligned with whatever the
        // original normal was.
        a = C_we.transpose() * n;
    }

    ocs2::ad_scalar_t a_n = n.dot(a);
    Vec2ad a_t = S * a;

    // constrain the normal acc to be non-negative
    VecXad constraints(getNumConstraints(0));
    constraints(0) = a_n;

    // linearized version: the quadratic cone does not play well with the
    // optimizer
    constraints(1) = settings_.alpha * a_n - a_t(0) - a_t(1);
    constraints(2) = settings_.alpha * a_n - a_t(0) + a_t(1);
    constraints(3) = settings_.alpha * a_n + a_t(0) - a_t(1);
    constraints(4) = settings_.alpha * a_n + a_t(0) + a_t(1);
    return constraints;
}

ocs2::ad_scalar_t InertialAlignmentCost::costFunction(
    ocs2::ad_scalar_t time, const VecXad& state, const VecXad& input,
    const VecXad& parameters) const {
    Mat3ad C_we = kinematics_ptr_->getOrientationCppAd(state);
    Vec3ad linear_acc = kinematics_ptr_->getAccelerationCppAd(state, input);

    Vec3ad gravity = gravity_.cast<ocs2::ad_scalar_t>();
    Vec3ad total_acc = linear_acc - gravity;

    if (settings_.use_angular_acceleration) {
        Vec3ad angular_vel =
            kinematics_ptr_->getAngularVelocityCppAd(state, input);
        Vec3ad angular_acc =
            kinematics_ptr_->getAngularAccelerationCppAd(state, input);

        Mat3ad ddC_we =
            dC_dtt<ocs2::ad_scalar_t>(C_we, angular_vel, angular_acc);
        Vec3ad com = settings_.com.cast<ocs2::ad_scalar_t>();

        total_acc += ddC_we * com;
    }

    // Vec3ad n = C_we * settings_.contact_plane_normal.cast<ocs2::ad_scalar_t>();

    // if (settings_.align_with_fixed_vector) {
    //     // Negative because we want to maximize
    //     return -settings_.cost_weight * n.dot(C_we.transpose() * n);
    // } else {
    Vec3ad n = settings_.contact_plane_normal.cast<ocs2::ad_scalar_t>();
    Vec3ad a = C_we.transpose() * total_acc / total_acc.norm();
    ocs2::ad_scalar_t angle = acos(n.dot(a));
    // return -settings_.cost_weight * n.dot(a);
    return 0.5 * settings_.cost_weight * angle * angle;

    // Vec3ad e = n.cross(C_we.transpose() * total_acc) / gravity.norm();
    // return 0.5 * settings_.cost_weight * e.dot(e);

    // Vec3ad e = n.cross(total_acc).cross(n) / gravity.norm();
    // return 0.5 * settings_.cost_weight * e.dot(e);

    // Mat23ad S = settings_.contact_plane_span.cast<ocs2::ad_scalar_t>();
    // Vec2ad e = S * (C_we.transpose() * total_acc) / gravity.norm();
    // return 0.5 * settings_.cost_weight * e(0) * e(0);

    // Mat23ad S = settings_.contact_plane_span.cast<ocs2::ad_scalar_t>();
    // Vec2ad e = S * (C_we.transpose() * total_acc) / gravity.norm();
    // return 0.5 * settings_.cost_weight * e.dot(e);
    // }
}

}  // namespace upright
