#include <tray_balance_ocs2/types.h>

#include "tray_balance_ocs2/constraint/BoundedBalancingConstraints.h"

namespace ocs2 {
namespace mobile_manipulator {

BoundedTrayBalanceConstraints::BoundedTrayBalanceConstraints(
    const PinocchioEndEffectorKinematicsCppAd& pinocchioEEKinematics,
    const TrayBalanceSettings& settings, const Vec3d& gravity,
    const RobotDimensions& dims, bool recompileLibraries)
    : StateInputConstraintCppAd(ConstraintOrder::Linear),
      pinocchioEEKinPtr_(pinocchioEEKinematics.clone()),
      gravity_(gravity),
      settings_(settings),
      dims_(dims) {
    if (pinocchioEEKinematics.getIds().size() != 1) {
        throw std::runtime_error(
            "[TrayBalanaceConstraint] endEffectorKinematics has wrong "
            "number of end effector IDs.");
    }

    // compile the CppAD library
    initialize(dims.x, dims.u, 0, "bounded_tray_balance_constraints",
               "/tmp/ocs2", recompileLibraries, true);

    num_constraints_ = num_balancing_constraints(settings_.objects);
}

ad_vector_t BoundedTrayBalanceConstraints::constraintFunction(
    ad_scalar_t time, const VecXad& state, const VecXad& input,
    const VecXad& parameters) const {
    Mat3ad C_we = pinocchioEEKinPtr_->getOrientationCppAd(state);
    Vec3ad angular_vel =
        pinocchioEEKinPtr_->getAngularVelocityCppAd(state, input);
    Vec3ad angular_acc =
        pinocchioEEKinPtr_->getAngularAccelerationCppAd(state, input);
    Vec3ad linear_acc = pinocchioEEKinPtr_->getAccelerationCppAd(state, input);

    // Cast to AD scalar type
    Vec3ad ad_gravity = gravity_.template cast<ad_scalar_t>();
    std::vector<BoundedBalancedObject<ad_scalar_t>> ad_objects;
    for (const auto& obj : settings_.objects) {
        ad_objects.push_back(obj.cast<ad_scalar_t>());
    }

    return balancing_constraints(ad_objects, ad_gravity,
                                 settings_.constraints_enabled, C_we,
                                 angular_vel, linear_acc, angular_acc);
}

std::ostream& operator<<(std::ostream& out,
                         const TrayBalanceSettings& settings) {
    out << "enabled = " << settings.enabled << std::endl
        << "normal enabled = " << settings.constraints_enabled.normal
        << std::endl
        << "friction enabled = " << settings.constraints_enabled.friction
        << std::endl
        << "ZMP enabled = " << settings.constraints_enabled.zmp << std::endl
        << "num objects = " << settings.objects.size() << std::endl
        << "mu = " << settings.mu << std::endl
        << "delta = " << settings.delta << std::endl;
    return out;
}

}  // namespace mobile_manipulator
}  // namespace ocs2
