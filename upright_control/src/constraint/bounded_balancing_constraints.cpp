#include <upright_control/types.h>
#include <upright_core/nominal.h>
#include <upright_core/bounded.h>
#include <upright_core/bounded_constraints.h>
#include <upright_core/contact.h>
#include <upright_core/contact_constraints.h>

#include "upright_control/constraint/bounded_balancing_constraints.h"

namespace upright {

std::ostream& operator<<(std::ostream& out, const BalancingSettings& settings) {
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

RigidBodyState<ocs2::ad_scalar_t> get_rigid_body_state(
    const std::unique_ptr<ocs2::PinocchioEndEffectorKinematicsCppAd>&
        kinematics_ptr,
    const VecXad& state, const VecXad& input) {
    RigidBodyState<ocs2::ad_scalar_t> X;
    X.pose.position = kinematics_ptr->getPositionCppAd(state);
    X.pose.orientation = kinematics_ptr->getOrientationCppAd(state);

    X.velocity.linear = kinematics_ptr->getVelocityCppAd(state, input);
    X.velocity.angular = kinematics_ptr->getAngularVelocityCppAd(state, input);

    X.acceleration.linear = kinematics_ptr->getAccelerationCppAd(state, input);
    X.acceleration.angular =
        kinematics_ptr->getAngularAccelerationCppAd(state, input);
    return X;
}

// BoundedBalancingConstraints::BoundedBalancingConstraints(
//     const ocs2::PinocchioEndEffectorKinematicsCppAd& pinocchioEEKinematics,
//     const BalancingSettings& settings, const Vec3d& gravity,
//     const OptimizationDimensions& dims, bool recompileLibraries)
//     : ocs2::StateInputConstraintCppAd(ocs2::ConstraintOrder::Linear),
//       pinocchioEEKinPtr_(pinocchioEEKinematics.clone()),
//       gravity_(gravity),
//       settings_(settings),
//       dims_(dims) {
//     if (pinocchioEEKinematics.getIds().size() != 1) {
//         throw std::runtime_error(
//             "[TrayBalanaceConstraint] endEffectorKinematics has wrong "
//             "number of end effector IDs.");
//     }
//
//     // compile the CppAD library
//     initialize(dims.x(), dims.u(), 0, "upright_bounded_balancing_constraints",
//                "/tmp/ocs2", recompileLibraries, true);
//
//     num_constraints_ = num_balancing_constraints(settings_.objects);
// }
//
// VecXad BoundedBalancingConstraints::constraintFunction(
//     ocs2::ad_scalar_t time, const VecXad& state, const VecXad& input,
//     const VecXad& parameters) const {
//     Mat3ad C_we = pinocchioEEKinPtr_->getOrientationCppAd(state);
//     Vec3ad angular_vel =
//         pinocchioEEKinPtr_->getAngularVelocityCppAd(state, input);
//     Vec3ad angular_acc =
//         pinocchioEEKinPtr_->getAngularAccelerationCppAd(state, input);
//     Vec3ad linear_acc = pinocchioEEKinPtr_->getAccelerationCppAd(state, input);
//
//     // Cast to AD scalar type
//     Vec3ad ad_gravity = gravity_.template cast<ocs2::ad_scalar_t>();
//     std::vector<BoundedBalancedObject<ocs2::ad_scalar_t>> ad_objects;
//     for (const auto& kv : settings_.objects) {
//         ad_objects.push_back(kv.second.cast<ocs2::ad_scalar_t>());
//     }
//
//     return balancing_constraints(ad_objects, ad_gravity,
//                                  settings_.constraints_enabled, C_we,
//                                  angular_vel, linear_acc, angular_acc);
// }

NominalBalancingConstraints::NominalBalancingConstraints(
    const ocs2::PinocchioEndEffectorKinematicsCppAd& pinocchioEEKinematics,
    const BalancingSettings& settings, const Vec3d& gravity,
    const OptimizationDimensions& dims, bool recompileLibraries)
    : ocs2::StateInputConstraintCppAd(ocs2::ConstraintOrder::Linear),
      pinocchioEEKinPtr_(pinocchioEEKinematics.clone()),
      gravity_(gravity),
      settings_(settings),
      arrangement_(settings.objects, settings.constraints_enabled, gravity),
      dims_(dims) {
    if (pinocchioEEKinematics.getIds().size() != 1) {
        throw std::runtime_error(
            "[TrayBalanaceConstraint] endEffectorKinematics has wrong "
            "number of end effector IDs.");
    }

    // compile the CppAD library
    initialize(dims.x(), dims.u(), 0, "upright_nominal_balancing_constraints",
               "/tmp/ocs2", recompileLibraries, true);
}

VecXad NominalBalancingConstraints::constraintFunction(
    ocs2::ad_scalar_t time, const VecXad& state, const VecXad& input,
    const VecXad& parameters) const {

    RigidBodyState<ocs2::ad_scalar_t> X =
        get_rigid_body_state(pinocchioEEKinPtr_, state, input);

    // Mat3ad C_we = pinocchioEEKinPtr_->getOrientationCppAd(state);
    // Vec3ad angular_vel =
    //     pinocchioEEKinPtr_->getAngularVelocityCppAd(state, input);
    // Vec3ad angular_acc =
    //     pinocchioEEKinPtr_->getAngularAccelerationCppAd(state, input);
    // Vec3ad linear_acc = pinocchioEEKinPtr_->getAccelerationCppAd(state, input);

    BalancedObjectArrangement<ocs2::ad_scalar_t> ad_arrangement =
        arrangement_.cast<ocs2::ad_scalar_t>();
    return ad_arrangement.balancing_constraints(X);
}

ContactForceBalancingConstraints::ContactForceBalancingConstraints(
    const ocs2::PinocchioEndEffectorKinematicsCppAd& pinocchioEEKinematics,
    const BalancingSettings& settings, const Vec3d& gravity,
    const OptimizationDimensions& dims, bool recompileLibraries)
    : ocs2::StateInputConstraintCppAd(ocs2::ConstraintOrder::Linear),
      pinocchioEEKinPtr_(pinocchioEEKinematics.clone()),
      gravity_(gravity),
      settings_(settings),
      dims_(dims) {
    if (pinocchioEEKinematics.getIds().size() != 1) {
        throw std::runtime_error(
            "[TrayBalanaceConstraint] endEffectorKinematics has wrong "
            "number of end effector IDs.");
    }

    // Important: this needs to come before the call to initialize, because it
    // is used in the constraintFunction which is called therein
    num_constraints_ = settings_.contacts.size() *
                       NUM_LINEARIZED_FRICTION_CONSTRAINTS_PER_CONTACT;

    // compile the CppAD library
    initialize(dims.x(), dims.u(), 0, "upright_contact_force_constraints",
               "/tmp/ocs2", recompileLibraries, true);
}

VecXad ContactForceBalancingConstraints::constraintFunction(
    ocs2::ad_scalar_t time, const VecXad& state, const VecXad& input,
    const VecXad& parameters) const {
    // All forces are expressed in the EE frame
    VecXad forces = input.tail(dims_.f());

    std::vector<ContactPoint<ocs2::ad_scalar_t>> ad_contacts;
    for (auto& contact : settings_.contacts) {
        ad_contacts.push_back(contact.template cast<ocs2::ad_scalar_t>());
    }

    return compute_contact_force_constraints_linearized(ad_contacts, forces);
}

ObjectDynamicsConstraints::ObjectDynamicsConstraints(
    const ocs2::PinocchioEndEffectorKinematicsCppAd& pinocchioEEKinematics,
    const BalancingSettings& settings, const Vec3d& gravity,
    const OptimizationDimensions& dims, bool recompileLibraries)
    : ocs2::StateInputConstraintCppAd(ocs2::ConstraintOrder::Linear),
      pinocchioEEKinPtr_(pinocchioEEKinematics.clone()),
      gravity_(gravity),
      settings_(settings),
      dims_(dims) {
    if (pinocchioEEKinematics.getIds().size() != 1) {
        throw std::runtime_error(
            "[TrayBalanaceConstraint] endEffectorKinematics has wrong "
            "number of end effector IDs.");
    }

    // Six constraints per object: three linear and three rotational.
    num_constraints_ =
        settings_.objects.size() * NUM_DYNAMICS_CONSTRAINTS_PER_OBJECT;

    // compile the CppAD library
    initialize(dims.x(), dims.u(), 0, "upright_object_dynamics_constraints",
               "/tmp/ocs2", recompileLibraries, true);
}

VecXad ObjectDynamicsConstraints::constraintFunction(
    ocs2::ad_scalar_t time, const VecXad& state, const VecXad& input,
    const VecXad& parameters) const {
    // All forces are expressed in the EE frame
    VecXad forces = input.tail(dims_.f());

    RigidBodyState<ocs2::ad_scalar_t> X =
        get_rigid_body_state(pinocchioEEKinPtr_, state, input);

    // Convert contact points to AD type
    std::vector<ContactPoint<ocs2::ad_scalar_t>> ad_contacts;
    for (const auto& contact : settings_.contacts) {
        ad_contacts.push_back(contact.template cast<ocs2::ad_scalar_t>());
    }

    // Convert objects to AD type
    std::map<std::string, BalancedObject<ocs2::ad_scalar_t>> ad_objects;
    for (const auto& kv : settings_.objects) {
        auto obj_ad = kv.second.template cast<ocs2::ad_scalar_t>();
        ad_objects.emplace(kv.first, obj_ad);
    }

    Vec3ad ad_gravity = gravity_.template cast<ocs2::ad_scalar_t>();

    return compute_object_dynamics_constraints(ad_objects, ad_contacts, forces,
                                               X, ad_gravity);
}

}  // namespace upright
