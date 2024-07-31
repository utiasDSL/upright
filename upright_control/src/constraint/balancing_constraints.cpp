#include "upright_control/constraint/balancing_constraints.h"

#include <upright_control/types.h>
#include <upright_core/contact.h>
#include <upright_core/contact_constraints.h>

namespace upright {

std::ostream& operator<<(std::ostream& out, const BalancingSettings& settings) {
    out << "enabled = " << settings.enabled << std::endl
        << "num bodies = " << settings.bodies.size() << std::endl;
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
            "[ContactForceBalancingConstraints] endEffectorKinematics has "
            "wrong "
            "number of end effector IDs.");
    }

    // Important: this needs to come before the call to initialize, because it
    // is used in the constraintFunction which is called therein
    num_constraints_ = settings_.contacts.size() *
                       NUM_LINEARIZED_FRICTION_CONSTRAINTS_PER_CONTACT;

    // compile the CppAD library
    const std::string lib_name = "upright_contact_force_constraints";
    initialize(dims.x(), dims.u(), 0, lib_name, "/tmp/ocs2", recompileLibraries,
               true);
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
            "[ObjectDynamicsConstraints] endEffectorKinematics has wrong "
            "number of end effector IDs.");
    }

    // Six constraints per object: three linear and three rotational.
    num_constraints_ =
        settings_.bodies.size() * NUM_DYNAMICS_CONSTRAINTS_PER_OBJECT;

    const size_t num_params = settings_.bodies.size() * 10;  // TODO unhardcode
                                                             //
    size_t i = 0;
    VecXd parameters(num_params);
    for (const auto& kv : settings_.bodies) {
        auto& body = kv.second;
        parameters.segment(i, 10) = body.get_parameters();
        i += body.num_parameters();
    }
    parameters_ = parameters;

    // compile the CppAD library
    const std::string lib_name = "upright_object_dynamics_constraints";
    initialize(dims.x(), dims.u(), num_params, lib_name, "/tmp/ocs2",
               recompileLibraries, true);
}

VecXd ObjectDynamicsConstraints::getParameters(ocs2::scalar_t time) const {
    return parameters_;
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

    // Convert bodies to AD type
    size_t i = 0;
    std::map<std::string, RigidBody<ocs2::ad_scalar_t>> ad_bodies;
    for (const auto& kv : settings_.bodies) {
        auto ad_body =
            RigidBody<ocs2::ad_scalar_t>::from_parameters(parameters, i);
        ad_bodies.emplace(kv.first, ad_body);
        i += ad_body.num_parameters();
    }

    Vec3ad ad_gravity = gravity_.template cast<ocs2::ad_scalar_t>();

    // Normalizing by the number of constraints appears to improve the
    // convergence of the controller (cost landscape is better behaved)
    // TODO
    // ocs2::ad_scalar_t n(sqrt(6 * ad_bodies.size()));
    // ocs2::ad_scalar_t n(ad_bodies.size());
    // ocs2::ad_scalar_t n(sqrt(ad_bodies.size()));
    // return compute_object_dynamics_constraints(ad_bodies, ad_contacts, forces,
    //                                            X, ad_gravity) /
    //        n;
    return compute_object_dynamics_constraints(ad_bodies, ad_contacts, forces,
                                               X, ad_gravity);
}

}  // namespace upright
