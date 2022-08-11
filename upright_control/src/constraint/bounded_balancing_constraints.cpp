#include <upright_control/types.h>
#include <upright_core/bounded.h>
#include <upright_core/contact.h>

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

BoundedBalancingConstraints::BoundedBalancingConstraints(
    const ocs2::PinocchioEndEffectorKinematicsCppAd& pinocchioEEKinematics,
    const BalancingSettings& settings, const Vec3d& gravity,
    const RobotDimensions& dims, bool recompileLibraries)
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

    // compile the CppAD library
    initialize(dims.x, dims.u, 0, "upright_bounded_balancing_constraints",
               "/tmp/ocs2", recompileLibraries, true);

    num_constraints_ = num_balancing_constraints(settings_.objects);
}

VecXad BoundedBalancingConstraints::constraintFunction(
    ocs2::ad_scalar_t time, const VecXad& state, const VecXad& input,
    const VecXad& parameters) const {
    Mat3ad C_we = pinocchioEEKinPtr_->getOrientationCppAd(state);
    Vec3ad angular_vel =
        pinocchioEEKinPtr_->getAngularVelocityCppAd(state, input);
    Vec3ad angular_acc =
        pinocchioEEKinPtr_->getAngularAccelerationCppAd(state, input);
    Vec3ad linear_acc = pinocchioEEKinPtr_->getAccelerationCppAd(state, input);

    // Cast to AD scalar type
    Vec3ad ad_gravity = gravity_.template cast<ocs2::ad_scalar_t>();
    std::vector<BoundedBalancedObject<ocs2::ad_scalar_t>> ad_objects;
    for (const auto& kv : settings_.objects) {
        ad_objects.push_back(kv.second.cast<ocs2::ad_scalar_t>());
    }

    return balancing_constraints(ad_objects, ad_gravity,
                                 settings_.constraints_enabled, C_we,
                                 angular_vel, linear_acc, angular_acc);
}

ContactForceBalancingConstraints::ContactForceBalancingConstraints(
    const ocs2::PinocchioEndEffectorKinematicsCppAd& pinocchioEEKinematics,
    const BalancingSettings& settings, const Vec3d& gravity,
    const RobotDimensions& dims, bool recompileLibraries)
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

    // compile the CppAD library
    initialize(dims.x, dims.u, 0, "upright_contact_force_balancing_constraints",
               "/tmp/ocs2", recompileLibraries, true);

    num_constraints_ = settings_.contacts.size() * NUM_CONSTRAINTS_PER_CONTACT;
}

VecXad ContactForceBalancingConstraints::constraintFunction(
    ocs2::ad_scalar_t time, const VecXad& state, const VecXad& input,
    const VecXad& parameters) const {
    // All forces are expressed in the EE frame
    VecXad forces = input.tail(3 * dims_.f);

    VecXad constraints(num_constraints_);
    for (int i = 0; i < dims_.f; ++i) {
        // Convert the contact point to AD type
        ContactPoint<ocs2::ad_scalar_t> contact =
            settings_.contacts[i].template cast<ocs2::ad_scalar_t>();
        Vec3ad f = forces.segment(i * 3, 3);

        // Normal force
        ocs2::ad_scalar_t fn = f.dot(contact.normal);

        // Squared magnitude of tangential force
        ocs2::ad_scalar_t ft_squared = f.dot(f) - fn * fn;

        // Constrain the normal force to be non-negative
        constraints(i * NUM_CONSTRAINTS_PER_CONTACT) = fn;

        // Constrain force to lie in friction cone
        // TODO this is not linearized
        constraints(i * NUM_CONSTRAINTS_PER_CONTACT + 1) =
            contact.mu * fn * fn - ft_squared;
    }
    return constraints;
}

ContactForceBalancingEqualityConstraints::
    ContactForceBalancingEqualityConstraints(
        const ocs2::PinocchioEndEffectorKinematicsCppAd& pinocchioEEKinematics,
        const BalancingSettings& settings, const Vec3d& gravity,
        const RobotDimensions& dims, bool recompileLibraries)
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

    // compile the CppAD library
    initialize(dims.x, dims.u, 0,
               "upright_contact_force_balancing_equality_constraints",
               "/tmp/ocs2", recompileLibraries, true);

    // Six constraints per object: three linear and three rotational.
    num_constraints_ = settings_.objects.size() * 6;
}

VecXad ContactForceBalancingEqualityConstraints::constraintFunction(
    ocs2::ad_scalar_t time, const VecXad& state, const VecXad& input,
    const VecXad& parameters) const {

    Mat3ad C_we = pinocchioEEKinPtr_->getOrientationCppAd(state);
    Vec3ad angular_vel =
        pinocchioEEKinPtr_->getAngularVelocityCppAd(state, input);
    Vec3ad angular_acc =
        pinocchioEEKinPtr_->getAngularAccelerationCppAd(state, input);
    Vec3ad linear_acc = pinocchioEEKinPtr_->getAccelerationCppAd(state, input);

    // All forces are expressed in the EE frame
    VecXad forces = input.tail(3 * dims_.f);

    // First we need to compute the total wrench on each object
    std::map<std::string, Vec3ad> obj_forces;
    std::map<std::string, Vec3ad> obj_torques;
    for (int i = 0; i < dims_.f; ++i) {
        // Convert the contact point to AD type
        ContactPoint<ocs2::ad_scalar_t> contact =
            settings_.contacts[i].template cast<ocs2::ad_scalar_t>();
        Vec3ad f = forces.segment(i * 3, 3);

        // auto obj1_force = obj_forces.find(contact.object1_name);
        // auto obj2_torque = obj_torques.find(contact.object1_name);
        if (obj_forces.find(contact.object1_name) == obj_forces.end()) {
            // TODO this relies on object frame having same orientation as EE
            // frame
            obj_forces[contact.object1_name] = f;
            obj_torques[contact.object1_name] = contact.r_co_o1.cross(f);
        } else {
            obj_forces[contact.object1_name] += f;
            obj_torques[contact.object1_name] += contact.r_co_o1.cross(f);
        }

        // For the second object, the forces are negative
        if (obj_forces.find(contact.object2_name) == obj_forces.end()) {
            obj_forces[contact.object2_name] = -f;
            obj_torques[contact.object2_name] = contact.r_co_o2.cross(-f);
        } else {
            obj_forces[contact.object2_name] -= f;
            obj_torques[contact.object2_name] += contact.r_co_o2.cross(-f);
        }
    }

    // Now we can express the constraint
    VecXad constraints(num_constraints_);
    Vec3ad ad_gravity = gravity_.template cast<ocs2::ad_scalar_t>();
    size_t i = 0;
    for (const auto& kv : settings_.objects) {
        auto ad_obj = kv.second.template cast<ocs2::ad_scalar_t>();

        // Linear dynamics (in the inertial/world frame)
        ocs2::ad_scalar_t m = ad_obj.body.mass_min;
        Vec3ad total_force = C_we * obj_forces[kv.first] + m * ad_gravity;
        Vec3ad desired_force = m * (linear_acc + C_we * ad_obj.body.com_ellipsoid.center());
        constraints.segment(i * 6, 3) = total_force - desired_force;

        // Rotational dynamics
        Vec3ad total_torque = C_we * obj_torques[kv.first];
        Mat3ad inertia = m * C_we * ad_obj.body.radii_of_gyration_matrix() * C_we.transpose();
        Vec3ad desired_torque = angular_vel.cross(inertia * angular_vel) + inertia * angular_acc;
        constraints.segment(i * 6 + 3, 3) = total_torque - desired_torque;

        i += 1;
    }
    return constraints;
}

}  // namespace upright
