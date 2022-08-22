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

    // Important: this needs to come before the call to initialize, because it
    // is used in the constraintFunction which is called therein
    num_constraints_ = settings_.contacts.size() * NUM_CONSTRAINTS_PER_CONTACT;

    // compile the CppAD library
    initialize(dims.x, dims.u, 0, "upright_contact_force_constraints",
               "/tmp/ocs2", recompileLibraries, true);
}

// Compute the a basis for the nullspace of the vector v.
MatXd null(const Vec3d& v) {
    Eigen::FullPivLU<MatXd> lu(v.transpose());
    MatXd N = lu.kernel();
    N.col(0).normalize();
    N.col(1).normalize();
    return N;
}

VecXad ContactForceBalancingConstraints::constraintFunction(
    ocs2::ad_scalar_t time, const VecXad& state, const VecXad& input,
    const VecXad& parameters) const {
    // All forces are expressed in the EE frame
    VecXad forces = input.tail(3 * dims_.f);

    VecXad constraints(num_constraints_);
    for (int i = 0; i < dims_.f; ++i) {
        MatXad N = null(settings_.contacts[i].normal).template cast<ocs2::ad_scalar_t>();

        // Convert the contact point to AD type
        ContactPoint<ocs2::ad_scalar_t> contact =
            settings_.contacts[i].template cast<ocs2::ad_scalar_t>();
        Vec3ad f = forces.segment(i * 3, 3);

        // Normal force
        // ocs2::ad_scalar_t fn = f.dot(contact.normal);
        ocs2::ad_scalar_t f_x = N.col(0).dot(f);
        ocs2::ad_scalar_t f_y = N.col(1).dot(f);
        ocs2::ad_scalar_t f_z = contact.normal.dot(f);

        // Squared magnitude of tangential force
        // ocs2::ad_scalar_t ft_squared = f.dot(f) - fn * fn;

        // TODO for now we assume normal is (0, 0, 1)
        Vec2ad f_xy = f.head(2);
        ocs2::ad_scalar_t fn = f(2);

        // Constrain the normal force to be non-negative
        constraints(i * NUM_CONSTRAINTS_PER_CONTACT) = fn;

        // TODO we need to compute f_x and f_y

        // Constrain force to lie in friction cone
        // TODO this is not linearized
        // constraints(i * NUM_CONSTRAINTS_PER_CONTACT + 1) =
        //     // contact.mu * contact.mu * fn * fn - ft_squared;
        //     (1 + contact.mu * contact.mu) * fn * fn - f.dot(f);

        constraints(i * NUM_CONSTRAINTS_PER_CONTACT + 1) = contact.mu * fn - f_xy(0) - f_xy(1);
        constraints(i * NUM_CONSTRAINTS_PER_CONTACT + 2) = contact.mu * fn - f_xy(0) + f_xy(1);
        constraints(i * NUM_CONSTRAINTS_PER_CONTACT + 3) = contact.mu * fn + f_xy(0) - f_xy(1);
        constraints(i * NUM_CONSTRAINTS_PER_CONTACT + 4) = contact.mu * fn + f_xy(0) + f_xy(1);

        // constraints(i * NUM_CONSTRAINTS_PER_CONTACT + 1) = contact.mu * fn - f_xy(0);
        // constraints(i * NUM_CONSTRAINTS_PER_CONTACT + 2) = contact.mu * fn + f_xy(0);
        // constraints(i * NUM_CONSTRAINTS_PER_CONTACT + 3) = contact.mu * fn - f_xy(1);
        // constraints(i * NUM_CONSTRAINTS_PER_CONTACT + 4) = contact.mu * fn + f_xy(1);
    }
    return constraints;
}

ObjectDynamicsConstraints::ObjectDynamicsConstraints(
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

    // Six constraints per object: three linear and three rotational.
    num_constraints_ = settings_.objects.size() * 6;

    // compile the CppAD library
    initialize(dims.x, dims.u, 0, "upright_object_dynamics_constraints",
               "/tmp/ocs2", recompileLibraries, true);
}

VecXad ObjectDynamicsConstraints::constraintFunction(
    ocs2::ad_scalar_t time, const VecXad& state, const VecXad& input,
    const VecXad& parameters) const {
    Mat3ad C_we = pinocchioEEKinPtr_->getOrientationCppAd(state);
    Vec3ad angular_vel =
        pinocchioEEKinPtr_->getAngularVelocityCppAd(state, input);
    Vec3ad angular_acc =
        pinocchioEEKinPtr_->getAngularAccelerationCppAd(state, input);
    Vec3ad linear_acc = pinocchioEEKinPtr_->getAccelerationCppAd(state, input);

    Mat3ad C_ew = C_we.transpose();
    Vec3ad angular_vel_e = C_ew * angular_vel;
    Vec3ad angular_acc_e = C_ew * angular_acc;
    Mat3ad ddC_we =
        rotation_matrix_second_derivative(C_we, angular_vel, angular_acc);

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
        Vec3ad total_force = obj_forces[kv.first];
        Vec3ad desired_force =
            m * C_ew * (linear_acc + ddC_we * ad_obj.body.com_ellipsoid.center() - ad_gravity);
        constraints.segment(i * 6, 3) = total_force - desired_force;

        // Rotational dynamics
        Vec3ad total_torque = obj_torques[kv.first];
        Mat3ad I_e = m * ad_obj.body.radii_of_gyration_matrix();
        Vec3ad desired_torque = angular_vel_e.cross(I_e * angular_vel_e) + I_e * angular_acc_e;
        constraints.segment(i * 6 + 3, 3) = total_torque - desired_torque;

        i += 1;
    }
    return constraints;
}

}  // namespace upright
