#pragma once

#include <Eigen/Eigen>

#include "upright_core/bounded.h"
#include "upright_core/contact.h"
#include "upright_core/types.h"
#include "upright_core/util.h"

namespace upright {

// Number of constraints per contact. One constraint for the normal force
// to be non-negative; one for the friction cone.
const size_t NUM_CONSTRAINTS_PER_CONTACT = 2;

// Three for linear and three for rotation.
const size_t NUM_DYNAMICS_CONSTRAINTS_PER_OBJECT = 6;

template <typename Scalar>
VecX<Scalar> compute_contact_force_constraints(
    const std::vector<ContactPoint<Scalar>>& contacts,
    const VecX<Scalar> forces) {
    VecX<Scalar> constraints(contacts.size() * NUM_CONSTRAINTS_PER_CONTACT);
    for (int i = 0; i < contacts.size(); ++i) {
        auto& contact = contacts[i];
        Vec3<Scalar> f = forces.segment(i * 3, 3);

        // normal force
        Scalar f_n = contact.normal.dot(f);

        // squared magnitude of tangential force
        Scalar f_t_squared = f.dot(f) - f_n * f_n;

        // constrain the normal force to be non-negative
        constraints(i * NUM_CONSTRAINTS_PER_CONTACT) = f_n;

        // non-linear exact version of friction cone
        constraints(i * NUM_CONSTRAINTS_PER_CONTACT + 1) =
            contact.mu * contact.mu * f_n * f_n - f_t_squared;

        // linearized version
        // Vec2<Scalar> f_xy = f.head(2);
        // constraints(i * NUM_CONSTRAINTS_PER_CONTACT + 1) =
        //     contact.mu * f_n - f_xy(0) - f_xy(1);
        // constraints(i * NUM_CONSTRAINTS_PER_CONTACT + 2) =
        //     contact.mu * f_n - f_xy(0) + f_xy(1);
        // constraints(i * NUM_CONSTRAINTS_PER_CONTACT + 3) =
        //     contact.mu * f_n + f_xy(0) - f_xy(1);
        // constraints(i * NUM_CONSTRAINTS_PER_CONTACT + 4) =
        //     contact.mu * f_n + f_xy(0) + f_xy(1);
    }
    return constraints;
}

template <typename Scalar>
Wrench<Scalar> compute_object_dynamics_constraint(
    const BoundedBalancedObject<Scalar>& object, const Wrench<Scalar>& wrench,
    const RigidBodyState<Scalar>& state, const Vec3<Scalar>& gravity) {
    Scalar m = object.body.mass_min;
    Mat3<Scalar> C_ew = state.pose.orientation.transpose();
    Vec3<Scalar> inertial_force =
        m * C_ew *
        (state.acceleration.linear +
         dC_dtt(state) * object.body.com_ellipsoid.center());

    Vec3<Scalar> angular_vel_e = C_ew * state.velocity.angular;
    Vec3<Scalar> angular_acc_e = C_ew * state.acceleration.angular;
    Mat3<Scalar> I_e = m * object.body.radii_of_gyration_matrix();
    Vec3<Scalar> inertial_torque =
        angular_vel_e.cross(I_e * angular_vel_e) + I_e * angular_acc_e;

    Wrench<Scalar> constraints;
    constraints.force = inertial_force - m * gravity - wrench.force;
    constraints.torque = inertial_torque - wrench.torque;
    return constraints;
}

// Sum the forces and torques acting on each object at their various contact
// points.
template <typename Scalar>
std::map<std::string, Wrench<Scalar>> compute_object_wrenches(
    const std::vector<ContactPoint<Scalar>>& contacts,
    const VecX<Scalar> forces) {
    std::map<std::string, Wrench<Scalar>> object_wrenches;
    for (int i = 0; i < contacts.size(); ++i) {
        auto& contact = contacts[i];
        Vec3<Scalar> f = forces.segment(i * 3, 3);

        // TODO this relies on object frame having same orientation as EE frame
        if (object_wrenches.find(contact.object1_name) ==
            object_wrenches.end()) {
            object_wrenches[contact.object1_name].force = f;
            object_wrenches[contact.object1_name].torque =
                contact.r_co_o1.cross(f);
        } else {
            object_wrenches[contact.object1_name].force += f;
            object_wrenches[contact.object1_name].torque +=
                contact.r_co_o1.cross(f);
        }

        // For the second object, the forces are negative
        if (object_wrenches.find(contact.object2_name) ==
            object_wrenches.end()) {
            object_wrenches[contact.object2_name].force = -f;
            object_wrenches[contact.object2_name].torque =
                contact.r_co_o2.cross(-f);
        } else {
            object_wrenches[contact.object2_name].force -= f;
            object_wrenches[contact.object2_name].torque +=
                contact.r_co_o2.cross(-f);
        }
    }
    return object_wrenches;
}

// Compute the dynamics constraints for all objects given the contact points
// and corresponding vector of contact forces.
template <typename Scalar>
VecX<Scalar> compute_object_dynamics_constraints(
    const std::map<std::string, BoundedBalancedObject<Scalar>>& objects,
    const std::vector<ContactPoint<Scalar>>& contacts,
    const VecX<Scalar>& forces, const RigidBodyState<Scalar>& state,
    const Vec3<Scalar>& gravity) {
    std::map<std::string, Wrench<Scalar>> object_wrenches =
        compute_object_wrenches(contacts, forces);

    VecX<Scalar> constraints(NUM_DYNAMICS_CONSTRAINTS_PER_OBJECT *
                             objects.size());
    size_t i = 0;
    for (const auto& kv : objects) {
        auto& name = kv.first;
        auto& object = kv.second;

        Wrench<Scalar> wrench = object_wrenches[name];
        Wrench<Scalar> constraint =
            compute_object_dynamics_constraint(object, wrench, state, gravity);
        constraints.segment(i * NUM_DYNAMICS_CONSTRAINTS_PER_OBJECT,
                            NUM_DYNAMICS_CONSTRAINTS_PER_OBJECT)
            << constraint.force,
            constraint.torque;
        i++;
    }
    return constraints;
}

}  // namespace upright
