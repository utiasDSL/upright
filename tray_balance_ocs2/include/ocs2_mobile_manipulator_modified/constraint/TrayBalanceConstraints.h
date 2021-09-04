#pragma once

#include <memory>

// #include <ocs2_mobile_manipulator_modified/constraint/TrayBalanceUtil.h>
#include <ocs2_mobile_manipulator_modified/definitions.h>
#include <ocs2_pinocchio_interface/PinocchioEndEffectorKinematicsCppAd.h>
#include <ocs2_robotic_tools/common/SkewSymmetricMatrix.h>
#include <ocs2_robotic_tools/end_effector/EndEffectorKinematics.h>

#include <ocs2_core/constraint/StateInputConstraintCppAd.h>
#include <ocs2_oc/synchronized_module/ReferenceManager.h>

#include <tray_balance_constraints/inequality_constraints.h>

namespace ocs2 {
namespace mobile_manipulator {

class TrayBalanceConstraints final : public StateInputConstraintCppAd {
   public:
    // using vector3_t = Eigen::Matrix<scalar_t, 3, 1>;
    // using quaternion_t = Eigen::Quaternion<scalar_t>;
    using ad_quaternion_t =
        PinocchioEndEffectorKinematicsCppAd::ad_quaternion_t;
    using ad_rotmat_t = PinocchioEndEffectorKinematicsCppAd::ad_rotmat_t;

    using ad_vec2_t = Eigen::Matrix<ad_scalar_t, 2, 1>;
    using ad_vec3_t = Eigen::Matrix<ad_scalar_t, 3, 1>;
    using ad_mat3_t = Eigen::Matrix<ad_scalar_t, 3, 3>;

    TrayBalanceConstraints(
        const PinocchioEndEffectorKinematicsCppAd& pinocchioEEKinematics)
        : StateInputConstraintCppAd(ConstraintOrder::Linear),
          pinocchioEEKinPtr_(pinocchioEEKinematics.clone()) {
        if (pinocchioEEKinematics.getIds().size() != 1) {
            throw std::runtime_error(
                "[EndEffectorConstraint] endEffectorKinematics has wrong "
                "number of "
                "end effector IDs.");
        }
        // initialize everything, mostly the CppAD interface
        initialize(STATE_DIM, INPUT_DIM, 0, "tray_balance_constraints",
                   "/tmp/ocs2", true, true);
    }

    // TrayBalanceConstraints() override = default;

    TrayBalanceConstraints* clone() const override {
        return new TrayBalanceConstraints(*pinocchioEEKinPtr_);
    }

    size_t getNumConstraints(scalar_t time) const override {
        // TODO this depends on the configuration at hand
        size_t n_tray_con = 2 + 3;
        size_t n_cuboid_con = 2 + 4;
        return n_tray_con + n_cuboid_con;
    }

    size_t getNumConstraints() const { return getNumConstraints(0); }

   protected:
    ad_vector_t constraintFunction(
        ad_scalar_t time, const ad_vector_t& state, const ad_vector_t& input,
        const ad_vector_t& parameters) const override {
        ad_rotmat_t C_we = pinocchioEEKinPtr_->getOrientationCppAd(state);
        ad_vector_t angular_vel =
            pinocchioEEKinPtr_->getAngularVelocityCppAd(state, input);
        ad_vector_t angular_acc =
            pinocchioEEKinPtr_->getAngularAccelerationCppAd(state, input);
        ad_vector_t linear_acc =
            pinocchioEEKinPtr_->getAccelerationCppAd(state, input);

        // generic object params
        ad_scalar_t obj_mass(1.0);
        ad_scalar_t obj_mu(0.5);
        ad_scalar_t obj_com_height(0.2);
        ad_scalar_t obj_zmp_margin(0.0);
        ad_vec2_t obj_support_offset = ad_vec2_t::Zero();

        // cylinder-specific params
        ad_scalar_t cylinder_radius(0.1);
        ad_scalar_t cylinder_height = obj_com_height * 2;
        ad_scalar_t cylinder_r_tau = circle_r_tau(cylinder_radius);
        ad_vec3_t cylinder_com(ad_scalar_t(0), ad_scalar_t(0),
                               ad_scalar_t(0.25));  // TODO
        ad_mat3_t cylinder_inertia =
            cylinder_inertia_matrix(obj_mass, cylinder_radius, cylinder_height);
        RigidBody<ad_scalar_t> cylinder_body(obj_mass, cylinder_inertia,
                                             cylinder_com);
        CircleSupportArea<ad_scalar_t> cylinder_support_area(
            cylinder_radius, obj_support_offset, obj_zmp_margin);

        // cuboid-specific params
        ad_vec3_t cuboid_side_lengths(ad_scalar_t(0.2), ad_scalar_t(0.2),
                                      obj_com_height * 2);
        ad_mat3_t cuboid_inertia =
            cuboid_inertia_matrix(obj_mass, cuboid_side_lengths);
        // NOTE: this assumes that the cuboid is -0.05 offset
        ad_vec3_t cuboid_com(ad_scalar_t(-0.05), ad_scalar_t(0),
                             ad_scalar_t(0.25));
        RigidBody<ad_scalar_t> cuboid_body(obj_mass, cuboid_inertia,
                                           cuboid_com);
        ad_scalar_t cuboid_r_tau =
            circle_r_tau(cuboid_side_lengths(0) * 0.5);  // TODO

        std::vector<ad_vec2_t> vertices =
            cuboid_support_vertices(cuboid_side_lengths);
        PolygonSupportArea<ad_scalar_t> cuboid_support_area(
            vertices, obj_support_offset, obj_zmp_margin);
        // CircleSupportArea<ad_scalar_t> cuboid_support_area(
        //     ad_scalar_t(0.1), obj_support_offset, obj_zmp_margin);

        BalancedObject<ad_scalar_t> cuboid(cuboid_body, obj_com_height,
                                           cuboid_support_area, cuboid_r_tau,
                                           obj_mu);

        ad_scalar_t tray_mass(0.5);
        ad_scalar_t tray_com_height(0.01);
        ad_scalar_t tray_height = tray_com_height * 2;
        ad_scalar_t tray_radius(0.25);
        ad_scalar_t tray_mu(0.5);
        ad_scalar_t ee_side_length(0.2);
        ad_scalar_t tray_zmp_margin(0.01);
        ad_vec3_t tray_com(3);  // wrt the EE frame
        tray_com << ad_scalar_t(0), ad_scalar_t(0), ad_scalar_t(0.04);

        ad_mat3_t tray_inertia = cylinder_inertia_matrix<ad_scalar_t>(
            tray_mass, tray_radius, tray_height);
        RigidBody<ad_scalar_t> tray_body(tray_mass, tray_inertia, tray_com);

        ad_scalar_t tray_r_tau =
            equilateral_triangle_inscribed_radius<ad_scalar_t>(ee_side_length);
        ad_vec2_t tray_support_offset = ad_vec2_t::Zero();

        // CircleSupportArea<ad_scalar_t> tray_support_area(
        //     tray_r_tau, tray_support_offset, ad_scalar_t(0));
        std::vector<ad_vec2_t> tray_support_vertices =
            equilateral_triangle_support_vertices(ee_side_length);
        PolygonSupportArea<ad_scalar_t> tray_support_area(
            tray_support_vertices, tray_support_offset, tray_zmp_margin);

        BalancedObject<ad_scalar_t> tray(
            tray_body, tray_com_height, tray_support_area, tray_r_tau, tray_mu);

        auto composite = BalancedObject<ad_scalar_t>::compose({tray, cuboid});

        ad_vector_t constraints = balancing_constraints<ad_scalar_t>(
            C_we, angular_vel, linear_acc, angular_acc, {composite, cuboid});
        return constraints;
    }

   private:
    TrayBalanceConstraints(const TrayBalanceConstraints& other) = default;

    std::unique_ptr<PinocchioEndEffectorKinematicsCppAd> pinocchioEEKinPtr_;

    // TODO may not need this
    // const ReferenceManager* referenceManagerPtr_;
};

}  // namespace mobile_manipulator
}  // namespace ocs2
