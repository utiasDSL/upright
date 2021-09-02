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
    // using ad_quaternion_t = Eigen::Quaternion<ad_scalar_t>;

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
        return 3;
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

        // TrayBalanceParameters<ad_scalar_t> params;
        // BalancedBody(RigidBody<Scalar>& body, Scalar com_height,
        //              SupportAreaBase<Scalar> support_area, Scalar r_tau,
        //              Scalar mu)
        ad_scalar_t tray_mass = ad_scalar_t(0.5);
        ad_scalar_t tray_com_height = ad_scalar_t(0.01);
        ad_scalar_t tray_height = tray_com_height * 2;
        ad_scalar_t tray_radius = ad_scalar_t(0.25);
        ad_scalar_t tray_mu = ad_scalar_t(0.5);
        ad_scalar_t ee_side_length = ad_scalar_t(0.3);
        ad_vector_t tray_com(3);  // wrt the EE frame
        tray_com << ad_scalar_t(0), ad_scalar_t(0), ad_scalar_t(0.067);  // TODO

        Eigen::Matrix<ad_scalar_t, 3, 3> tray_inertia =
            cylinder_inertia_matrix<ad_scalar_t>(tray_mass, tray_radius,
                                                 tray_height);
        RigidBody<ad_scalar_t> tray_body(tray_mass, tray_inertia, tray_com);

        ad_scalar_t tray_r_tau =
            equilateral_triangle_inscribed_radius<ad_scalar_t>(ee_side_length);
        Eigen::Matrix<ad_scalar_t, 2, 1> tray_support_offset =
            Eigen::Matrix<ad_scalar_t, 2, 1>::Zero();
        CircleSupportArea<ad_scalar_t> tray_support_area(
            tray_r_tau, tray_support_offset, ad_scalar_t(0));

        BalancedObject<ad_scalar_t> tray(
            tray_body, tray_com_height, tray_support_area, tray_r_tau, tray_mu);

        // TODO single object for now
        ad_vector_t constraints = inequality_constraints<ad_scalar_t>(
            C_we, angular_vel, linear_acc, angular_acc, tray);
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
