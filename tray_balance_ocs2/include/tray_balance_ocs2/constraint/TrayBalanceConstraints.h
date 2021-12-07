#pragma once

#include <memory>

#include <ocs2_pinocchio_interface/PinocchioEndEffectorKinematicsCppAd.h>
#include <ocs2_robotic_tools/common/SkewSymmetricMatrix.h>
#include <ocs2_robotic_tools/end_effector/EndEffectorKinematics.h>
#include <tray_balance_ocs2/definitions.h>

#include <ocs2_core/constraint/StateInputConstraintCppAd.h>
#include <ocs2_oc/synchronized_module/ReferenceManager.h>

#include <tray_balance_constraints/inequality_constraints.h>
#include <tray_balance_constraints/robust.h>
#include <tray_balance_ocs2/constraint/TrayBalanceConfigurations.h>

namespace ocs2 {
namespace mobile_manipulator {


class TrayBalanceConstraints final : public StateInputConstraintCppAd {
   public:
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
        // size_t n_cylinder_con = 2 + 1;
        return 2 * 3;
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

        // for non-robust baseline approach
        // ad_vector_t constraints = balancing_constraints<ad_scalar_t>(
        //     C_we, angular_vel, linear_acc, angular_acc, stack3_config());

        // Vec3<ad_scalar_t> center(
        //     ad_scalar_t(0), ad_scalar_t(0), ad_scalar_t(0.02 + 0.01));  // tray's CoM
        // ad_scalar_t radius(0.12); // for flat

        // stacked
        // Vec3<ad_scalar_t> center1(ad_scalar_t(0), ad_scalar_t(0), ad_scalar_t(0.21));
        // ad_scalar_t radius1(0.21);
        Vec3<ad_scalar_t> center1(ad_scalar_t(0), ad_scalar_t(0), ad_scalar_t(0.1));
        ad_scalar_t radius1(0.12);
        Ball<ad_scalar_t> ball1(center1, radius1);

        Vec3<ad_scalar_t> center2(ad_scalar_t(0), ad_scalar_t(0), ad_scalar_t(0.3));
        ad_scalar_t radius2(0.12);
        Ball<ad_scalar_t> ball2(center2, radius2);

        ad_scalar_t min_support_dist(0.05);
        ad_scalar_t min_mu(0.5);
        ad_scalar_t min_r_tau = circle_r_tau(min_support_dist);

        ParameterSet<ad_scalar_t> param_set({ball1, ball2}, min_support_dist,
                                            min_mu, min_r_tau);

        ad_vector_t constraints = robust_balancing_constraints<ad_scalar_t>(
            C_we, angular_vel, linear_acc, angular_acc, param_set);

        return constraints;
    }

   private:
    TrayBalanceConstraints(const TrayBalanceConstraints& other) = default;

    std::unique_ptr<PinocchioEndEffectorKinematicsCppAd> pinocchioEEKinPtr_;
};

}  // namespace mobile_manipulator
}  // namespace ocs2
