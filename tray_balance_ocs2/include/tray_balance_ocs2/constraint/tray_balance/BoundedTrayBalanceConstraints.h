#pragma once

#include <memory>

#include <ocs2_pinocchio_interface/PinocchioEndEffectorKinematicsCppAd.h>
#include <ocs2_robotic_tools/common/SkewSymmetricMatrix.h>
#include <ocs2_robotic_tools/end_effector/EndEffectorKinematics.h>
#include <tray_balance_ocs2/definitions.h>

#include <ocs2_core/constraint/StateInputConstraintCppAd.h>
#include <ocs2_oc/synchronized_module/ReferenceManager.h>

#include <tray_balance_constraints/robust2.h>

namespace ocs2 {
namespace mobile_manipulator {

class BoundedTrayBalanceConstraints final : public StateInputConstraintCppAd {
   public:
    using ad_quaternion_t =
        PinocchioEndEffectorKinematicsCppAd::ad_quaternion_t;
    using ad_rotmat_t = PinocchioEndEffectorKinematicsCppAd::ad_rotmat_t;

    using ad_vec2_t = Eigen::Matrix<ad_scalar_t, 2, 1>;
    using ad_vec3_t = Eigen::Matrix<ad_scalar_t, 3, 1>;
    using ad_mat3_t = Eigen::Matrix<ad_scalar_t, 3, 3>;

    BoundedTrayBalanceConstraints(
        const PinocchioEndEffectorKinematicsCppAd& pinocchioEEKinematics,
        const BoundedTrayBalanceConfiguration<scalar_t>& config,
        bool recompileLibraries)
        : StateInputConstraintCppAd(ConstraintOrder::Linear),
          pinocchioEEKinPtr_(pinocchioEEKinematics.clone()),
          config_(config) {
        if (pinocchioEEKinematics.getIds().size() != 1) {
            throw std::runtime_error(
                "[TrayBalanaceConstraint] endEffectorKinematics has wrong "
                "number of end effector IDs.");
        }

        // compile the CppAD library
        initialize(STATE_DIM, INPUT_DIM, 0, "bounded_tray_balance_constraints",
                   "/tmp/ocs2", recompileLibraries, true);
    }

    BoundedTrayBalanceConstraints* clone() const override {
        // Always pass recompileLibraries = false to avoid recompiling the same
        // library just because this object is cloned.
        return new BoundedTrayBalanceConstraints(*pinocchioEEKinPtr_, config_,
                                                 false);
    }

    size_t getNumConstraints(scalar_t time) const override {
        // return config_.num_constraints();
        return 1;
    }

    size_t getNumConstraints() const { return getNumConstraints(0); }

    vector_t getParameters(scalar_t time) const override {
        // Parameters are constant for now
        // return params_;
        return vector_t(0);
    }

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

        // auto config = config_.cast_with_parameters<ad_scalar_t>(parameters);
        auto config = config_.cast<ad_scalar_t>();
        return config.balancing_constraints(C_we, angular_vel, linear_acc,
                                            angular_acc);
    }

   private:
    BoundedTrayBalanceConstraints(const BoundedTrayBalanceConstraints& other) =
        default;

    std::unique_ptr<PinocchioEndEffectorKinematicsCppAd> pinocchioEEKinPtr_;
    BoundedTrayBalanceConfiguration<scalar_t> config_;
    // vector_t params_;  // TODO unused
};

}  // namespace mobile_manipulator
}  // namespace ocs2
