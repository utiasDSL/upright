#pragma once

#include <memory>

#include <ocs2_pinocchio_interface/PinocchioEndEffectorKinematicsCppAd.h>
#include <ocs2_robotic_tools/common/SkewSymmetricMatrix.h>
#include <ocs2_robotic_tools/end_effector/EndEffectorKinematics.h>
#include <ocs2_core/constraint/StateInputConstraintCppAd.h>
#include <ocs2_oc/synchronized_module/ReferenceManager.h>

#include <tray_balance_constraints/nominal.h>

#include <tray_balance_ocs2/dynamics/Dimensions.h>

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
        const PinocchioEndEffectorKinematicsCppAd& pinocchioEEKinematics,
        const TrayBalanceConfiguration<scalar_t>& config,
        const RobotDimensions& dims,
        bool recompileLibraries)
        : StateInputConstraintCppAd(ConstraintOrder::Linear),
          pinocchioEEKinPtr_(pinocchioEEKinematics.clone()),
          config_(config), dims_(dims),
          params_(config.get_parameters()) {
        if (pinocchioEEKinematics.getIds().size() != 1) {
            throw std::runtime_error(
                "[TrayBalanaceConstraint] endEffectorKinematics has wrong "
                "number of end effector IDs.");
        }

        // initialize everything, mostly the CppAD interface (compile the
        // library)
        initialize(dims.x, dims.u, config_.num_parameters(),
                   "tray_balance_constraints", "/tmp/ocs2", recompileLibraries,
                   true);
    }

    TrayBalanceConstraints* clone() const override {
        // Always pass recompileLibraries = false to avoid recompiling the same
        // library just because this object is cloned.
        return new TrayBalanceConstraints(*pinocchioEEKinPtr_, config_, dims_, false);
    }

    size_t getNumConstraints(scalar_t time) const override {
        return config_.num_constraints();
    }

    size_t getNumConstraints() const { return getNumConstraints(0); }

    vector_t getParameters(scalar_t time) const override {
        // Parameters are constant for now
        return params_;
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

        auto config = config_.cast_with_parameters<ad_scalar_t>(parameters);
        return config.balancing_constraints(C_we, angular_vel, linear_acc,
                                            angular_acc);
    }

   private:
    TrayBalanceConstraints(const TrayBalanceConstraints& other) = default;

    std::unique_ptr<PinocchioEndEffectorKinematicsCppAd> pinocchioEEKinPtr_;
    TrayBalanceConfiguration<scalar_t> config_;
    vector_t params_;
    RobotDimensions dims_;
};

}  // namespace mobile_manipulator
}  // namespace ocs2
