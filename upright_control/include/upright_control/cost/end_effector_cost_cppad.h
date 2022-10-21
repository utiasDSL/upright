#pragma once

#include <ocs2_core/cost/StateCostCppAd.h>
#include <ocs2_pinocchio_interface/PinocchioEndEffectorKinematicsCppAd.h>

#include <upright_control/dimensions.h>
#include <upright_control/types.h>

namespace upright {

class EndEffectorCostCppAd final : public ocs2::StateCostCppAd {
   public:
    EndEffectorCostCppAd(
        const MatXd& W,
        const ocs2::PinocchioEndEffectorKinematicsCppAd& kinematics,
        const OptimizationDimensions& dims,
        bool recompileLibraries)
        : W_(W),
          kinematics_ptr_(kinematics.clone()),
          dims_(dims) {
        initialize(dims.x(), 7, "end_effector_cost_cppad", "/tmp/ocs2",
                   recompileLibraries, true);
    }

    EndEffectorCostCppAd* clone() const override {
        return new EndEffectorCostCppAd(W_, *kinematics_ptr_, dims_, false);
    }

    VecXd getParameters(ocs2::scalar_t time, const ocs2::TargetTrajectories& target) const override {
        const auto desired_pose = interpolate_end_effector_pose(time, target);
        VecXd p(7);
        p << desired_pose.first, desired_pose.second.coeffs();
        return p;
    };

   protected:
    ocs2::ad_scalar_t costFunction(ocs2::ad_scalar_t time_ad,
                                   const VecXad& state,
                                   const VecXad& parameters) const {
        Vec3ad desired_position = parameters.head(3);
        Quatad desired_orientation;
        desired_orientation.coeffs() = parameters.tail(4);

        Vec3ad actual_position = kinematics_ptr_->getPositionCppAd(state);
        Vec3ad pos_error = actual_position - desired_position;
        Vec3ad orn_error = kinematics_ptr_->getOrientationErrorCppAd(
            state, desired_orientation);

        MatXad W = W_.cast<ocs2::ad_scalar_t>();
        VecXad pose_error(6);
        pose_error << pos_error, orn_error;
        return ocs2::ad_scalar_t(0.5) * pose_error.dot(W * pose_error);
    }

   private:
    EndEffectorCostCppAd(const EndEffectorCostCppAd& other) = default;

    MatXd W_;  // weight matrix
    std::unique_ptr<ocs2::PinocchioEndEffectorKinematicsCppAd> kinematics_ptr_;
    OptimizationDimensions dims_;
};

}  // namespace upright
