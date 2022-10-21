#pragma once

#include <ocs2_core/cost/StateCost.h>
#include <ocs2_core/cost/StateCostCppAd.h>
#include <ocs2_pinocchio_interface/PinocchioEndEffectorKinematics.h>
#include <ocs2_pinocchio_interface/PinocchioEndEffectorKinematicsCppAd.h>
#include <ocs2_robotic_tools/end_effector/EndEffectorKinematics.h>

#include <upright_control/reference_trajectory.h>
#include <upright_control/types.h>

namespace upright {

class EndEffectorCost final : public ocs2::StateCost {
   public:
    EndEffectorCost(
        const MatXd& W,
        const ocs2::EndEffectorKinematics<ocs2::scalar_t>& kinematics)
        : W_(W), kinematics_ptr_(kinematics.clone()) {
        if (kinematics.getIds().size() != 1) {
            throw std::runtime_error(
                "[EndEffectorConstraint] endEffectorKinematics has wrong "
                "number of end effector IDs.");
        }
    }

    ~EndEffectorCost() override = default;

    EndEffectorCost* clone() const override {
        return new EndEffectorCost(W_, *kinematics_ptr_);
    }

    ocs2::scalar_t getValue(ocs2::scalar_t time, const VecXd& state,
                            const ocs2::TargetTrajectories& target,
                            const ocs2::PreComputation&) const override {
        const auto desired_pose = interpolate_end_effector_pose(time, target);

        VecXd err = VecXd::Zero(6);
        err.head<3>() =
            kinematics_ptr_->getPosition(state).front() - desired_pose.first;
        err.tail<3>() =
            kinematics_ptr_->getOrientationError(state, {desired_pose.second})
                .front();

        return 0.5 * err.transpose() * W_ * err;
    }

    ocs2::ScalarFunctionQuadraticApproximation getQuadraticApproximation(
        ocs2::scalar_t time, const VecXd& state,
        const ocs2::TargetTrajectories& target,
        const ocs2::PreComputation&) const override {
        const auto desired_pose = interpolate_end_effector_pose(time, target);

        // NOTE: input is not used in this state cost, so we give it a
        // dimension of zero.
        auto approximation =
            ocs2::ScalarFunctionQuadraticApproximation(state.rows(), 0);
        approximation.setZero(state.rows(), 0);

        // Linear approximations of position and orientation error
        const auto eePosition =
            kinematics_ptr_->getPositionLinearApproximation(state).front();
        const auto eeOrientationError =
            kinematics_ptr_
                ->getOrientationErrorLinearApproximation(state,
                                                         {desired_pose.second})
                .front();

        // Function value
        VecXd e = VecXd::Zero(6);
        e << eePosition.f - desired_pose.first, eeOrientationError.f;
        approximation.f = 0.5 * e.transpose() * W_ * e;

        // Jacobian
        MatXd dedx(6, state.rows());
        dedx.setZero();
        dedx << eePosition.dfdx, eeOrientationError.dfdx;
        approximation.dfdx = e.transpose() * W_ * dedx;

        // Hessian (Gauss-Newton approximation)
        approximation.dfdxx = dedx.transpose() * W_ * dedx;

        return approximation;
    }

   private:
    EndEffectorCost(const EndEffectorCost& other) = default;

    MatXd W_;  // weight matrix
    std::unique_ptr<ocs2::EndEffectorKinematics<ocs2::scalar_t>>
        kinematics_ptr_;
};

// Auto-diff version gives us the full Hessian
class EndEffectorCostCppAd final : public ocs2::StateCostCppAd {
   public:
    EndEffectorCostCppAd(
        const MatXd& W,
        const ocs2::PinocchioEndEffectorKinematicsCppAd& kinematics,
        const OptimizationDimensions& dims, bool recompileLibraries)
        : W_(W), kinematics_ptr_(kinematics.clone()), dims_(dims) {
        const size_t num_params = 7;
        initialize(dims.x(), num_params, "end_effector_cost_cppad", "/tmp/ocs2",
                   recompileLibraries, true);
    }

    EndEffectorCostCppAd* clone() const override {
        return new EndEffectorCostCppAd(W_, *kinematics_ptr_, dims_, false);
    }

    VecXd getParameters(ocs2::scalar_t time,
                        const ocs2::TargetTrajectories& target) const override {
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
