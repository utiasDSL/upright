#pragma once

#include <ocs2_core/cost/StateCost.h>
#include <ocs2_pinocchio_interface/PinocchioEndEffectorKinematics.h>
#include <ocs2_robotic_tools/end_effector/EndEffectorKinematics.h>

#include <upright_control/reference_trajectory.h>
#include <upright_control/types.h>

namespace upright {

class EndEffectorCost final : public ocs2::StateCost {
   public:
    EndEffectorCost(const MatXd& W,
                    const ocs2::EndEffectorKinematics<ocs2::scalar_t>&
                        end_effector_kinematics)
        : W_(W), end_effector_kinematics_ptr_(end_effector_kinematics.clone()) {
        if (end_effector_kinematics.getIds().size() != 1) {
            throw std::runtime_error(
                "[EndEffectorConstraint] endEffectorKinematics has wrong "
                "number of "
                "end effector IDs.");
        }
    }

    ~EndEffectorCost() override = default;

    EndEffectorCost* clone() const override {
        return new EndEffectorCost(W_, *end_effector_kinematics_ptr_);
    }

    ocs2::scalar_t getValue(ocs2::scalar_t time, const VecXd& state,
                            const ocs2::TargetTrajectories& target,
                            const ocs2::PreComputation&) const override {
        const auto desired_pose = interpolateEndEffectorPose(time, target);

        VecXd err = VecXd::Zero(6);
        err.head<3>() =
            end_effector_kinematics_ptr_->getPosition(state).front() -
            desired_pose.first;
        err.tail<3>() = end_effector_kinematics_ptr_
                            ->getOrientationError(state, {desired_pose.second})
                            .front();

        return 0.5 * err.transpose() * W_ * err;
    }

    ocs2::ScalarFunctionQuadraticApproximation getQuadraticApproximation(
        ocs2::scalar_t time, const VecXd& state,
        const ocs2::TargetTrajectories& target,
        const ocs2::PreComputation&) const override {
        const auto desired_pose = interpolateEndEffectorPose(time, target);

        // NOTE: input is not used in this state cost, so we give it a
        // dimension of zero.
        auto approximation =
            ocs2::ScalarFunctionQuadraticApproximation(state.rows(), 0);
        approximation.setZero(state.rows(), 0);

        // Linear approximations of position and orientation error
        const auto eePosition =
            end_effector_kinematics_ptr_->getPositionLinearApproximation(state)
                .front();
        const auto eeOrientationError =
            end_effector_kinematics_ptr_
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
        end_effector_kinematics_ptr_;
};

}  // namespace upright
