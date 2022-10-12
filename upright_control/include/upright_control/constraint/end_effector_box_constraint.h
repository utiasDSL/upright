#pragma once

#include <ocs2_core/constraint/StateConstraint.h>
#include <ocs2_oc/synchronized_module/ReferenceManager.h>
#include <ocs2_robotic_tools/end_effector/EndEffectorKinematics.h>

#include <upright_control/reference_trajectory.h>
#include <upright_control/types.h>

namespace upright {

class EndEffectorBoxConstraint final : public ocs2::StateConstraint {
   public:
    EndEffectorBoxConstraint(const VecXd& xyz_lower, const VecXd& xyz_upper,
                             const ocs2::EndEffectorKinematics<ocs2::scalar_t>&
                                 end_effector_kinematics,
                             const ocs2::ReferenceManager& reference_manager)
        : ocs2::StateConstraint(ocs2::ConstraintOrder::Linear),
          xyz_lower_(xyz_lower),
          xyz_upper_(xyz_upper),
          end_effector_kinematics_ptr_(end_effector_kinematics.clone()),
          reference_manager_ptr_(&reference_manager) {
        if (end_effector_kinematics.getIds().size() != 1) {
            throw std::runtime_error(
                "[EndEffectorBoxConstraint] end_effector_kinematics has wrong "
                "number of end effector IDs.");
        }
        if (xyz_lower.rows() != 3) {
            throw std::runtime_error(
                "[EndEffectorBoxConstraint] lower bound must be of length 3.");
        }
        if (xyz_upper.rows() != 3) {
            throw std::runtime_error(
                "[EndEffectorBoxConstraint] upper bound must be of length 3.");
        }
    }

    ~EndEffectorBoxConstraint() override = default;

    EndEffectorBoxConstraint* clone() const override {
        return new EndEffectorBoxConstraint(xyz_lower_, xyz_upper_,
                                            *end_effector_kinematics_ptr_,
                                            *reference_manager_ptr_);
    }

    size_t getNumConstraints(ocs2::scalar_t time) const override { return 6; }

    VecXd getValue(ocs2::scalar_t time, const VecXd& state,
                   const ocs2::PreComputation&) const override {
        const auto& target = reference_manager_ptr_->getTargetTrajectories();
        const auto desired_pose = interpolateEndEffectorPose(time, target);
        Vec3d desired_position = desired_pose.first;
        Vec3d actual_position =
            end_effector_kinematics_ptr_->getPosition(state).front();

        VecXd value = VecXd::Zero(6);
        value.head<3>() = desired_position + xyz_upper_ - actual_position;
        value.tail<3>() = actual_position - (desired_position + xyz_lower_);
        return value;
    }

    ocs2::VectorFunctionLinearApproximation getLinearApproximation(
        ocs2::scalar_t time, const VecXd& state,
        const ocs2::PreComputation& pre_comp) const override {
        // input is not used in this state cost, so we give it a dimension of
        // zero.
        auto approximation =
            ocs2::VectorFunctionLinearApproximation(6, state.rows(), 0);
        approximation.setZero(6, state.rows(), 0);

        approximation.f = getValue(time, state, pre_comp);

        const auto position_approx =
            end_effector_kinematics_ptr_->getPositionLinearApproximation(state)
                .front();
        approximation.dfdx << -position_approx.dfdx, position_approx.dfdx;

        return approximation;
    }

   private:
    EndEffectorBoxConstraint(const EndEffectorBoxConstraint& other) = default;

    // bounds
    VecXd xyz_lower_;
    VecXd xyz_upper_;

    std::unique_ptr<ocs2::EndEffectorKinematics<ocs2::scalar_t>>
        end_effector_kinematics_ptr_;
    const ocs2::ReferenceManager* reference_manager_ptr_;
};

}  // namespace upright
