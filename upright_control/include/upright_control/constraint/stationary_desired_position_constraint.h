#pragma once

#include <ocs2_core/constraint/StateConstraint.h>
#include <ocs2_oc/synchronized_module/ReferenceManager.h>
#include <ocs2_robotic_tools/end_effector/EndEffectorKinematics.h>
#include <upright_control/reference_trajectory.h>
#include <upright_control/types.h>

namespace upright {

class StationaryDesiredPositionConstraint final : public ocs2::StateConstraint {
   public:
    StationaryDesiredPositionConstraint(
        const ocs2::EndEffectorKinematics<ocs2::scalar_t>& kinematics,
        const ocs2::ReferenceManager& reference_manager,
        const OptimizationDimensions& dims)
        : ocs2::StateConstraint(ocs2::ConstraintOrder::Linear),
          kinematics_ptr_(kinematics.clone()),
          reference_manager_ptr_(&reference_manager),
          dims_(dims) {
        if (kinematics.getIds().size() != 1) {
            throw std::runtime_error(
                "[EndEffectorBoxConstraint] kinematics has wrong "
                "number of end effector IDs.");
        }
    }

    ~StationaryDesiredPositionConstraint() override = default;

    StationaryDesiredPositionConstraint* clone() const override {
        return new StationaryDesiredPositionConstraint(
            *kinematics_ptr_, *reference_manager_ptr_, dims_);
    }

    size_t getNumConstraints(ocs2::scalar_t time) const override {
        return 3 + 2 * dims_.robot.v;
    }

    VecXd getValue(ocs2::scalar_t time, const VecXd& state,
                   const ocs2::PreComputation&) const override {
        const auto& target = reference_manager_ptr_->getTargetTrajectories();
        const auto desired_pose = interpolate_end_effector_pose(time, target);
        Vec3d desired_position = desired_pose.first;
        Vec3d actual_position = kinematics_ptr_->getPosition(state).front();

        // Value is the position error followed by joint velocity and
        // acceleration
        VecXd value(getNumConstraints(time));
        value << desired_position - actual_position,
            state.segment(dims_.robot.q, 2 * dims_.robot.v);
        return value;
    }

    ocs2::VectorFunctionLinearApproximation getLinearApproximation(
        ocs2::scalar_t time, const VecXd& state,
        const ocs2::PreComputation& pre_comp) const override {
        // input is not used in this state constraint, so we give it a dimension
        // of zero.
        const size_t n = getNumConstraints(time);
        auto approximation =
            ocs2::VectorFunctionLinearApproximation(n, state.rows(), 0);
        approximation.setZero(n, state.rows(), 0);

        approximation.f = getValue(time, state, pre_comp);

        const auto position_approx =
            kinematics_ptr_->getPositionLinearApproximation(state).front();

        approximation.dfdx.topRows<3>() = -position_approx.dfdx;
        const size_t d = 2 * dims_.robot.v;
        approximation.dfdx.bottomRightCorner(d, d) = MatXd::Identity(d, d);

        return approximation;
    }

   private:
    StationaryDesiredPositionConstraint(
        const StationaryDesiredPositionConstraint& other) = default;

    OptimizationDimensions dims_;
    std::unique_ptr<ocs2::EndEffectorKinematics<ocs2::scalar_t>>
        kinematics_ptr_;
    const ocs2::ReferenceManager* reference_manager_ptr_;
};

}  // namespace upright
