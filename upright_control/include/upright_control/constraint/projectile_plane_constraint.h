#pragma once

#include <ocs2_core/constraint/StateConstraint.h>
#include <ocs2_oc/synchronized_module/ReferenceManager.h>
#include <ocs2_robotic_tools/end_effector/EndEffectorKinematics.h>
#include <upright_control/reference_trajectory.h>
#include <upright_control/types.h>

namespace upright {

class ProjectilePlaneConstraint final : public ocs2::StateConstraint {
   public:
    ProjectilePlaneConstraint(
        const ocs2::EndEffectorKinematics<ocs2::scalar_t>& kinematics,
        const ocs2::ReferenceManager& reference_manager)
        : ocs2::StateConstraint(ocs2::ConstraintOrder::Linear),
          kinematics_ptr_(kinematics.clone()),
          reference_manager_ptr_(&reference_manager) {
        if (kinematics.getIds().size() != 1) {
            throw std::runtime_error(
                "[ProjectilePlaneConstraint] kinematics has wrong "
                "number of end effector IDs.");
        }
    }

    ~ProjectilePlaneConstraint() override = default;

    ProjectilePlaneConstraint* clone() const override {
        return new ProjectilePlaneConstraint(*kinematics_ptr_,
                                             *reference_manager_ptr_);
    }

    size_t getNumConstraints(ocs2::scalar_t time) const override { return 1; }

    VecXd getValue(ocs2::scalar_t time, const VecXd& state,
                   const ocs2::PreComputation&) const override {
        const auto& target = reference_manager_ptr_->getTargetTrajectories();
        VecXd xd = target.stateTrajectory[0];
        // TODO could be update this with state information over time?
        // VecXd p = xd.segment(7, 3);

        ocs2::scalar_t s = xd(7);
        VecXd n = xd.segment(8, 3);

        Vec3d r_ew_w = kinematics_ptr_->getPosition(state).front();
        Vec3d r_obs = state.tail(9).head(3);

        ocs2::scalar_t w = 0.3;
        ocs2::scalar_t d = n.dot(r_ew_w - r_obs);
        VecXd constraint(1);
        constraint(0) = s * (d - w);  // distance is at least w
        return constraint;
    }

    ocs2::VectorFunctionLinearApproximation getLinearApproximation(
        ocs2::scalar_t time, const VecXd& state,
        const ocs2::PreComputation& pre_comp) const override {
        const auto& target = reference_manager_ptr_->getTargetTrajectories();

        VecXd xd = target.stateTrajectory[0];
        // VecXd p = xd.segment(7, 3);
        // VecXd n = xd.segment(10, 3);
        //
        ocs2::scalar_t s = xd(7);
        VecXd n = xd.segment(8, 3);

        // input is not used in this state cost, so we give it a dimension of
        // zero.
        auto approximation =
            ocs2::VectorFunctionLinearApproximation(1, state.rows(), 0);
        approximation.setZero(1, state.rows(), 0);

        approximation.f = getValue(time, state, pre_comp);

        // TODO we are now also dependendent on x via r_obs
        MatXd dr_obs_dx = MatXd::Zero(3, state.size());
        dr_obs_dx.block(0, state.size() - 9, 3, 3) = Mat3d::Identity();

        const auto position_approx =
            kinematics_ptr_->getPositionLinearApproximation(state).front();
        approximation.dfdx << s * n.transpose() * (position_approx.dfdx - dr_obs_dx);

        return approximation;
    }

   private:
    ProjectilePlaneConstraint(const ProjectilePlaneConstraint& other) = default;

    std::unique_ptr<ocs2::EndEffectorKinematics<ocs2::scalar_t>>
        kinematics_ptr_;
    const ocs2::ReferenceManager* reference_manager_ptr_;
};

}  // namespace upright
