#pragma once

#include <ocs2_core/constraint/StateConstraint.h>
#include <ocs2_core/constraint/StateInputConstraint.h>

#include <upright_control/types.h>

namespace upright {

// Kludge to make a state-only constraint into a state input constraint and
// thus work as a hard inequality constraint.
class StateToStateInputConstraint final : public ocs2::StateInputConstraint {
   public:
    StateToStateInputConstraint(const ocs2::StateConstraint& constraint)
        : ocs2::StateInputConstraint(ocs2::ConstraintOrder::Linear),
          constraint_ptr_(constraint.clone()) {}

    ~StateToStateInputConstraint() override = default;

    StateToStateInputConstraint(const StateToStateInputConstraint& other) = default;

    StateToStateInputConstraint* clone() const override {
        return new StateToStateInputConstraint(*constraint_ptr_);
    }

    size_t getNumConstraints(ocs2::scalar_t time) const override {
        return constraint_ptr_->getNumConstraints(time);
    }

    VecXd getValue(ocs2::scalar_t time, const VecXd& state, const VecXd& input,
                   const ocs2::PreComputation& preComp) const override {
        return constraint_ptr_->getValue(time, state, preComp);
    }

    ocs2::VectorFunctionLinearApproximation getLinearApproximation(
        ocs2::scalar_t time, const VecXd& state, const VecXd& input,
        const ocs2::PreComputation& preComp) const override {
        ocs2::VectorFunctionLinearApproximation approx = constraint_ptr_->getLinearApproximation(time, state, preComp);
        approx.dfdu.setZero(approx.dfdx.rows(), input.rows());
        return approx;
    }

   private:
    // Underlying state constraint
    std::unique_ptr<ocs2::StateConstraint> constraint_ptr_;
};

}  // namespace upright
