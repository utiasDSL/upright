#pragma once

#include <ocs2_core/cost/QuadraticStateCost.h>
#include <ocs2_core/cost/QuadraticStateInputCost.h>
#include <ocs2_core/reference/TargetTrajectories.h>

namespace upright {

class QuadraticJointStateInputCost final
    : public ocs2::QuadraticStateInputCost {
   public:
    explicit QuadraticJointStateInputCost(const MatXd& Q, const MatXd& R)
        : ocs2::QuadraticStateInputCost(Q, R) {
        xd_ = VecXd::Zero(Q.rows());
    }

    explicit QuadraticJointStateInputCost(const MatXd& Q, const MatXd& R,
                                          const VecXd& xd)
        : ocs2::QuadraticStateInputCost(Q, R), xd_(xd) {}

    ~QuadraticJointStateInputCost() override = default;

    QuadraticJointStateInputCost* clone() const override {
        return new QuadraticJointStateInputCost(*this);
    }

    std::pair<VecXd, VecXd> getStateInputDeviation(
        ocs2::scalar_t time, const VecXd& state, const VecXd& input,
        const ocs2::TargetTrajectories& targetTrajectories) const override {
        return {state - xd_, input};
    }

    VecXd xd_;
};

class QuadraticJointStateCost final : public ocs2::QuadraticStateCost {
   public:
    explicit QuadraticJointStateCost(const MatXd& Q)
        : ocs2::QuadraticStateCost(Q) {}

    ~QuadraticJointStateCost() override = default;

    QuadraticJointStateCost* clone() const override {
        return new QuadraticJointStateCost(*this);
    }

    VecXd getStateDeviation(ocs2::scalar_t time, const VecXd& state,
                            const ocs2::TargetTrajectories&) const override {
        return state;
    }
};

}  // namespace upright
