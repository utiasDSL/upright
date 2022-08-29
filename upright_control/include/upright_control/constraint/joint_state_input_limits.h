/******************************************************************************
Copyright (c) 2020, Farbod Farshidian. All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

 * Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

 * Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

 * Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
******************************************************************************/

#pragma once

#include <ocs2_core/constraint/StateInputConstraint.h>

#include <upright_control/dynamics/dimensions.h>
#include <upright_control/types.h>

namespace upright {

class JointStateInputLimits final : public ocs2::StateInputConstraint {
   public:
    JointStateInputLimits(const RobotDimensions& dims)
        : ocs2::StateInputConstraint(ocs2::ConstraintOrder::Linear),
          dims_(dims) {}

    ~JointStateInputLimits() override = default;

    JointStateInputLimits* clone() const override {
        return new JointStateInputLimits(*this);
    }

    size_t getNumConstraints(ocs2::scalar_t time) const override {
        return dims_.x + dims_.u;
    }

    VecXd getValue(ocs2::scalar_t time, const VecXd& state, const VecXd& input,
                   const ocs2::PreComputation&) const override {
        VecXd value(getNumConstraints(time));
        value << state, input.head(dims_.u);
        return value;
    }

    ocs2::VectorFunctionLinearApproximation getLinearApproximation(
        ocs2::scalar_t time, const VecXd& state, const VecXd& input,
        const ocs2::PreComputation& precomp) const override {
        ocs2::VectorFunctionLinearApproximation limits(
            getNumConstraints(time), state.rows(), input.rows());

        limits.f = getValue(time, state, input, precomp);
        limits.dfdx.setZero();
        limits.dfdx.topRows(state.rows()).setIdentity();
        limits.dfdu.setZero();
        limits.dfdu.bottomLeftCorner(dims_.u, dims_.u).setIdentity();

        return limits;
    }

   private:
    JointStateInputLimits(const JointStateInputLimits& other) = default;

    RobotDimensions dims_;
};

// For hard inequalities. TODO: these should be replaced with box constraints
// eventually.
class JointStateInputConstraint final : public ocs2::StateInputConstraint {
   public:
    JointStateInputConstraint(const RobotDimensions& dims,
                              const VecXd& state_limit_lower,
                              const VecXd& state_limit_upper,
                              const VecXd& input_limit_lower,
                              const VecXd& input_limit_upper)
        : ocs2::StateInputConstraint(ocs2::ConstraintOrder::Linear),
          dims_(dims) {
        size_t n = 2 * (dims.x + dims.u);
        MatXd Ix = MatXd::Identity(dims.x, dims.x);
        MatXd Iu = MatXd::Identity(dims.u, dims.u);

        C_ = MatXd::Zero(n, dims.x);
        C_.topRows(dims.x) = Ix;
        C_.middleRows(dims.x, dims.x) = -Ix;

        D_ = MatXd::Zero(n, dims.ou());
        D_.block(2 * dims.x, 0, dims.u, dims.u) = Iu;
        D_.bottomLeftCorner(dims.u, dims.u) = -Iu;

        e_ = VecXd::Zero(n);
        e_ << -state_limit_lower, state_limit_upper, -input_limit_lower,
            input_limit_upper;
    }

    ~JointStateInputConstraint() override = default;

    JointStateInputConstraint* clone() const override {
        return new JointStateInputConstraint(*this);
    }

    size_t getNumConstraints(ocs2::scalar_t time) const override {
        // Limits are double-sided, hence factor of two.
        return e_.rows();
    }

    VecXd getValue(ocs2::scalar_t time, const VecXd& state, const VecXd& input,
                   const ocs2::PreComputation&) const override {
        VecXd g = e_;
        g.noalias() += C_ * state;
        g.noalias() += D_ * input;
        return g;
    }

    ocs2::VectorFunctionLinearApproximation getLinearApproximation(
        ocs2::scalar_t time, const VecXd& state, const VecXd& input,
        const ocs2::PreComputation&) const {
        ocs2::VectorFunctionLinearApproximation g;
        g.f = e_;
        g.f.noalias() += C_ * state;
        g.f.noalias() += D_ * input;
        g.dfdx = C_;
        g.dfdu = D_;
        return g;
    }

   private:
    JointStateInputConstraint(const JointStateInputConstraint& other) = default;

    RobotDimensions dims_;
    MatXd C_;
    MatXd D_;
    VecXd e_;
};

}  // namespace upright
