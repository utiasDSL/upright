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

#include <upright_control/dimensions.h>
#include <upright_control/types.h>

namespace upright {

class JointStateInputLimits final : public ocs2::StateInputConstraint {
   public:
    JointStateInputLimits(const OptimizationDimensions& dims)
        : ocs2::StateInputConstraint(ocs2::ConstraintOrder::Linear),
          dims_(dims) {}

    ~JointStateInputLimits() override = default;

    JointStateInputLimits* clone() const override {
        return new JointStateInputLimits(*this);
    }

    size_t getNumConstraints(ocs2::scalar_t time) const override {
        return dims_.robot.x + dims_.robot.u;
    }

    // Suppose we have x = [x_1, x_2] and u = [u_1, u_2] and our function value
    // is thus f = [x_1, u_1] (i.e., the part of the state and input we want to
    // limit)
    VecXd getValue(ocs2::scalar_t time, const VecXd& state, const VecXd& input,
                   const ocs2::PreComputation&) const override {
        VecXd value(getNumConstraints(time));
        value << state.head(dims_.robot.x), input.head(dims_.robot.u);
        return value;
    }

    // Following from the above, we have
    //   df/dx = [df/dx_1 df/dx_2] = |dx_1/dx_1 dx_1/dx_2| = |I 0|
    //                               |du_1/dx_1 du_1/dx_2|   |0 0|
    // and
    //   df/du = [df/du_1 df/du_2] = |dx_1/du_1 dx_1/du_2| = |0 0|
    //                               |du_1/du_1 du_1/du_2|   |I 0|
    ocs2::VectorFunctionLinearApproximation getLinearApproximation(
        ocs2::scalar_t time, const VecXd& state, const VecXd& input,
        const ocs2::PreComputation& precomp) const override {
        ocs2::VectorFunctionLinearApproximation limits(
            getNumConstraints(time), state.rows(), input.rows());

        limits.f = getValue(time, state, input, precomp);
        limits.dfdx.setZero();
        limits.dfdx.topLeftCorner(dims_.robot.x, dims_.robot.x).setIdentity();
        limits.dfdu.setZero();
        limits.dfdu.bottomLeftCorner(dims_.robot.u, dims_.robot.u).setIdentity();

        return limits;
    }

   private:
    JointStateInputLimits(const JointStateInputLimits& other) = default;

    OptimizationDimensions dims_;
};

// For hard inequalities. TODO: these should be replaced with box constraints
// eventually.
class JointStateInputConstraint final : public ocs2::StateInputConstraint {
   public:
    JointStateInputConstraint(const OptimizationDimensions& dims,
                              const VecXd& state_limit_lower,
                              const VecXd& state_limit_upper,
                              const VecXd& input_limit_lower,
                              const VecXd& input_limit_upper)
        : ocs2::StateInputConstraint(ocs2::ConstraintOrder::Linear),
          dims_(dims) {
        size_t rx = dims.robot.x;
        size_t ru = dims.robot.u;

        // f = C * x + D * u + e >= 0
        //   = | I 0| * x + | 0 0| * u
        //     |-I 0|       | 0 0|
        //     | 0 0|       | I 0|
        //     | 0 0|       |-I 0|
        size_t n = 2 * (rx + ru);
        MatXd Ix = MatXd::Identity(rx, rx);
        MatXd Iu = MatXd::Identity(ru, ru);

        C_ = MatXd::Zero(n, dims.x());
        C_.topLeftCorner(rx, rx) = Ix;
        C_.block(rx, 0, rx, rx) = -Ix;

        D_ = MatXd::Zero(n, dims.u());
        D_.block(2 * rx, 0, ru, ru) = Iu;
        D_.bottomLeftCorner(ru, ru) = -Iu;

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
        VecXd f = C_ * state + D_ * input + e_;
        return f;
    }

    ocs2::VectorFunctionLinearApproximation getLinearApproximation(
        ocs2::scalar_t time, const VecXd& state, const VecXd& input,
        const ocs2::PreComputation&) const {
        ocs2::VectorFunctionLinearApproximation g;
        g.f = C_ * state + D_ * input + e_;
        g.dfdx = C_;
        g.dfdu = D_;
        return g;
    }

   private:
    JointStateInputConstraint(const JointStateInputConstraint& other) = default;

    OptimizationDimensions dims_;
    MatXd C_;
    MatXd D_;
    VecXd e_;
};

}  // namespace upright
