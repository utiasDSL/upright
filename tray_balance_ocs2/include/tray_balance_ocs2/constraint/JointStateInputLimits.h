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

#include <memory>

#include <ocs2_core/constraint/StateInputConstraint.h>

#include <tray_balance_ocs2/dynamics/Dimensions.h>


namespace ocs2 {
namespace mobile_manipulator {

class JointStateInputLimits final : public StateInputConstraint {
   public:
    JointStateInputLimits(const RobotDimensions& dims)
        : StateInputConstraint(ConstraintOrder::Linear), dims_(dims) {}

    ~JointStateInputLimits() override = default;

    JointStateInputLimits* clone() const override {
        return new JointStateInputLimits(*this);
    }

    size_t getNumConstraints(scalar_t time) const override {
        return dims_.x + dims_.u;
    }

    vector_t getValue(scalar_t time, const vector_t& state,
                      const vector_t& input,
                      const PreComputation&) const override {
        vector_t value(getNumConstraints(time));
        value << state, input;
        return value;
    }

    VectorFunctionLinearApproximation getLinearApproximation(
        scalar_t time, const vector_t& state, const vector_t& input,
        const PreComputation& precomp) const override {
        VectorFunctionLinearApproximation limits(getNumConstraints(time),
                                                 state.rows(), input.rows());
        limits.f = getValue(time, state, input, precomp);
        limits.dfdx.setZero();
        limits.dfdx.topRows(state.rows()).setIdentity();
        limits.dfdu.setZero();
        limits.dfdu.bottomRows(input.rows()).setIdentity();

        // std::cout << "limits.dfdx = " << limits.dfdx << std::endl;
        // std::cout << "limits.dfdu = " << limits.dfdu << std::endl;

        return limits;
    }

   private:
    JointStateInputLimits(const JointStateInputLimits& other) = default;

    RobotDimensions dims_;
};

}  // namespace mobile_manipulator
}  // namespace ocs2
