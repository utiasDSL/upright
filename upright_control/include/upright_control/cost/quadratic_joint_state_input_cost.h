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

#include <ocs2_core/cost/QuadraticStateCost.h>
#include <ocs2_core/cost/QuadraticStateInputCost.h>
#include <ocs2_core/reference/TargetTrajectories.h>

namespace upright {

class QuadraticJointStateInputCost final
    : public ocs2::QuadraticStateInputCost {
   public:
    explicit QuadraticJointStateInputCost(const MatXd& Q, const MatXd& R)
        : ocs2::QuadraticStateInputCost(Q, R) {}
    ~QuadraticJointStateInputCost() override = default;

    QuadraticJointStateInputCost* clone() const override {
        return new QuadraticJointStateInputCost(*this);
    }

    std::pair<VecXd, VecXd> getStateInputDeviation(
        ocs2::scalar_t time, const VecXd& state, const VecXd& input,
        const ocs2::TargetTrajectories& targetTrajectories) const override {
        return {state, input};
    }
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
