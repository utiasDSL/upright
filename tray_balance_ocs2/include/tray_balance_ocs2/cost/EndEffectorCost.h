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

#include <ocs2_core/cost/StateCost.h>
#include <ocs2_oc/synchronized_module/ReferenceManager.h>
#include <ocs2_pinocchio_interface/PinocchioEndEffectorKinematics.h>
#include <ocs2_robotic_tools/end_effector/EndEffectorKinematics.h>

#include <tray_balance_ocs2/MobileManipulatorReferenceTrajectory.h>
#include <tray_balance_ocs2/types.h>

namespace upright {

class EndEffectorCost final : public ocs2::StateCost {
   public:
    EndEffectorCost(const MatXd W,  // note not reference
                    const ocs2::EndEffectorKinematics<ocs2::scalar_t>&
                        endEffectorKinematics,
                    const ocs2::ReferenceManager& referenceManager)
        : W_(std::move(W)),
          endEffectorKinematicsPtr_(endEffectorKinematics.clone()),
          referenceManagerPtr_(&referenceManager) {
        if (endEffectorKinematics.getIds().size() != 1) {
            throw std::runtime_error(
                "[EndEffectorConstraint] endEffectorKinematics has wrong "
                "number of "
                "end effector IDs.");
        }
        pinocchioEEKinPtr_ =
            dynamic_cast<ocs2::PinocchioEndEffectorKinematics*>(
                endEffectorKinematicsPtr_.get());
    }

    ~EndEffectorCost() override = default;

    EndEffectorCost* clone() const override {
        return new EndEffectorCost(W_, *endEffectorKinematicsPtr_,
                                   *referenceManagerPtr_);
    }

    ocs2::scalar_t getValue(
        ocs2::scalar_t time, const VecXd& state,
        const ocs2::TargetTrajectories& targetTrajectories,
        const ocs2::PreComputation& preComp) const override {
        const auto desiredPositionOrientation =
            interpolateEndEffectorPose(time, targetTrajectories);

        VecXd err = VecXd::Zero(6);
        err.head<3>() = endEffectorKinematicsPtr_->getPosition(state).front() -
                        desiredPositionOrientation.first;
        err.tail<3>() = endEffectorKinematicsPtr_
                            ->getOrientationError(
                                state, {desiredPositionOrientation.second})
                            .front();

        return 0.5 * err.transpose() * W_ * err;
    }

    ocs2::ScalarFunctionQuadraticApproximation getQuadraticApproximation(
        ocs2::scalar_t time, const VecXd& state,
        const ocs2::TargetTrajectories& targetTrajectories,
        const ocs2::PreComputation& preComp) const override {
        const auto desiredPositionOrientation =
            interpolateEndEffectorPose(time, targetTrajectories);

        // NOTE: input is not used in this state cost, so we give it a
        // dimension of zero.
        auto approximation =
            ocs2::ScalarFunctionQuadraticApproximation(state.rows(), 0);
        approximation.setZero(state.rows(), 0);

        // Linear approximations of position and orientation error
        const auto eePosition =
            endEffectorKinematicsPtr_->getPositionLinearApproximation(state)
                .front();
        const auto eeOrientationError =
            endEffectorKinematicsPtr_
                ->getOrientationErrorLinearApproximation(
                    state, {desiredPositionOrientation.second})
                .front();

        // Function value
        VecXd e = VecXd::Zero(6);
        e << eePosition.f - desiredPositionOrientation.first,
            eeOrientationError.f;
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

    /** Cached pointer to the pinocchio end effector kinematics. Is set to
     * nullptr if not used. */
    ocs2::PinocchioEndEffectorKinematics* pinocchioEEKinPtr_ = nullptr;

    Vec3d eeDesiredPosition_;
    Quatd eeDesiredOrientation_;
    std::unique_ptr<ocs2::EndEffectorKinematics<ocs2::scalar_t>>
        endEffectorKinematicsPtr_;
    const ocs2::ReferenceManager* referenceManagerPtr_;
};

}  // namespace upright
