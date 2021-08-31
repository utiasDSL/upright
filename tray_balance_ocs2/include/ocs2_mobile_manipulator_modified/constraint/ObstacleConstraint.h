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

#include <ocs2_core/constraint/StateConstraint.h>
#include <ocs2_oc/synchronized_module/ReferenceManager.h>
#include <ocs2_pinocchio_interface/PinocchioEndEffectorKinematics.h>
#include <ocs2_robotic_tools/end_effector/EndEffectorKinematics.h>

#include <ocs2_mobile_manipulator_modified/MobileManipulatorReferenceTrajectory.h>
#include <ocs2_mobile_manipulator_modified/definitions.h>

namespace ocs2 {
namespace mobile_manipulator {

class ObstacleConstraint final : public StateConstraint {
   public:
    using vector3_t = Eigen::Matrix<scalar_t, 3, 1>;
    using quaternion_t = Eigen::Quaternion<scalar_t>;

    ObstacleConstraint(
        const EndEffectorKinematics<scalar_t>& endEffectorKinematics,
        const ReferenceManager& referenceManager)
        : StateConstraint(ConstraintOrder::Linear),
          endEffectorKinematicsPtr_(endEffectorKinematics.clone()),
          referenceManagerPtr_(&referenceManager) {
        if (endEffectorKinematics.getIds().size() != 1) {
            throw std::runtime_error(
                "[EndEffectorConstraint] endEffectorKinematics has wrong "
                "number of end effector IDs.");
        }
        pinocchioEEKinPtr_ = dynamic_cast<PinocchioEndEffectorKinematics*>(
            endEffectorKinematicsPtr_.get());
    }

    ~ObstacleConstraint() override = default;

    ObstacleConstraint* clone() const override {
        return new ObstacleConstraint(*endEffectorKinematicsPtr_,
                                      *referenceManagerPtr_);
    }

    size_t getNumConstraints(scalar_t time) const override {
        return 1;  // TODO
    }

    vector_t getValue(scalar_t time, const vector_t& state,
                      const PreComputation& preComputation) const override {
        const auto& targetTrajectories =
            referenceManagerPtr_->getTargetTrajectories();
        vector3_t obstacle_pos =
            interpolate_obstacle_position(time, targetTrajectories);
        vector3_t ee_pos =
            endEffectorKinematicsPtr_->getPosition(state).front();
        vector3_t vec = ee_pos - obstacle_pos;

        scalar_t r_objects = 0.25;
        scalar_t r_obstacle = 0.1;
        scalar_t r_safety = 0.1;
        scalar_t r = r_objects + r_obstacle + r_safety;

        // here we are only worrying about obstacle and the objects, not any
        // other part of the robot
        vector_t constraints;
        constraints << vec.dot(vec) - r * r;
        return constraints;
    }

    VectorFunctionLinearApproximation getLinearApproximation(
        scalar_t time, const vector_t& state,
        const PreComputation& preComputation) const override {
        auto approximation = VectorFunctionLinearApproximation(
            getNumConstraints(time), state.rows(), 0);
        approximation.setZero(getNumConstraints(time), state.rows(), 0);

        const auto& targetTrajectories =
            referenceManagerPtr_->getTargetTrajectories();
        vector3_t obstacle_pos =
            interpolate_obstacle_position(time, targetTrajectories);
        const auto ee_pos =
            endEffectorKinematicsPtr_->getPositionLinearApproximation(state)
                .front();
        vector3_t vec = ee_pos.f - obstacle_pos;

        // the .f part is just the value
        approximation.f = getValue(time, state, preComputation);
        approximation.dfdx = 2 * vec.transpose() * ee_pos.dfdx;

        // approximation.f.head<3>() =
        //     eePosition.f - desiredPositionOrientation.first;
        // approximation.dfdx.topRows<3>() = eePosition.dfdx;
        //
        // const auto eeOrientationError =
        //     endEffectorKinematicsPtr_
        //         ->getOrientationErrorLinearApproximation(
        //             state, {desiredPositionOrientation.second})
        //         .front();
        // approximation.f.tail<3>() = eeOrientationError.f;
        // approximation.dfdx.bottomRows<3>() = eeOrientationError.dfdx;

        return approximation;
    }

   private:
    ObstacleConstraint(const ObstacleConstraint& other) = default;

    /** Cached pointer to the pinocchio end effector kinematics. Is set to
     * nullptr if not used. */
    PinocchioEndEffectorKinematics* pinocchioEEKinPtr_ = nullptr;

    vector3_t eeDesiredPosition_;
    quaternion_t eeDesiredOrientation_;
    std::unique_ptr<EndEffectorKinematics<scalar_t>> endEffectorKinematicsPtr_;
    const ReferenceManager* referenceManagerPtr_;
};  // class ObstacleConstraint

}  // namespace mobile_manipulator
}  // namespace ocs2
