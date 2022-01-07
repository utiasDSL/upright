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

#include <tray_balance_ocs2/MobileManipulatorReferenceTrajectory.h>
#include <tray_balance_ocs2/definitions.h>

namespace ocs2 {
namespace mobile_manipulator {

class DynamicObstacleConstraint final : public StateConstraint {
   public:
    using vector3_t = Eigen::Matrix<scalar_t, 3, 1>;
    using quaternion_t = Eigen::Quaternion<scalar_t>;

    DynamicObstacleConstraint(
        const EndEffectorKinematics<scalar_t>& endEffectorKinematics,
        const ReferenceManager& referenceManager,
        const std::vector<scalar_t>& collision_sphere_radii,
        const scalar_t obstacle_radius)
        : StateConstraint(ConstraintOrder::Linear),
          endEffectorKinematicsPtr_(endEffectorKinematics.clone()),
          referenceManagerPtr_(&referenceManager),
          collision_sphere_radii_(collision_sphere_radii),
          obstacle_radius_(obstacle_radius) {
        if (endEffectorKinematics.getIds().size() !=
            collision_sphere_radii.size()) {
            throw std::runtime_error(
                "[DynamicObstacleConstraint] Number of collision sphere radii "
                "must match number of end effector IDs.");
        }
        pinocchioEEKinPtr_ = dynamic_cast<PinocchioEndEffectorKinematics*>(
            endEffectorKinematicsPtr_.get());
    }

    ~DynamicObstacleConstraint() override = default;

    DynamicObstacleConstraint* clone() const override {
        return new DynamicObstacleConstraint(
            *endEffectorKinematicsPtr_, *referenceManagerPtr_,
            collision_sphere_radii_, obstacle_radius_);
    }

    size_t getNumConstraints(scalar_t time) const override {
        return collision_sphere_radii_.size();
    }

    vector_t getValue(scalar_t time, const vector_t& state,
                      const PreComputation& preComputation) const override {
        const auto& targetTrajectories =
            referenceManagerPtr_->getTargetTrajectories();
        vector3_t obstacle_pos =
            interpolate_obstacle_position(time, targetTrajectories);

        std::vector<vector3_t> ee_positions =
            endEffectorKinematicsPtr_->getPosition(state);

        vector_t constraints(getNumConstraints(time));
        for (int i = 0; i < ee_positions.size(); ++i) {
            vector3_t vec = ee_positions[i] - obstacle_pos;
            scalar_t r = collision_sphere_radii_[i] + obstacle_radius_;
            constraints(i) = vec.norm() - r;
        }
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

        // the .f part is just the value
        approximation.f = getValue(time, state, preComputation);

        const auto ee_positions =
            endEffectorKinematicsPtr_->getPositionLinearApproximation(state);

        for (int i = 0; i < ee_positions.size(); ++i) {
            vector3_t vec = ee_positions[i].f - obstacle_pos;
            approximation.dfdx.row(i) =
                vec.transpose() * ee_positions[i].dfdx / vec.norm();
        }

        return approximation;
    }

   private:
    DynamicObstacleConstraint(const DynamicObstacleConstraint& other) = default;

    /** Cached pointer to the pinocchio end effector kinematics. Is set to
     * nullptr if not used. */
    PinocchioEndEffectorKinematics* pinocchioEEKinPtr_ = nullptr;

    vector3_t eeDesiredPosition_;
    quaternion_t eeDesiredOrientation_;
    std::unique_ptr<EndEffectorKinematics<scalar_t>> endEffectorKinematicsPtr_;
    const ReferenceManager* referenceManagerPtr_;

    // Radii of collision spheres
    std::vector<scalar_t> collision_sphere_radii_;
    scalar_t obstacle_radius_;  // Radius of obstacle
};                              // class DynamicObstacleConstraint

}  // namespace mobile_manipulator
}  // namespace ocs2
