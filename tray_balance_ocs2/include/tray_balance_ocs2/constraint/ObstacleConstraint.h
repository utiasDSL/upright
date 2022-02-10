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
#include <tray_balance_ocs2/constraint/CollisionAvoidanceConstraint.h>
#include <tray_balance_ocs2/definitions.h>

namespace ocs2 {
namespace mobile_manipulator {

struct DynamicObstacleSettings {
    bool enabled = false;
    std::vector<std::string> collision_link_names;
    std::vector<scalar_t> collision_sphere_radii;
    std::vector<CollisionSphere<scalar_t>> collision_spheres;
    scalar_t obstacle_radius = 0.1;
    scalar_t mu = 1e-3;
    scalar_t delta = 1e-3;

    std::vector<std::string> get_collision_frame_names() const {
        std::vector<std::string> frame_names;
        for (const auto& sphere : collision_spheres) {
            frame_names.push_back(sphere.name);
        }
        return frame_names;
    }
};

class DynamicObstacleConstraint final : public StateConstraint {
   public:
    using vector3_t = Eigen::Matrix<scalar_t, 3, 1>;
    using quaternion_t = Eigen::Quaternion<scalar_t>;

    DynamicObstacleConstraint(
        const EndEffectorKinematics<scalar_t>& endEffectorKinematics,
        const ReferenceManager& referenceManager,
        const DynamicObstacleSettings& settings)
        : StateConstraint(ConstraintOrder::Linear),
          endEffectorKinematicsPtr_(endEffectorKinematics.clone()),
          referenceManagerPtr_(&referenceManager),
          settings_(settings) {
        // if (endEffectorKinematics.getIds().size() !=
        //     settings_.collision_sphere_radii.size()) {
        //     throw std::runtime_error(
        //         "[DynamicObstacleConstraint] Number of collision sphere radii
        //         " "must match number of end effector IDs.");
        // }
        pinocchioEEKinPtr_ = dynamic_cast<PinocchioEndEffectorKinematics*>(
            endEffectorKinematicsPtr_.get());
    }

    ~DynamicObstacleConstraint() override = default;

    DynamicObstacleConstraint* clone() const override {
        return new DynamicObstacleConstraint(*endEffectorKinematicsPtr_,
                                             *referenceManagerPtr_, settings_);
    }

    size_t getNumConstraints(scalar_t time) const override {
        return settings_.collision_spheres.size();
    }

    vector_t getValue(scalar_t time, const vector_t& state,
                      const PreComputation& preComputation) const override {
        const auto& targetTrajectories =
            referenceManagerPtr_->getTargetTrajectories();
        vector3_t r_ow_w =
            interpolate_obstacle_position(time, targetTrajectories);

        std::vector<vector3_t> frame_positions =
            endEffectorKinematicsPtr_->getPosition(state);
        // std::vector<quaternion_t> frame_orientations =
        //     endEffectorKinematicsPtr_->getOrientation(state);

        vector_t constraints(getNumConstraints(time));
        for (int i = 0; i < frame_positions.size(); ++i) {
            vector3_t r_sw_w = frame_positions[i];
            // quaternion_t Q_wf = frame_orientations[i];
            // pinocchio::SE3 T_wf(Q_wf, r_fw_w);

            // vector3_t r_sf_f = settings_.collision_spheres[i].offset;
            // vector3_t r_sw_w = T_wf * r_sf_f;

            vector3_t r_so_w = r_sw_w - r_ow_w;
            scalar_t r = settings_.collision_spheres[i].radius +
                         settings_.obstacle_radius;
            constraints(i) = r_so_w.norm() - r;
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

    DynamicObstacleSettings settings_;
};  // class DynamicObstacleConstraint

}  // namespace mobile_manipulator
}  // namespace ocs2
