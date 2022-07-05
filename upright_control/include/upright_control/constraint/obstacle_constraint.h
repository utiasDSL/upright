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

#include <upright_control/reference_trajectory.h>
#include <upright_control/types.h>

namespace upright {

template <typename Scalar>
struct CollisionSphere {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    // Empty constructor required for binding as opaque vector type.
    CollisionSphere() {
        name = "";
        parent_frame_name = "";
        offset = Vec3<Scalar>::Zero();
        radius = 0;
    }

    CollisionSphere(const std::string& name,
                    const std::string& parent_frame_name,
                    const Vec3<Scalar>& offset, const Scalar radius)
        : name(name),
          parent_frame_name(parent_frame_name),
          offset(offset),
          radius(radius) {}

    // Name of this collision sphere.
    std::string name;

    // Name of the robot joint this collision sphere is attached to.
    std::string parent_frame_name;

    // Offset from that joint (in the joint's frame).
    Vec3<Scalar> offset;

    // Radius of this collision sphere.
    Scalar radius;
};

struct StaticObstacleSettings {
    bool enabled = false;

    // List of pairs of collision objects to check
    std::vector<std::pair<std::string, std::string>> collision_link_pairs;

    // Minimum distance allowed between collision objects
    ocs2::scalar_t minimum_distance = 0;

    // Relaxed barrier function parameters
    ocs2::scalar_t mu = 1e-2;
    ocs2::scalar_t delta = 1e-3;

    // Extra collision spheres to attach to the robot body for collision
    // avoidance.
    std::vector<CollisionSphere<ocs2::scalar_t>> extra_spheres;
};

struct DynamicObstacleSettings {
    bool enabled = false;
    std::vector<CollisionSphere<ocs2::scalar_t>> collision_spheres;
    ocs2::scalar_t obstacle_radius = 0.1;
    ocs2::scalar_t mu = 1e-3;
    ocs2::scalar_t delta = 1e-3;

    std::vector<std::string> get_collision_frame_names() const {
        std::vector<std::string> frame_names;
        for (const auto& sphere : collision_spheres) {
            frame_names.push_back(sphere.name);
        }
        return frame_names;
    }
};

class DynamicObstacleConstraint final : public ocs2::StateConstraint {
   public:
    DynamicObstacleConstraint(const ocs2::EndEffectorKinematics<ocs2::scalar_t>&
                                  endEffectorKinematics,
                              const ocs2::ReferenceManager& referenceManager,
                              const DynamicObstacleSettings& settings)
        : StateConstraint(ocs2::ConstraintOrder::Linear),
          endEffectorKinematicsPtr_(endEffectorKinematics.clone()),
          referenceManagerPtr_(&referenceManager),
          settings_(settings) {
        // if (endEffectorKinematics.getIds().size() !=
        //     settings_.collision_sphere_radii.size()) {
        //     throw std::runtime_error(
        //         "[DynamicObstacleConstraint] Number of collision sphere radii
        //         " "must match number of end effector IDs.");
        // }
        pinocchioEEKinPtr_ =
            dynamic_cast<ocs2::PinocchioEndEffectorKinematics*>(
                endEffectorKinematicsPtr_.get());
    }

    ~DynamicObstacleConstraint() override = default;

    DynamicObstacleConstraint* clone() const override {
        return new DynamicObstacleConstraint(*endEffectorKinematicsPtr_,
                                             *referenceManagerPtr_, settings_);
    }

    size_t getNumConstraints(ocs2::scalar_t time) const override {
        return settings_.collision_spheres.size() + 1;
    }

    VecXd getValue(ocs2::scalar_t time, const VecXd& state,
                   const ocs2::PreComputation& preComputation) const override {
        const auto& targetTrajectories =
            referenceManagerPtr_->getTargetTrajectories();
        Vec3d r_ow_w = interpolate_obstacle_position(time, targetTrajectories);

        std::vector<Vec3d> frame_positions =
            endEffectorKinematicsPtr_->getPosition(state);
        // std::vector<Quatd> frame_orientations =
        //     endEffectorKinematicsPtr_->getOrientation(state);

        VecXd constraints(getNumConstraints(time));
        for (int i = 0; i < frame_positions.size(); ++i) {
            Vec3d r_sw_w = frame_positions[i];
            // Quatd Q_wf = frame_orientations[i];
            // pinocchio::SE3 T_wf(Q_wf, r_fw_w);

            // Vec3d r_sf_f = settings_.collision_spheres[i].offset;
            // Vec3d r_sw_w = T_wf * r_sf_f;

            Vec3d r_so_w = r_sw_w - r_ow_w;
            ocs2::scalar_t r = settings_.collision_spheres[i].radius +
                               settings_.obstacle_radius;
            // Controller seems to require less compute when use this smooth
            // version of the constraint
            constraints(i) = r_so_w.dot(r_so_w) - r * r;
            // constraints(i) = r_so_w.norm() - r;
        }

        // Extra hard-coded hack to do self-collision avoidance between
        // balanced objects and the forearm.
        Vec3d vec = frame_positions[4] - frame_positions[1];
        ocs2::scalar_t r = settings_.collision_spheres[1].radius +
                           settings_.collision_spheres[4].radius;
        constraints(frame_positions.size()) = vec.dot(vec) - r * r;

        return constraints;
    }

    ocs2::VectorFunctionLinearApproximation getLinearApproximation(
        ocs2::scalar_t time, const VecXd& state,
        const ocs2::PreComputation& preComputation) const override {
        auto approximation = ocs2::VectorFunctionLinearApproximation(
            getNumConstraints(time), state.rows(), 0);
        approximation.setZero(getNumConstraints(time), state.rows(), 0);

        const auto& targetTrajectories =
            referenceManagerPtr_->getTargetTrajectories();
        Vec3d obstacle_pos =
            interpolate_obstacle_position(time, targetTrajectories);

        // the .f part is just the value
        approximation.f = getValue(time, state, preComputation);

        const auto ee_positions =
            endEffectorKinematicsPtr_->getPositionLinearApproximation(state);

        for (int i = 0; i < ee_positions.size(); ++i) {
            Vec3d vec = ee_positions[i].f - obstacle_pos;
            approximation.dfdx.row(i) =
                2 * vec.transpose() * ee_positions[i].dfdx;
            // approximation.dfdx.row(i) =
            //     vec.transpose() * ee_positions[i].dfdx / vec.norm();
        }

        Vec3d vec = ee_positions[4].f - ee_positions[1].f;
        approximation.dfdx.row(ee_positions.size()) =
            2 * vec.transpose() * ee_positions[4].dfdx -
            2 * vec.transpose() * ee_positions[1].dfdx;

        return approximation;
    }

   private:
    DynamicObstacleConstraint(const DynamicObstacleConstraint& other) = default;

    /** Cached pointer to the pinocchio end effector kinematics. Is set to
     * nullptr if not used. */
    ocs2::PinocchioEndEffectorKinematics* pinocchioEEKinPtr_ = nullptr;

    Vec3d eeDesiredPosition_;
    Quatd eeDesiredOrientation_;
    std::unique_ptr<ocs2::EndEffectorKinematics<ocs2::scalar_t>>
        endEffectorKinematicsPtr_;
    const ocs2::ReferenceManager* referenceManagerPtr_;

    DynamicObstacleSettings settings_;
};  // class DynamicObstacleConstraint

}  // namespace upright
