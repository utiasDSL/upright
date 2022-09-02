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

#include <string>

#include <pinocchio/parsers/urdf.hpp>

#include <ocs2_core/Types.h>
#include <ocs2_core/initialization/Initializer.h>
#include <ocs2_ddp/DDP_Settings.h>
#include <ocs2_mpc/MPC_BASE.h>
#include <ocs2_oc/oc_problem/OptimalControlProblem.h>
#include <ocs2_oc/rollout/RolloutBase.h>
#include <ocs2_oc/synchronized_module/ReferenceManager.h>
#include <ocs2_pinocchio_interface/PinocchioEndEffectorKinematicsCppAd.h>
#include <ocs2_pinocchio_interface/PinocchioInterface.h>
#include <ocs2_robotic_tools/common/RobotInterface.h>
#include <ocs2_sqp/MultipleShootingSettings.h>

#include <upright_control/constraint/obstacle_constraint.h>
#include <upright_control/controller_settings.h>
#include <upright_control/types.h>

namespace upright {

/**
 * Mobile Manipulator Robot Interface class
 */
class ControllerInterface final : public ocs2::RobotInterface {
   public:
    /**
     * Constructor
     */
    explicit ControllerInterface(const ControllerSettings& settings);

    const VecXd& getInitialState() { return initialState_; }

    ocs2::ddp::Settings& ddpSettings() { return ddpSettings_; }

    ocs2::mpc::Settings& mpcSettings() { return mpcSettings_; }

    std::unique_ptr<ocs2::MPC_BASE> getMpc();

    const ocs2::OptimalControlProblem& getOptimalControlProblem()
        const override {
        return problem_;
    }

    const ocs2::Initializer& getInitializer() const override {
        return *initializerPtr_;
    }

    std::shared_ptr<ocs2::ReferenceManagerInterface> getReferenceManagerPtr()
        const override {
        return referenceManagerPtr_;
    }

    const ocs2::RolloutBase& getRollout() const { return *rolloutPtr_; }

    const ocs2::PinocchioInterface& getPinocchioInterface() const {
        return *pinocchioInterfacePtr_;
    }

    ocs2::PinocchioInterface build_pinocchio_interface(
        const std::string& urdfPath);

    static pinocchio::GeometryModel build_geometry_model(
        const std::string& urdf_path);

   private:
    std::unique_ptr<ocs2::StateInputCost> getQuadraticStateInputCost(
        const std::string& taskFile);

    std::unique_ptr<ocs2::StateCost> getEndEffectorCost(
        const ocs2::PinocchioEndEffectorKinematicsCppAd&
            end_effector_kinematics);

    // Hard static obstacle avoidance constraint.
    std::unique_ptr<ocs2::StateConstraint> get_static_obstacle_constraint(
        ocs2::PinocchioInterface& pinocchioInterface,
        const StaticObstacleSettings& settings,
        const std::string& obstacle_urdf_path, bool useCaching,
        const std::string& libraryFolder, bool recompileLibraries);

    // Soft static obstacle avoidance constraint.
    std::unique_ptr<ocs2::StateCost> get_soft_static_obstacle_constraint(
        ocs2::PinocchioInterface pinocchioInterface,
        const StaticObstacleSettings& settings,
        const std::string& obstacle_urdf_path, bool useCaching,
        const std::string& libraryFolder, bool recompileLibraries);

    // Hard state and input limits.
    std::unique_ptr<ocs2::StateInputConstraint>
    get_joint_state_input_limit_constraint();

    // Soft state and input limits
    std::unique_ptr<ocs2::StateInputCost>
    get_soft_joint_state_input_limit_constraint();

    // Hard balancing inequality constraints.
    std::unique_ptr<ocs2::StateInputConstraint> get_balancing_constraint(
        const ocs2::PinocchioEndEffectorKinematicsCppAd&
            end_effector_kinematics,
        bool recompileLibraries);

    // Soft version of the balancing constraints (i.e. formulated as a cost via
    // penalty functions).
    std::unique_ptr<ocs2::StateInputCost> get_soft_balancing_constraint(
        const ocs2::PinocchioEndEffectorKinematicsCppAd&
            end_effector_kinematics,
        bool recompileLibraries);

    std::unique_ptr<ocs2::StateInputConstraint> get_object_dynamics_constraint(
        const ocs2::PinocchioEndEffectorKinematicsCppAd&
            end_effector_kinematics,
        bool recompileLibraries);

    std::unique_ptr<ocs2::StateInputCost> get_soft_object_dynamics_constraint(
        const ocs2::PinocchioEndEffectorKinematicsCppAd&
            end_effector_kinematics,
        bool recompileLibraries);

    std::unique_ptr<ocs2::StateInputConstraint> get_contact_force_constraint(
        const ocs2::PinocchioEndEffectorKinematicsCppAd&
            end_effector_kinematics,
        bool recompileLibraries);

    std::unique_ptr<ocs2::StateInputCost> get_soft_contact_force_constraint(
        const ocs2::PinocchioEndEffectorKinematicsCppAd&
            end_effector_kinematics,
        bool recompileLibraries);

    // Dynamic obstacle avoidance constraint.
    std::unique_ptr<ocs2::StateCost> getDynamicObstacleConstraint(
        ocs2::PinocchioInterface pinocchioInterface,
        const DynamicObstacleSettings& settings, bool usePreComputation,
        const std::string& libraryFolder, bool recompileLibraries);

    // TODO remove
    void loadSettings();

    ocs2::ddp::Settings ddpSettings_;
    ocs2::mpc::Settings mpcSettings_;
    ocs2::multiple_shooting::Settings sqpSettings_;
    ControllerSettings settings_;

    ocs2::OptimalControlProblem problem_;
    std::unique_ptr<ocs2::RolloutBase> rolloutPtr_;
    std::unique_ptr<ocs2::Initializer> initializerPtr_;
    std::shared_ptr<ocs2::ReferenceManager> referenceManagerPtr_;

    std::unique_ptr<ocs2::PinocchioInterface> pinocchioInterfacePtr_;

    VecXd initialState_;
};

}  // namespace upright
