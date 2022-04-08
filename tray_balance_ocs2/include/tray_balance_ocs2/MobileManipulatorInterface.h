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

#include <ocs2_core/Types.h>
#include <ocs2_core/initialization/Initializer.h>
#include <ocs2_mpc/MPC_BASE.h>
#include <ocs2_mpc/MPC_DDP.h>
#include <ocs2_oc/oc_problem/OptimalControlProblem.h>
#include <ocs2_oc/synchronized_module/ReferenceManager.h>
#include <ocs2_robotic_tools/common/RobotInterface.h>
#include <ocs2_sqp/MultipleShootingSettings.h>

#include <pinocchio/parsers/urdf.hpp>

#include <ocs2_pinocchio_interface/PinocchioEndEffectorKinematicsCppAd.h>
#include <ocs2_pinocchio_interface/PinocchioInterface.h>
#include <tray_balance_ocs2/ControllerSettings.h>
#include <tray_balance_ocs2/constraint/CollisionAvoidanceConstraint.h>
#include <tray_balance_ocs2/constraint/ObstacleConstraint.h>
#include <tray_balance_ocs2/constraint/balancing/BalancingSettings.h>
#include <tray_balance_ocs2/definitions.h>

namespace ocs2 {
namespace mobile_manipulator {

/**
 * Mobile Manipulator Robot Interface class
 */
class MobileManipulatorInterface final : public RobotInterface {
   public:
    /**
     * Constructor
     */
    explicit MobileManipulatorInterface(const ControllerSettings& settings);

    const vector_t& getInitialState() { return initialState_; }

    ddp::Settings& ddpSettings() { return ddpSettings_; }

    mpc::Settings& mpcSettings() { return mpcSettings_; }

    std::unique_ptr<MPC_BASE> getMpc();

    const OptimalControlProblem& getOptimalControlProblem() const override {
        return problem_;
    }

    const Initializer& getInitializer() const override {
        return *initializerPtr_;
    }

    std::shared_ptr<ReferenceManagerInterface> getReferenceManagerPtr()
        const override {
        return referenceManagerPtr_;
    }

    const RolloutBase& getRollout() const { return *rolloutPtr_; }

    const PinocchioInterface& getPinocchioInterface() const {
        return *pinocchioInterfacePtr_;
    }

    /** MobileManipulator PinocchioInterface factory */
    static PinocchioInterface buildPinocchioInterface(
        const std::string& urdfPath, const std::string& obstacle_urdfPath);

    static pinocchio::GeometryModel build_geometry_model(
        const std::string& urdf_path);

   private:
    std::unique_ptr<StateInputCost> getQuadraticStateInputCost(
        const std::string& taskFile);

    // std::unique_ptr<StateCost> getEndEffectorCost(
    //     PinocchioInterface pinocchioInterface, const std::string& taskFile,
    //     const std::string& prefix, bool useCaching,
    //     const std::string& libraryFolder, bool recompileLibraries);

    std::unique_ptr<StateCost> getEndEffectorCost(
        const PinocchioEndEffectorKinematicsCppAd& end_effector_kinematics);

    std::unique_ptr<StateCost> getCollisionAvoidanceConstraint(
        PinocchioInterface pinocchioInterface,
        const CollisionAvoidanceSettings& settings,
        const std::string& obstacle_urdf_path, bool useCaching,
        const std::string& libraryFolder, bool recompileLibraries);

    std::unique_ptr<StateInputCost> getJointStateInputLimitConstraint(
        const std::string& taskFile);

    std::unique_ptr<StateInputConstraint> getTrayBalanceConstraint(
        const TrayBalanceSettings& settings,
        const PinocchioEndEffectorKinematicsCppAd& end_effector_kinematics,
        bool recompileLibraries);

    // Soft version of the above (i.e. formulated as a cost via penalty
    // functions)
    std::unique_ptr<StateInputCost> getTrayBalanceSoftConstraint(
        const TrayBalanceSettings& settings,
        const PinocchioEndEffectorKinematicsCppAd& end_effector_kinematics,
        bool recompileLibraries);

    std::unique_ptr<StateCost> getDynamicObstacleConstraint(
        PinocchioInterface pinocchioInterface,
        const DynamicObstacleSettings& settings, bool usePreComputation,
        const std::string& libraryFolder, bool recompileLibraries);

    // std::unique_ptr<StateInputCost> get_zmp_cost(
    //     PinocchioInterface pinocchioInterface, const std::string& taskFile,
    //     const std::string& prefix, bool usePreComputation,
    //     const std::string& libraryFolder, bool recompileLibraries);

    std::unique_ptr<StateConstraint> getNonHolonomicConstraint();

    void loadSettings();

    ddp::Settings ddpSettings_;
    mpc::Settings mpcSettings_;
    multiple_shooting::Settings sqpSettings_;

    ControllerSettings settings_;

    OptimalControlProblem problem_;
    std::unique_ptr<RolloutBase> rolloutPtr_;
    std::unique_ptr<Initializer> initializerPtr_;
    std::shared_ptr<ReferenceManager> referenceManagerPtr_;

    std::unique_ptr<PinocchioInterface> pinocchioInterfacePtr_;

    vector_t initialState_{STATE_DIM};
};

}  // namespace mobile_manipulator
}  // namespace ocs2
