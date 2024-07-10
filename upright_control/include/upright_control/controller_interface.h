#pragma once

#include <string>

#include <ocs2_core/Types.h>
#include <ocs2_core/initialization/Initializer.h>
#include <ocs2_mpc/MPC_BASE.h>
#include <ocs2_oc/oc_problem/OptimalControlProblem.h>
#include <ocs2_oc/rollout/RolloutBase.h>
#include <ocs2_oc/synchronized_module/ReferenceManager.h>
#include <ocs2_oc/synchronized_module/ReferenceManagerInterface.h>
#include <ocs2_pinocchio_interface/PinocchioEndEffectorKinematicsCppAd.h>
#include <ocs2_pinocchio_interface/PinocchioEndEffectorKinematics.h>
#include <ocs2_pinocchio_interface/PinocchioInterface.h>
#include <ocs2_robotic_tools/common/RobotInterface.h>
#include <ocs2_self_collision/PinocchioGeometryInterface.h>

#include <upright_control/constraint/obstacle_constraint.h>
#include <upright_control/controller_settings.h>
#include <upright_control/types.h>

namespace upright {

class ControllerInterface final : public ocs2::RobotInterface {
   public:
    explicit ControllerInterface(const ControllerSettings& settings);

    const ocs2::OptimalControlProblem& getOptimalControlProblem()
        const override {
        return problem_;
    }

    const ocs2::Initializer& getInitializer() const override {
        return *initializer_ptr_;
    }

    std::shared_ptr<ocs2::ReferenceManagerInterface> getReferenceManagerPtr()
        const override {
        return reference_manager_ptr_;
    }

    const VecXd& get_initial_state() { return initial_state_; }

    std::unique_ptr<ocs2::MPC_BASE> get_mpc();

    const ocs2::RolloutBase& get_rollout() const { return *rollout_ptr_; }

    const ocs2::PinocchioInterface& get_pinocchio_interface() const {
        return *pinocchio_interface_ptr;
    }

    ocs2::PinocchioEndEffectorKinematicsCppAd& get_end_effector_kinematics() const {
        return *end_effector_kinematics_ptr_;
    }

   private:
    std::unique_ptr<ocs2::StateInputCost> get_quadratic_state_input_cost();

    // Hard static obstacle avoidance constraint.
    std::unique_ptr<ocs2::StateConstraint> get_obstacle_constraint(
        ocs2::PinocchioInterface& pinocchio_interface,
        ocs2::PinocchioGeometryInterface& geom_interface,
        const ObstacleSettings& settings, const std::string& library_folder,
        bool recompile_libraries);

    // Hard state and input limits.
    std::unique_ptr<ocs2::StateInputConstraint>
    get_joint_state_input_limit_constraint();

    std::unique_ptr<ocs2::StateInputConstraint> get_object_dynamics_constraint(
        const ocs2::PinocchioEndEffectorKinematicsCppAd&
            end_effector_kinematics,
        bool recompileLibraries);

    std::unique_ptr<ocs2::StateInputConstraint> get_contact_force_constraint(
        const ocs2::PinocchioEndEffectorKinematicsCppAd&
            end_effector_kinematics,
        bool recompileLibraries);

    ControllerSettings settings_;
    ocs2::OptimalControlProblem problem_;
    std::unique_ptr<ocs2::RolloutBase> rollout_ptr_;
    std::unique_ptr<ocs2::Initializer> initializer_ptr_;
    std::shared_ptr<ocs2::ReferenceManager> reference_manager_ptr_;
    std::unique_ptr<ocs2::PinocchioInterface> pinocchio_interface_ptr;
    std::unique_ptr<ocs2::PinocchioEndEffectorKinematicsCppAd> end_effector_kinematics_ptr_;

    VecXd initial_state_;
};

}  // namespace upright
