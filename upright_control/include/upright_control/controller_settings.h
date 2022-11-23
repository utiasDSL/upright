#pragma once

#include <ostream>

#include <ocs2_core/reference/TargetTrajectories.h>
#include <ocs2_sqp/MultipleShootingSettings.h>
#include <ocs2_mpc/MPC_Settings.h>
#include <ocs2_oc/rollout/RolloutSettings.h>

#include <upright_control/inertial_alignment.h>
#include <upright_control/dynamics/base_type.h>
#include <upright_control/dimensions.h>
#include <upright_control/constraint/constraint_type.h>
#include <upright_control/constraint/bounded_balancing_constraints.h>
#include <upright_control/constraint/obstacle_constraint.h>

namespace upright {


struct TrackingSettings {

    // Frequency for tracking controller [Hz].
    ocs2::scalar_t rate = 125;

    ocs2::scalar_t min_policy_update_time = 0.01;

    bool enforce_state_limits = true;
    bool enforce_input_limits = false;
    bool enforce_ee_position_limits = false;

    ocs2::scalar_t state_violation_margin = 0.1;
    ocs2::scalar_t input_violation_margin = 1.0;
    ocs2::scalar_t ee_position_violation_margin = 0.1;
};

struct ControllerSettings {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    enum class SolverMethod {
        DDP,
        SQP,
    };

    VecXd initial_state;
    Vec3d gravity;

    SolverMethod solver_method = SolverMethod::SQP;
    ocs2::mpc::Settings mpc;
    ocs2::multiple_shooting::Settings sqp;
    ocs2::rollout::Settings rollout;
    TrackingSettings tracking;

    // Weights
    MatXd input_weight;
    MatXd state_weight;
    MatXd end_effector_weight;

    // Limits
    ConstraintType limit_constraint_type = ConstraintType::Soft;
    VecXd input_limit_lower;
    VecXd input_limit_upper;
    ocs2::scalar_t input_limit_mu = 1e-2;
    ocs2::scalar_t input_limit_delta = 1e-3;

    VecXd state_limit_lower;
    VecXd state_limit_upper;
    ocs2::scalar_t state_limit_mu = 1e-2;
    ocs2::scalar_t state_limit_delta = 1e-3;

    // End effector position box constraint
    bool end_effector_box_constraint_enabled = false;
    VecXd xyz_lower;
    VecXd xyz_upper;

    // We can linearize around a set of operating points instead of just using
    // a stationary trajectory.
    bool use_operating_points = false;
    // TODO this should be wrapped in a TargetTrajectories class
    // TargetTrajectories operating_trajectory;
    ocs2::scalar_array_t operating_times;
    ocs2::vector_array_t operating_states;
    ocs2::vector_array_t operating_inputs;

    // URDFs
    std::string robot_urdf_path;

    // OCS2 settings
    std::string lib_folder;

    // Robot settings
    RobotBaseType robot_base_type = RobotBaseType::Fixed;
    OptimizationDimensions dims;
    std::string end_effector_link_name;
    Vec3d base_pose; // optional, only affects the fixed base configuration
    std::map<std::string, ocs2::scalar_t> locked_joints;

    // Additional settings for constraints
    BalancingSettings balancing_settings;
    InertialAlignmentSettings inertial_alignment_settings;
    ObstacleSettings obstacle_settings;

    static ControllerSettings::SolverMethod solver_method_from_string(
        const std::string& s) {
        if (s == "ddp") {
            return ControllerSettings::SolverMethod::DDP;
        } else if (s == "sqp") {
            return ControllerSettings::SolverMethod::SQP;
        }
        throw std::runtime_error("Cannot parse SolverMethod from string.");
    }

    static std::string solver_method_to_string(
        const ControllerSettings::SolverMethod& method) {
        if (method == ControllerSettings::SolverMethod::DDP) {
            return "ddp";
        } else {
            return "sqp";
        }
    }
};

std::ostream& operator<<(std::ostream& out,
                         const ControllerSettings& settings) {
    out << "gravity = " << settings.gravity.transpose() << std::endl
        << "x0 = " << settings.initial_state.transpose() << std::endl
        << "input_weight = " << settings.input_weight << std::endl
        << "state_weight = " << settings.state_weight << std::endl
        << "end_effector_weight = " << settings.end_effector_weight << std::endl
        << "input_limit_lower = " << settings.input_limit_lower.transpose()
        << std::endl
        << "input_limit_upper = " << settings.input_limit_upper.transpose()
        << std::endl
        << "input_limit_mu = " << settings.input_limit_mu << std::endl
        << "input_limit_delta = " << settings.input_limit_delta << std::endl
        << "state_limit_lower = " << settings.state_limit_lower.transpose()
        << std::endl
        << "state_limit_upper = " << settings.state_limit_upper.transpose()
        << std::endl
        << "state_limit_mu = " << settings.state_limit_mu << std::endl
        << "state_limit_delta = " << settings.state_limit_delta << std::endl
        << "use_operating_points = " << settings.use_operating_points
        << std::endl
        << "robot_urdf_path = " << settings.robot_urdf_path << std::endl
        << "robot_base_type = "
        << robot_base_type_to_string(settings.robot_base_type) << std::endl
        << "end_effector_link_name = " << settings.end_effector_link_name
        << std::endl
        << "dims" << std::endl
        << settings.dims << std::endl
        << "balancing_settings" << std::endl
        << settings.balancing_settings << std::endl
        << "inertial_alignment_settings" << std::endl
        << settings.inertial_alignment_settings << std::endl;
    return out;
}

}  // namespace upright
