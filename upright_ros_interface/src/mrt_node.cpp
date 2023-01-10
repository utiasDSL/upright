#include <mobile_manipulation_central/kalman_filter.h>
#include <mobile_manipulation_central/projectile.h>
#include <mobile_manipulation_central/robot_interfaces.h>
#include <ocs2_mpc/SystemObservation.h>
#include <ocs2_msgs/mpc_observation.h>
#include <ocs2_pinocchio_interface/PinocchioEndEffectorKinematics.h>
#include <ocs2_pinocchio_interface/PinocchioInterface.h>
#include <ocs2_ros_interfaces/mrt/MRT_ROS_Interface.h>
#include <pybind11/embed.h>
#include <ros/init.h>
#include <ros/package.h>
#include <signal.h>
#include <upright_control/constraint/obstacle_constraint.h>
#include <upright_control/controller_interface.h>
#include <upright_control/dynamics/system_pinocchio_mapping.h>
#include <upright_control/reference_trajectory.h>
#include <upright_ros_interface/parsing.h>

#include <iostream>
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/fwd.hpp>

using namespace upright;

const double PROJECTILE_ACTIVATION_HEIGHT = 1.0;  // meters

// Robot is a global variable so we can send it a brake command in the SIGINT
// handler
std::unique_ptr<mm::RobotROSInterface> robot_ptr;

// Custom SIGINT handler
void sigint_handler(int sig) {
    std::cerr << "Received SIGINT." << std::endl;
    std::cerr << "Braking robot." << std::endl;
    robot_ptr->brake();
    ros::shutdown();
}

// Double integration using semi-implicit Euler method
std::tuple<VecXd, VecXd> double_integrate(const VecXd& v, const VecXd& a,
                                          const VecXd& u,
                                          ocs2::scalar_t timestep) {
    VecXd a_new = a + timestep * u;
    VecXd v_new = v + timestep * a_new;
    return std::tuple<VecXd, VecXd>(v_new, a_new);
}

std::tuple<VecXd, VecXd> double_integrate_repeated(const VecXd& v,
                                                   const VecXd& a,
                                                   const VecXd& u,
                                                   ocs2::scalar_t timestep,
                                                   size_t n) {
    ocs2::scalar_t dt = timestep / n;
    VecXd v_new = v;
    VecXd a_new = a;
    for (int i = 0; i < n; ++i) {
        std::tie(v_new, a_new) = double_integrate(v_new, a_new, u, dt);
    }
    return std::tuple<VecXd, VecXd>(v_new, a_new);
}

class SafetyMonitor {
   public:
    SafetyMonitor(const ControllerSettings& settings,
                  const ocs2::PinocchioInterface& pinocchio_interface)
        : settings_(settings), pinocchio_interface_(pinocchio_interface) {
        SystemPinocchioMapping<TripleIntegratorPinocchioMapping<ocs2::scalar_t>,
                               ocs2::scalar_t>
            mapping(settings.dims);
        kinematics_ptr_.reset(new ocs2::PinocchioEndEffectorKinematics(
            pinocchio_interface, mapping, {settings.end_effector_link_name}));
        kinematics_ptr_->setPinocchioInterface(pinocchio_interface_);
    }

    bool state_limits_violated(const VecXd& x) const {
        VecXd x_robot = x.head(settings_.dims.robot.x);

        if (((x_robot - settings_.state_limit_lower).array() <
             -settings_.tracking.state_violation_margin)
                .any()) {
            std::cout << "x = " << x_robot.transpose() << std::endl;
            std::cout << "State violated lower limits!" << std::endl;
            return true;
        }
        if (((settings_.state_limit_upper - x_robot).array() <
             -settings_.tracking.state_violation_margin)
                .any()) {
            std::cout << "x = " << x_robot.transpose() << std::endl;
            std::cout << "State violated upper limits!" << std::endl;
            return true;
        }
        return false;
    }

    bool input_limits_violated(const VecXd& u) const {
        VecXd u_robot = u.head(settings_.dims.robot.u);

        if (((u_robot - settings_.input_limit_lower).array() <
             -settings_.tracking.input_violation_margin)
                .any()) {
            std::cout << "u = " << u_robot.transpose() << std::endl;
            std::cout << "Input violated lower limits!" << std::endl;
            return true;
        }
        if (((settings_.input_limit_upper - u_robot).array() <
             -settings_.tracking.input_violation_margin)
                .any()) {
            std::cout << "u = " << u_robot.transpose() << std::endl;
            std::cout << "Input violated upper limits!" << std::endl;
            return true;
        }
        return false;
    }

    bool end_effector_position_violated(const ocs2::TargetTrajectories& target,
                                        ocs2::scalar_t t, const VecXd& x) {
        VecXd q = VecXd::Zero(settings_.dims.q());
        q.tail(settings_.dims.robot.q) = x.head(settings_.dims.robot.q);

        pinocchio::forwardKinematics(pinocchio_interface_.getModel(),
                                     pinocchio_interface_.getData(), q);
        pinocchio::updateFramePlacements(pinocchio_interface_.getModel(),
                                         pinocchio_interface_.getData());
        Vec3d actual_position = kinematics_ptr_->getPosition(x).front();
        Vec3d desired_position = interpolate_end_effector_pose(t, target).first;

        VecXd position_constraint(6);
        position_constraint
            << desired_position + settings_.xyz_upper - actual_position,
            actual_position - (desired_position + settings_.xyz_lower);

        if (position_constraint.minCoeff() <
            -settings_.tracking.ee_position_violation_margin) {
            std::cerr << "Controller violated position limits!" << std::endl;
            std::cerr << "Controller position = " << actual_position.transpose()
                      << std::endl;
            std::cerr << "Desired position = " << desired_position.transpose()
                      << std::endl;
            std::cerr << "q = " << q.transpose() << std::endl;
            return true;
        }
        return false;
    }

   private:
    ControllerSettings settings_;
    std::unique_ptr<ocs2::PinocchioEndEffectorKinematics> kinematics_ptr_;
    ocs2::PinocchioInterface pinocchio_interface_;
};

int main(int argc, char** argv) {
    const std::string robot_name = "mobile_manipulator";

    if (argc < 2) {
        throw std::runtime_error("Config path is required.");
    }

    // Initialize ros node
    ros::init(argc, argv, robot_name + "_mrt");
    ros::NodeHandle nh;
    std::string config_path = std::string(argv[1]);

    // controller interface
    // Python interpreter required for now because we actually load the control
    // settings and the target trajectories using Python - not ideal but easier
    // than re-implementing the parsing logic in C++
    py::scoped_interpreter guard{};
    ControllerSettings settings = parse_control_settings(config_path);
    std::cout << settings << std::endl;
    ControllerInterface interface(settings);

    SafetyMonitor monitor(settings, interface.get_pinocchio_interface());

    // MRT
    ocs2::MRT_ROS_Interface mrt(robot_name);
    mrt.initRollout(&interface.get_rollout());
    mrt.launchNodes(nh);

    double robot_proc_var, robot_meas_var;
    nh.param<double>("robot_proc_var", robot_proc_var, 1.0);
    nh.param<double>("robot_meas_var", robot_meas_var, 1.0);
    std::cout << "Robot process variance = " << robot_proc_var << std::endl;
    std::cout << "Robot measurement variance = " << robot_meas_var << std::endl;

    ros::Publisher est_pub =
        nh.advertise<ocs2_msgs::mpc_observation>("/mobile_manipulator_state_estimate", 1);

    // nominal initial state and interface to the real robot
    VecXd x0 = interface.get_initial_state();

    // Initialize the robot interface
    if (settings.dims.robot.q == 3) {
        robot_ptr.reset(new mm::RidgebackROSInterface(nh));
    } else if (settings.dims.robot.q == 6) {
        robot_ptr.reset(new mm::UR10ROSInterface(nh));
    } else if (settings.dims.robot.q == 9) {
        robot_ptr.reset(new mm::MobileManipulatorROSInterface(nh));
    } else {
        throw std::runtime_error("Unsupported base type.");
    }

    // Set up a custom SIGINT handler to brake the robot before shutting down
    // (this is why we set it up after the robot is initialized)
    signal(SIGINT, sigint_handler);

    // Initialize interface to dynamic obstacle estimator
    mm::ProjectileROSInterface projectile(nh, "Projectile");
    bool avoid_dynamic_obstacle = false;

    ocs2::scalar_t timestep = 1.0 / settings.tracking.rate;
    ros::Rate rate(settings.tracking.rate);

    // wait until we get feedback from the robot
    while (ros::ok()) {
        ros::spinOnce();
        if (robot_ptr->ready()) {
            break;
        }
        rate.sleep();
    }
    std::cout << "Received feedback from robot." << std::endl;

    // update to the real initial state
    x0.head(settings.dims.robot.q) = robot_ptr->q();

    // reset MPC
    ocs2::TargetTrajectories target = parse_target_trajectory(config_path, x0);
    mrt.resetMpcNode(target);

    ocs2::SystemObservation observation;
    observation.state = x0;
    observation.input.setZero(settings.dims.u());
    observation.time = ros::Time::now().toSec();
    mrt.setCurrentObservation(observation);

    // Let MPC generate the initial plan
    while (ros::ok()) {
        mrt.spinMRT();
        if (mrt.initialPolicyReceived()) {
            break;
        }
        rate.sleep();
    }
    mrt.updatePolicy();
    std::cout << "Received first policy." << std::endl;

    VecXd x = x0;
    VecXd xd = VecXd::Zero(x.size());
    VecXd u = VecXd::Zero(settings.dims.u());
    size_t mode = 0;

    VecXd v_ff = VecXd::Zero(settings.dims.robot.v);
    VecXd a_ff = VecXd::Zero(settings.dims.robot.v);

    ros::Time now = ros::Time::now();
    ros::Time last_policy_update_time = now;
    ros::Duration policy_update_delay(settings.tracking.min_policy_update_time);

    // Obstacle setup.
    const bool using_projectile =
        settings.dims.o > 0 && settings.tracking.use_projectile;
    const bool using_stationary =
        settings.dims.o > 0 && !settings.tracking.use_projectile;

    std::cout << "using_projectile = " << using_projectile << std::endl;
    std::cout << "using_stationary = " << using_stationary << std::endl;

    DynamicObstacle* obstacle;
    if (settings.dims.o > 0) {
        obstacle = &settings.obstacle_settings.dynamic_obstacles[0];
        const size_t num_modes = obstacle->modes.size();
        if ((using_projectile && num_modes != 1) ||
            (using_stationary && num_modes != 2)) {
            throw std::runtime_error(
                "Dynamic obstacle has wrong number of modes.");
        }
    }

    ocs2::scalar_t t = now.toSec();
    ocs2::scalar_t last_t = t;
    const ocs2::scalar_t t0 = t;

    // Estimation
    mm::GaussianEstimate estimate;
    estimate.x = x;
    estimate.P = MatXd::Identity(settings.dims.robot.x, settings.dims.robot.x);

    MatXd I = MatXd::Identity(settings.dims.robot.q, settings.dims.robot.q);
    MatXd Z = MatXd::Zero(settings.dims.robot.q, settings.dims.robot.q);
    MatXd C(settings.dims.robot.q, settings.dims.robot.x);
    C << I, Z, Z;

    MatXd Q0 = robot_proc_var * I;
    MatXd R0 = robot_meas_var * I;

    MatXd A(settings.dims.robot.x, settings.dims.robot.x);
    MatXd B(settings.dims.robot.x, settings.dims.robot.v);

    while (ros::ok()) {
        now = ros::Time::now();
        last_t = t;
        t = now.toSec();
        ocs2::scalar_t dt = t - last_t;

        // Robot feedback
        VecXd q = robot_ptr->q();

        // clang-format off
        A << I, dt * I, 0.5 * dt * dt * I,
             Z, I, dt * I,
             Z, Z, I;
        B << dt * dt * dt * I / 6, 0.5 * dt * dt * I, dt * I;
        // clang-format on

        MatXd Q = B * Q0 * B.transpose();

        // Integrate our internal model to get velocity and acceleration
        // "feedback"
        VecXd u_robot = u.head(settings.dims.robot.u);

        estimate = mm::kf_predict(estimate, A, Q, B * u_robot);
        estimate = mm::kf_correct(estimate, C, R0, q);

        ocs2_msgs::mpc_observation obs_msg;
        obs_msg.time = t;
        VecX<float> xf = estimate.x.cast<float>();
        obs_msg.state.value = std::vector<float>(xf.data(), xf.data() + xf.size());
        est_pub.publish(obs_msg);

        std::tie(v_ff, a_ff) = double_integrate(v_ff, a_ff, u_robot, dt);

        // Current state is built from robot feedback for q; for velocity and
        // acceleration we just assume we are tracking well since we don't get
        // good feedback on this
        x.head(settings.dims.robot.x) << q, v_ff, a_ff;

        if (using_projectile && projectile.ready()) {
            Vec3d q_obs = projectile.q();
            if (q_obs(2) > PROJECTILE_ACTIVATION_HEIGHT) {
                avoid_dynamic_obstacle = true;
                std::cout << "  q_obs = " << q_obs.transpose() << std::endl;
            } else {
                std::cout << "~ q_obs = " << q_obs.transpose() << std::endl;
            }

            // TODO we could have the MPC reset if the projectile was inside
            // the "awareness zone" but then leaves, such that the robot is
            // ready for the next throw

            // TODO should this eventually stop? like when the obstacle goes
            // below a certain threshold?
            if (avoid_dynamic_obstacle) {
                Vec3d v_obs = projectile.v();
                Vec3d a_obs = obstacle->modes[0].acceleration;
                x.tail(9) << q_obs, v_obs, a_obs;
            }
        } else if (using_stationary) {
            if (t - t0 <= obstacle->modes[1].time) {
                x.tail(9) = obstacle->modes[0].state();
            } else {
                x.tail(9) = obstacle->modes[1].state();
            }
        }

        // // Dynamic obstacle
        // if (settings.dims.o > 0 && projectile.ready()) {
        //     Vec3d q_obs = projectile.q();
        //     if (q_obs(2) > PROJECTILE_ACTIVATION_HEIGHT) {
        //         avoid_dynamic_obstacle = true;
        //         std::cout << "  q_obs = " << q_obs.transpose() << std::endl;
        //     } else {
        //         std::cout << "~ q_obs = " << q_obs.transpose() << std::endl;
        //     }
        //
        //     // TODO we could have the MPC reset if the projectile was inside
        //     // the "awareness zone" but then leaves, such that the robot is
        //     // ready for the next throw
        //
        //     // TODO should this eventually stop? like when the obstacle goes
        //     // below a certain threshold?
        //     if (avoid_dynamic_obstacle) {
        //         Vec3d v_obs = projectile.v();
        //         Vec3d a_obs = obstacle.modes[0].acceleration;
        //         x.tail(9) << q_obs, v_obs, a_obs;
        //     }
        // }

        // Compute optimal state and input using current policy
        mrt.evaluatePolicy(t, x, xd, u, mode);

        if (using_projectile || using_stationary) {
            std::cout << "x_obs = " << x.tail(9).transpose() << std::endl;
            std::cout << "xd_obs = " << xd.tail(9).transpose() << std::endl;
        }

        // Check that controller is not making the end effector leave allowed
        // region
        if (settings.tracking.enforce_ee_position_limits &&
            monitor.end_effector_position_violated(target, t, xd)) {
            break;
        }

        // Check that the controller has provided sane values.
        // Check monitor first so we can still log violations
        if (settings.tracking.enforce_state_limits &&
            monitor.state_limits_violated(xd)) {
            break;
        }
        if (settings.tracking.enforce_input_limits &&
            monitor.input_limits_violated(u)) {
            break;
        }

        robot_ptr->publish_cmd_vel(v_ff, /* bodyframe = */ false);

        // Send observation to MPC
        observation.time = t;
        observation.state = x;
        observation.input = u;
        mrt.setCurrentObservation(observation);

        ros::spinOnce();

        // Get new policy messages and update the policy if available
        mrt.spinMRT();
        if (now - last_policy_update_time >= policy_update_delay) {
            mrt.updatePolicy();
            last_policy_update_time = now;
        }

        rate.sleep();
    }

    // stop the robot when we're done
    std::cout << "Braking robot." << std::endl;
    robot_ptr->brake();
    ros::shutdown();

    // Successful exit
    return 0;
}
