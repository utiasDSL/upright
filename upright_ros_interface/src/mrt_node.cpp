#include <mobile_manipulation_central/kalman_filter.h>
#include <mobile_manipulation_central/projectile.h>
#include <mobile_manipulation_central/robot_interfaces.h>
#include <ocs2_mpc/SystemObservation.h>
#include <ocs2_msgs/mpc_observation.h>
#include <ocs2_ros_interfaces/mrt/MRT_ROS_Interface.h>
#include <pybind11/embed.h>
#include <ros/init.h>
#include <ros/package.h>
#include <signal.h>
#include <upright_control/constraint/obstacle_constraint.h>
#include <upright_control/controller_interface.h>
#include <upright_control/reference_trajectory.h>
#include <upright_ros_interface/parsing.h>
#include <upright_ros_interface/safety.h>

#include <iostream>

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
    const auto& r = settings.dims.robot;

    SafetyMonitor monitor(settings, interface.get_pinocchio_interface());

    // MRT
    ocs2::MRT_ROS_Interface mrt(robot_name);
    mrt.initRollout(&interface.get_rollout());
    mrt.launchNodes(nh);

    // Estimation parameters
    double robot_proc_var, robot_meas_var;
    nh.param<double>("robot_proc_var", robot_proc_var, 1.0);
    nh.param<double>("robot_meas_var", robot_meas_var, 1.0);
    std::cout << "Robot process variance = " << robot_proc_var << std::endl;
    std::cout << "Robot measurement variance = " << robot_meas_var << std::endl;

    // nominal initial state and interface to the real robot
    VecXd x0 = interface.get_initial_state();

    // Initialize the robot interface
    if (r.q == 3) {
        robot_ptr.reset(new mm::RidgebackROSInterface(nh));
    } else if (r.q == 6) {
        robot_ptr.reset(new mm::UR10ROSInterface(nh));
    } else if (r.q == 9) {
        robot_ptr.reset(new mm::MobileManipulatorROSInterface(nh));
    } else {
        throw std::runtime_error("Unsupported base type.");
    }

    // Set up a custom SIGINT handler to brake the robot before shutting down
    // (this is why we set it up after the robot is initialized)
    signal(SIGINT, sigint_handler);

    // Initialize interface to dynamic obstacle estimator
    mm::ProjectileROSInterface projectile(nh, "ThingProjectile");
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
    x0.head(r.q) = robot_ptr->q();

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

    ros::Time now = ros::Time::now();
    ros::Time last_policy_update_time = now;
    ros::Duration policy_update_delay(settings.tracking.min_policy_update_time);

    // Obstacle setup.
    const bool using_projectile =
        settings.dims.o > 0 && settings.tracking.use_projectile;
    const bool using_stationary =
        settings.dims.o > 0 && !settings.tracking.use_projectile;

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
    const ocs2::scalar_t dt0 = 1 / settings.tracking.rate;
    const ocs2::scalar_t dt_warn = 1.5 / settings.tracking.rate;

    // Estimation
    mm::kf::GaussianEstimate estimate;
    estimate.x = x;
    estimate.P = MatXd::Identity(r.x, r.x);

    const MatXd I = MatXd::Identity(r.q, r.q);
    const MatXd Z = MatXd::Zero(r.q, r.q);
    MatXd C(r.q, r.x);
    C << I, Z, Z;

    const MatXd Q0 = robot_proc_var * I;
    const MatXd R0 = robot_meas_var * I;

    MatXd A(r.x, r.x);
    MatXd B(r.x, r.v);

    while (ros::ok()) {
        now = ros::Time::now();
        last_t = t;
        t = now.toSec();
        ocs2::scalar_t dt = t - last_t;

        if (dt >= dt_warn) {
            ROS_WARN_STREAM("Loop is slow: dt = " << 1000 * dt << " ms.");
        }

        // Robot feedback
        VecXd q = robot_ptr->q();

        // Build KF matrices
        // clang-format off
        A << I, dt * I, 0.5 * dt * dt * I,
             Z, I, dt * I,
             Z, Z, I;
        // clang-format on

        B << dt * dt * dt * I / 6, 0.5 * dt * dt * I, dt * I;
        MatXd Q = B * Q0 * B.transpose();

        // Estimate current state from joint position and jerk input using
        // Kalman filter
        VecXd u_robot = u.head(r.u);
        estimate = mm::kf::predict(estimate, A, Q, B * u_robot);
        estimate = mm::kf::correct(estimate, C, R0, q);
        x.head(r.x) = estimate.x;

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

        // Given current state, controller produces a desired jerk. We can
        // start using that jerk by incorporating it into the command now
        // TODO does this actually make sense?
        VecXd v_cmd = xd.segment(r.q, r.v) + dt0 * xd.segment(r.q + r.v, r.v) +
                      0.5 * dt0 * dt0 * u.head(r.u);

        // TODO probably should be a real-time publisher
        robot_ptr->publish_cmd_vel(v_cmd, /* bodyframe = */ false);

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
