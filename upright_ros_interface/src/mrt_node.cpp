#include <mobile_manipulation_central/kalman_filter.h>
#include <mobile_manipulation_central/projectile.h>
#include <mobile_manipulation_central/robot_interfaces.h>
#include <ocs2_mpc/SystemObservation.h>
#include <ocs2_msgs/mpc_observation.h>
#include <ocs2_pinocchio_interface/PinocchioEndEffectorKinematics.h>
#include <ocs2_ros_interfaces/mrt/MRT_ROS_Interface.h>
#include <pybind11/embed.h>
#include <realtime_tools/realtime_publisher.h>
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

enum class ProjectileState {
    Preflight,
    Flight,
    Postflight,
};

const double PROJECTILE_ACTIVATION_HEIGHT = 1.0;  // meters
const double PROJECTILE_DEACTIVATION_HEIGHT = 0.2;

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

// Solve for (x, y) and time t when projectile reaches a given height (in the
// future).
std::tuple<bool, Vec3d> solve_projectile_height(const Vec3d& r0,
                                                const Vec3d& v0, double h,
                                                double g) {
    bool success = false;
    double z0 = r0(2);
    double vz = v0(2);

    // solve for intersection time
    // check if discriminant is negative (which means no real solutions)
    double t = 0;
    double disc = vz * vz - 2 * (z0 - h) * g;
    if (disc >= 0) {
        success = true;
        double s = sqrt(disc);
        double t1 = (-vz - s) / g;
        double t2 = (-vz + s) / g;
        t = std::max(t1, t2);
    }

    // solve for intersection point
    Vec3d r = r0 + t * v0 + 0.5 * t * t * Vec3d(0, 0, g);

    return std::tuple<bool, Vec3d>(success, r);
}

Vec2d perp2d(const Vec2d& a) { return Vec2d(-a(1), a(0)); }

double angle_between(const Vec2d& a, const Vec2d& b) {
    double angle = acos(a.dot(b));
    Vec2d a_perp = perp2d(a);
    if (b.dot(a_perp) > 0) {
        angle = -angle;
    }
    return angle;
}

Mat2d rot2d(double angle) {
    double c = cos(angle);
    double s = sin(angle);
    Mat2d C;
    C << c, -s, s, c;
    return C;
}

Vec3d compute_goal_from_projectile(const VecXd& x, const Vec3d& r_ew_w,
                                   double distance) {
    VecXd q = x.head(9);  // TODO hard code
    VecXd x_obs = x.tail(9);

    Vec3d r_obs = x_obs.head(3);
    Vec3d v_obs = x_obs.segment(3, 3);
    // Vec3d r_int;
    // bool success;
    // std::tie(success, r_int) =
    //     solve_projectile_height(r_obs, v_obs, r_ew_w(2), -9.81);
    // if (!success) {
    //     std::cout << "FAILED TO SOLVE PROJECTILE HEIGHT!" << std::endl;
    // }

    std::cout << "x_obs = " << x_obs.transpose() << std::endl;
    std::cout << "r_ew_w = " << r_ew_w.transpose() << std::endl;
    // std::cout << "r_int = " << r_int.transpose() << std::endl;

    // Normal of the plane of flight of the ball
    // We make it so that it points away from the predicted intersection
    // location
    // TODO r_int is not needed: we can find direction directly from n_obs,
    // r_obs, and r_ew_w
    Vec3d n_obs = v_obs.cross(Vec3d::UnitZ()).normalized();
    Vec3d delta = r_ew_w - r_obs;
    if (n_obs.dot(delta) < 0) {
        n_obs *= -1;
    }

    std::cout << "n_obs = " << n_obs.transpose() << std::endl;

    // Find angle of ball normal w.r.t. the EE direction
    // TODO bit of a hack
    double yaw = q(2);
    Vec2d n_ee(cos(yaw), sin(yaw));
    double angle = angle_between(n_obs.head(2), n_ee);

    std::cout << "angle = " << angle << std::endl;

    // Limit the angle to pre-specified bounds w.r.t. the EE. These are roughly
    // chosen as fairly "free" directions in which the robot can move quickly.
    if (angle > 0) {
        angle = std::min(std::max(angle, 0 * M_PI), 0.875 * M_PI);
    } else {
        angle = -std::min(std::max(-angle, 0 * M_PI), 0.875 * M_PI);
    }

    // if ball is going to land in front of EE, go up
    // if ball is going to land behind EE, go down
    double dz_goal = 0;
    // if (v_obs.head(2).dot(delta) >= 0) {
    //     // ball is in front of EE: go up
    //     dz_goal = 0.25;
    // } else {
    //     // ball is behind EE: go down
    //     dz_goal = -0.25;
    // }

    // for now, just always move the EE
    Vec2d n_goal = rot2d(angle) * n_ee;
    Vec3d n_goal3d;
    n_goal3d << distance * n_goal, dz_goal;
    Vec3d goal = r_ew_w + n_goal3d;
    return goal;

    // std::cout << "n_goal = " << n_goal.transpose() << std::endl;
    //
    // Vec2d delta2d = -delta.head(2);
    // Vec2d delta2d_perp = delta2d - delta2d.dot(n_goal) * n_goal;
    // double w = delta2d_perp.norm();
    // if (w >= d) {
    //
    // }
    // double d = sqrt(distance * distance - w * w);
    // Vec2d goal2d = r_int.head(2) - delta2d_perp + d * n_goal;
    //
    // std::cout << "d = " << d << std::endl;
    // std::cout << "delta2d = " << delta2d.transpose() << std::endl;
    // std::cout << "delta2d_perp = " << delta2d_perp.transpose() << std::endl;
    // std::cout << "goal2d = " << goal2d.transpose() << std::endl;
    //
    // Vec3d goal;
    // goal << goal2d, r_ew_w(2);
    // return goal;
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

    // When debugging, we publish the desired state and input planned by the
    // MPC at each timestep.
    realtime_tools::RealtimePublisher<ocs2_msgs::mpc_observation> mpc_plan_pub(
        nh, robot_name + "_mpc_plan", 1);

    realtime_tools::RealtimePublisher<ocs2_msgs::mpc_observation> cmd_pub(
        nh, robot_name + "_cmds", 1);

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

    // Initialize interface to dynamic obstacle estimator if we are using
    // dynamic obstacles
    mm::ProjectileROSInterface projectile;
    // if (settings.dims.o > 0) {  TODO after experiments
    projectile.init(nh);
    // }
    // bool avoid_dynamic_obstacle = false;
    ProjectileState projectile_state = ProjectileState::Preflight;

    ros::Rate rate(settings.tracking.rate);

    // Wait until we get feedback from the robot to do remaining setup.
    while (ros::ok()) {
        ros::spinOnce();
        if (robot_ptr->ready()) {
            break;
        }
        rate.sleep();
    }
    std::cout << "Received feedback from robot." << std::endl;

    // Update initial state with robot feedback
    VecXd x0 = interface.get_initial_state();
    x0.head(r.q) = robot_ptr->q();

    // Reset MPC with our desired target trajectory
    ocs2::TargetTrajectories target = parse_target_trajectory(config_path, x0);
    mrt.resetMpcNode(target);

    // Initial state and input
    VecXd x = x0;
    VecXd xd = VecXd::Zero(x.size());
    VecXd u = VecXd::Zero(settings.dims.u());
    size_t mode = 0;

    ocs2::SystemObservation observation;
    observation.state = x0;
    observation.input = u;

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

    ros::Duration policy_update_delay(settings.tracking.min_policy_update_time);
    const ocs2::scalar_t dt0 = 1 / settings.tracking.rate;
    const ocs2::scalar_t dt_warn = 1.5 / settings.tracking.rate;

    // Estimation
    mm::kf::GaussianEstimate estimate;
    estimate.x = x;
    estimate.P =
        settings.estimation.robot_init_variance * MatXd::Identity(r.x, r.x);

    const MatXd I = MatXd::Identity(r.q, r.q);
    const MatXd Z = MatXd::Zero(r.q, r.q);
    MatXd C(r.q, r.x);
    C << I, Z, Z;

    const MatXd Q0 = settings.estimation.robot_process_variance * I;
    const MatXd R0 = settings.estimation.robot_measurement_variance * I;

    MatXd A(r.x, r.x);
    MatXd B(r.x, r.v);

    // Commands
    VecXd v_cmd = VecXd::Zero(r.v);
    VecXd u_cmd = VecXd::Zero(r.u);

    // Manual state feedback gain
    MatXd Kx(r.u, r.x);
    Kx << settings.tracking.kp * I, settings.tracking.kv * I,
        settings.tracking.ka * I;

    // Let MPC generate the initial plan
    observation.time = ros::Time::now().toSec();
    mrt.setCurrentObservation(observation);
    while (ros::ok()) {
        mrt.spinMRT();
        if (mrt.initialPolicyReceived()) {
            break;
        }
        rate.sleep();
    }
    mrt.updatePolicy();
    std::cout << "Received first policy." << std::endl;

    // Initialize time
    ros::Time now = ros::Time::now();
    ros::Time last_policy_update_time = now;
    ocs2::scalar_t t = now.toSec();
    ocs2::scalar_t last_t = t;
    const ocs2::scalar_t t0 = t;

    std::unique_ptr<ocs2::PinocchioEndEffectorKinematicsCppAd> kinematics_ptr(
        interface.get_end_effector_kinematics().clone());

    // Initial EE position
    const Vec3d r_ew_w0 = kinematics_ptr->getPosition(x0).front();

    // Now that we're all set up and have an initial policy, we can get started
    // moving the robot.
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
        // VecXd u_robot = u.head(r.u);
        estimate = mm::kf::predict(estimate, A, Q, B * u_cmd);
        estimate = mm::kf::correct(estimate, C, R0, q);
        x.head(r.x) = estimate.x;

        // Dynamic obstacles
        if (using_projectile && projectile.ready()) {
            Vec3d q_obs = projectile.q();

            if (projectile_state == ProjectileState::Preflight &&
                q_obs(2) > PROJECTILE_ACTIVATION_HEIGHT) {
                // Ball is detected: avoid the ball
                Vec3d v_obs = projectile.v();
                Vec3d a_obs = obstacle->modes[0].acceleration;
                x.tail(9) << q_obs, v_obs, a_obs;

                Vec3d r_ew_w = kinematics_ptr->getPosition(x).front();
                Vec3d goal = compute_goal_from_projectile(x, r_ew_w, 1);
                ocs2::vector_array_t new_xs = target.stateTrajectory;
                new_xs[0].head(3) = goal;
                ocs2::TargetTrajectories new_target(
                    target.timeTrajectory, new_xs, target.inputTrajectory);

                std::cout << "x = " << x.transpose() << std::endl;
                std::cout << "P = " << new_xs[0].transpose() << std::endl;

                // mrt.resetMpcNode(new_target);
                mrt.resetTarget(new_target);

                projectile_state = ProjectileState::Flight;
            } else if (projectile_state == ProjectileState::Flight &&
                       q_obs(2) < PROJECTILE_DEACTIVATION_HEIGHT) {
                // Ball has passed: go back to the origin
                ocs2::vector_array_t new_xs = target.stateTrajectory;
                new_xs[0].head(3) = r_ew_w0;
                ocs2::TargetTrajectories new_target(
                    target.timeTrajectory, new_xs, target.inputTrajectory);
                mrt.resetTarget(new_target);

                projectile_state = ProjectileState::Postflight;
            }

            // Always update state once we're past preflight
            if (projectile_state != ProjectileState::Preflight) {
                Vec3d v_obs = projectile.v();
                Vec3d a_obs = obstacle->modes[0].acceleration;
                x.tail(9) << q_obs, v_obs, a_obs;
            }

            //
            // Vec3d q_obs = projectile.q();
            // if (q_obs(2) > PROJECTILE_ACTIVATION_HEIGHT) {
            //     if (projectile_state == ProjectileState::Preflight) {
            //         // activated for the first time: plan a new trajectory to
            //         // avoid the ball
            //         Vec3d v_obs = projectile.v();
            //         Vec3d a_obs = obstacle->modes[0].acceleration;
            //         x.tail(9) << q_obs, v_obs, a_obs;
            //
            //         Vec3d r_ew_w = kinematics_ptr->getPosition(x).front();
            //         Vec3d goal = compute_goal_from_projectile(x, r_ew_w, 1);
            //         ocs2::vector_array_t new_xs = target.stateTrajectory;
            //         new_xs[0].head(3) = goal;
            //         ocs2::TargetTrajectories new_target(
            //             target.timeTrajectory, new_xs,
            //             target.inputTrajectory);
            //
            //         std::cout << "x = " << x.transpose() << std::endl;
            //         std::cout << "P = " << new_xs[0].transpose() <<
            //         std::endl;
            //
            //         // mrt.resetMpcNode(new_target);
            //         mrt.resetTarget(new_target);
            //     }
            //     projectile_state = ProjectileState::Flight;
            //     // avoid_dynamic_obstacle = true;
            //     std::cout << "  q_obs = " << q_obs.transpose() << std::endl;
            // } else {
            //     std::cout << "~ q_obs = " << q_obs.transpose() << std::endl;
            // }

            // TODO we could have the MPC reset if the projectile was inside
            // the "awareness zone" but then leaves, such that the robot is
            // ready for the next throw

            // TODO should this eventually stop? like when the obstacle goes
            // below a certain threshold?
            // if (projectile_state == projectile_state::Flight) {
            //     Vec3d v_obs = projectile.v();
            //     Vec3d a_obs = obstacle->modes[0].acceleration;
            //     x.tail(9) << q_obs, v_obs, a_obs;
            // }
        } else if (using_stationary) {
            if (t - t0 <= obstacle->modes[1].time) {
                x.tail(9) = obstacle->modes[0].state();
            } else {
                x.tail(9) = obstacle->modes[1].state();
            }
        }

        // Compute optimal state and input using current policy
        mrt.evaluatePolicy(t, x, xd, u, mode);

        if (settings.debug) {
            if (using_projectile) {
                std::cout << "x_obs = " << x.tail(9).transpose() << std::endl;
                std::cout << "xd_obs = " << xd.tail(9).transpose() << std::endl;
            }

            // Publish planned state and input
            if (mpc_plan_pub.trylock()) {
                VecX<float> xf = xd.cast<float>();
                VecX<float> uf = u.cast<float>();

                mpc_plan_pub.msg_.time = t;
                mpc_plan_pub.msg_.state.value =
                    std::vector<float>(xf.data(), xf.data() + xf.size());
                mpc_plan_pub.msg_.input.value =
                    std::vector<float>(uf.data(), uf.data() + uf.size());
                mpc_plan_pub.unlockAndPublish();
            }

            // Publish commanded (integrated) velocity and acceleration
            if (cmd_pub.trylock()) {
                VecX<float> xf = VecX<float>::Zero(r.x);
                xf.segment(r.q, r.v) = v_cmd.cast<float>();

                cmd_pub.msg_.time = t;
                cmd_pub.msg_.state.value =
                    std::vector<float>(xf.data(), xf.data() + xf.size());
                cmd_pub.unlockAndPublish();
            }
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

        // State feedback
        // This should only be used when an optimal feedback policy is not
        // computed internally by the MPC
        u_cmd = Kx * (xd - x).head(r.x) + u.head(r.u);
        u.head(r.u) = u_cmd;

        // Double integrate the commanded jerk to get commanded velocity
        v_cmd = x.segment(r.q, r.v) + dt * x.segment(r.q + r.v, r.v) +
                0.5 * dt * dt * u_cmd;

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
