#include <iostream>

#include <pybind11/embed.h>
#include <ros/init.h>
#include <ros/package.h>

#include <ocs2_mpc/SystemObservation.h>
#include <ocs2_ros_interfaces/mrt/MRT_ROS_Interface.h>

#include <mobile_manipulation_central/robot_interfaces.h>
#include <mobile_manipulation_central/projectile.h>

#include <upright_control/controller_interface.h>
#include <upright_ros_interface/parsing.h>

using namespace upright;

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

bool limits_violated(const ControllerSettings& settings, const VecXd& x,
                     const VecXd& u) {
    VecXd x_robot = x.head(settings.dims.robot.x);
    VecXd u_robot = u.head(settings.dims.robot.u);

    if (((x_robot - settings.state_limit_lower).array() < 0).any()) {
        std::cout << "x = " << x_robot.transpose() << std::endl;
        std::cout << "State violated lower limits!" << std::endl;
        return true;
    }
    if (((settings.state_limit_upper - x_robot).array() < 0).any()) {
        std::cout << "x = " << x_robot.transpose() << std::endl;
        std::cout << "State violated upper limits!" << std::endl;
        return true;
    }
    if (((u_robot - settings.input_limit_lower).array() < 0).any()) {
        std::cout << "u = " << u_robot.transpose() << std::endl;
        std::cout << "Input violated lower limits!" << std::endl;
        return true;
    }
    if (((settings.input_limit_upper - u_robot).array() < 0).any()) {
        std::cout << "u = " << u_robot.transpose() << std::endl;
        std::cout << "Input violated upper limits!" << std::endl;
        return true;
    }
    return false;
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
    // Python interpret required for now because we actually load the control
    // settings and the target trajectories using Python - not ideal but easier
    // than re-implementing the parsing logic in C++ for now
    py::scoped_interpreter guard{};
    ControllerSettings settings = parse_control_settings(config_path);
    std::cout << settings << std::endl;
    ControllerInterface interface(settings);

    // MRT
    ocs2::MRT_ROS_Interface mrt(robot_name);
    mrt.initRollout(&interface.getRollout());
    mrt.launchNodes(nh);

    // nominal initial state and interface to the real robot
    VecXd x0 = interface.getInitialState();

    // Initialize the robot interface
    std::unique_ptr<mm::RobotROSInterface> robot_ptr;
    if (settings.robot_base_type == RobotBaseType::Fixed) {
        robot_ptr.reset(new mm::UR10ROSInterface(nh));
    } else if (settings.robot_base_type == RobotBaseType::Omnidirectional) {
        robot_ptr.reset(new mm::MobileManipulatorROSInterface(nh));
    } else {
        throw std::runtime_error("Unsupported base type.");
    }

    // Initialize interface to dynamic obstacle estimator
    mm::ProjectileROSInterface projectile(nh, "projectile");
    bool avoid_dynamic_obstacle = false;

    ocs2::scalar_t timestep = 1.0 / settings.rate;
    ros::Rate rate(settings.rate);

    // wait until we get feedback from the robot
    while (ros::ok() && ros::master::check()) {
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
    while (ros::ok() && ros::master::check()) {
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
    VecXd ou = VecXd::Zero(settings.dims.u());
    size_t mode = 0;

    VecXd v_ff = VecXd::Zero(settings.dims.robot.v);
    VecXd a_ff = VecXd::Zero(settings.dims.robot.v);

    ros::Time now = ros::Time::now();
    ros::Time last_policy_update_time = now;
    ros::Duration policy_update_delay(0.05);  // TODO note

    ocs2::scalar_t t = now.toSec();
    ocs2::scalar_t last_t = t;

    while (ros::ok() && ros::master::check()) {
        now = ros::Time::now();
        last_t = t;
        t = now.toSec();
        ocs2::scalar_t dt = t - last_t;

        // Re-evaluate the policy to forward simulate the obstacle
        // mrt.evaluatePolicy(t, x, xd, ou, mode);

        // Robot feedback
        VecXd q = robot_ptr->q();
        // VecXd v = robot.get_joint_velocity_raw();

        // Integrate our internal model to get velocity and acceleration
        // "feedback"
        VecXd u = ou.head(settings.dims.robot.u);
        std::tie(v_ff, a_ff) = double_integrate(v_ff, a_ff, u, dt);

        // Current state is built from robot feedback for q and v; for
        // acceleration we just assume we are tracking well since we don't get
        // good feedback on this
        x.head(settings.dims.robot.x) << q, v_ff, a_ff;

        // Dynamic obstacle
        if (settings.dims.o > 0 && projectile.ready()) {
            Vec3d q_obs = projectile.q();
            std::cout << "q_obs = " << q_obs.transpose() << std::endl;
            if (q_obs(2) > 0.5) {  // TODO
                avoid_dynamic_obstacle = true;
            }

            // TODO we could also have this trigger a case where we now assume
            // the trajectory of the object is perfect

            if (avoid_dynamic_obstacle) {
                Vec3d v_obs = projectile.v();
                std::cout << "v_obs = " << v_obs.transpose() << std::endl;
                Vec3d a_obs = settings.obstacle_settings.dynamic_obstacles[0].acceleration;
                x.tail(9) << q_obs, v_obs, a_obs;
            }
        }

        // Compute optimal state and input using current policy
        mrt.evaluatePolicy(t, x, xd, ou, mode);

        // Check that the controller has provided sane values.
        // if (limits_violated(settings, xd, ou)) {
        //     break;
        // }

        if (ros::isShuttingDown()) {
            robot_ptr->brake();
        } else {
            robot_ptr->publish_cmd_vel(v_ff, /* bodyframe = */ false);
        }

        // Send observation to MPC
        observation.time = t;
        observation.state = x;
        observation.input = ou;
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
    // NOTE: doesn't really work because we can't publish after the node is
    // shutdown
    robot_ptr->brake();

    // Successful exit
    return 0;
}
