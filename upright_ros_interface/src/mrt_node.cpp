#include <iostream>

#include <ros/init.h>
#include <ros/package.h>

#include <ocs2_mpc/SystemObservation.h>
#include <ocs2_ros_interfaces/common/RosMsgConversions.h>
#include <ocs2_ros_interfaces/mrt/MRT_ROS_Interface.h>

#include <mobile_manipulation_central/robot_interfaces.h>

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
    ControllerSettings settings = parse_control_settings(config_path);
    std::cout << settings << std::endl;
    ControllerInterface interface(settings);

    // MRT
    ocs2::MRT_ROS_Interface mrt(robot_name);
    mrt.initRollout(&interface.getRollout());
    mrt.launchNodes(nh);

    // nominal initial state and interface to the real robot
    VecXd x0 = interface.getInitialState();
    mm::UR10ROSInterface robot(nh);

    ocs2::scalar_t timestep = 1.0 / settings.rate;
    ros::Rate rate(settings.rate);

    // wait until we get feedback from the robot
    while (ros::ok() && ros::master::check()) {
        ros::spinOnce();
        if (robot.ready()) {
            break;
        }
        rate.sleep();
    }
    std::cout << "Received feedback from robot." << std::endl;

    // update to the real initial state
    x0.head(settings.dims.q) = robot.q();

    ocs2::SystemObservation initial_observation;
    initial_observation.state = x0;
    initial_observation.input.setZero(settings.dims.u);
    initial_observation.time = ros::Time::now().toSec();

    // reset MPC
    ocs2::TargetTrajectories target = parse_target_trajectory(config_path, x0);
    mrt.resetMpcNode(target);
    mrt.setCurrentObservation(initial_observation);

    // Let MPC generate the initial plan
    while (ros::ok() && ros::master::check()) {
        mrt.spinMRT();
        mrt.setCurrentObservation(initial_observation);
        if (mrt.initialPolicyReceived()) {
            break;
        }
        rate.sleep();
    }
    mrt.updatePolicy();  // TODO not sure if needed

    std::cout << "Received first policy." << std::endl;

    VecXd x = x0;
    VecXd xd = VecXd::Zero(x.size());
    VecXd u = VecXd::Zero(settings.dims.u);
    size_t mode = 0;
    ocs2::SystemObservation observation = initial_observation;

    VecXd v_ff = VecXd::Zero(settings.dims.v);
    VecXd a_ff = VecXd::Zero(settings.dims.v);

    ros::Time now = ros::Time::now();
    ros::Time last_policy_update_time = now;
    ros::Duration policy_update_delay(0.05);

    ocs2::scalar_t t = now.toSec();
    ocs2::scalar_t last_t = t;

    while (ros::ok() && ros::master::check()) {
        now = ros::Time::now();
        last_t = t;
        t = now.toSec();
        ocs2::scalar_t dt = t - last_t;

        // Robot feedback
        VecXd q = robot.q();
        // VecXd v = robot.get_joint_velocity_raw();

        // Integrate our internal model to get velocity and acceleration
        // "feedback"
        std::tie(v_ff, a_ff) =
            double_integrate(v_ff, a_ff, u.head(settings.dims.v), dt);

        // Current state is built from robot feedback for q and v; for
        // acceleration we just assume we are tracking well since we don't get
        // good feedback on this
        x.head(settings.dims.q) = q;
        x.segment(settings.dims.q, settings.dims.v) =
            v_ff;  // xd.segment(settings.dims.q, settings.dims.v);
        x.tail(settings.dims.v) = a_ff;  // xd.tail(settings.dims.v);

        // Compute optimal state and input using current policy
        mrt.evaluatePolicy(t, x, xd, u, mode);

        // Check that the cntroller has provided sane values.
        if (((xd - settings.state_limit_lower).array() < 0).any()) {
            std::cout << "State violated lower limits!" << std::endl;
            break;
        }
        if (((settings.state_limit_upper - xd).array() < 0).any()) {
            std::cout << "State violated upper limits!" << std::endl;
            break;
        }
        if (((u - settings.input_limit_lower).array() < 0).any()) {
            std::cout << "Input violated lower limits!" << std::endl;
            break;
        }
        if (((settings.input_limit_upper - u).array() < 0).any()) {
            std::cout << "Input violated upper limits!" << std::endl;
            break;
        }

        // PD controller to generate robot command
        // VecXd qd = xd.head(settings.dims.q);
        // VecXd vd = xd.segment(settings.dims.q, settings.dims.v);
        // VecXd v_cmd = Kp * (qd - q) + v_ff;
        // VecXd v_cmd = v_ff;
        robot.publish_cmd_vel(v_ff);

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
    robot.brake();

    // Successful exit
    return 0;
}
