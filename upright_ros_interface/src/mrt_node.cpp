#include <iostream>

#include <ros/init.h>
#include <ros/package.h>
#include <sensor_msgs/JointState.h>
#include <std_msgs/Float64MultiArray.h>

#include <ocs2_mpc/SystemObservation.h>
#include <ocs2_ros_interfaces/common/RosMsgConversions.h>
#include <ocs2_ros_interfaces/mrt/MRT_ROS_Interface.h>

#include <upright_control/controller_interface.h>
#include <upright_ros_interface/ParseControlSettings.h>
#include <upright_ros_interface/ParseTargetTrajectory.h>
#include <upright_ros_interface/parsing.h>

using namespace upright;

// Map joint names to their correct indices in the configuration vector q
std::map<std::string, size_t> JOINT_INDEX_MAP = {
    {"ur10_arm_shoulder_pan_joint", 0}, {"ur10_arm_shoulder_lift_joint", 1},
    {"ur10_arm_elbow_joint", 2},        {"ur10_arm_wrist_1_joint", 3},
    {"ur10_arm_wrist_2_joint", 4},      {"ur10_arm_wrist_3_joint", 5}};

class ExponentialSmoother {
   public:
    ExponentialSmoother(const ocs2::scalar_t& factor, const VecXd& x0)
        : factor_(factor), x_(x0) {}

    VecXd update(const VecXd& x) {
        x_ = factor_ * x + (1 - factor_) * x_;
        return x_;
    }

    VecXd get_estimate() { return x_; }

   private:
    ocs2::scalar_t factor_;
    VecXd x_;
};

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

// Get feedback and send commands to the robot
class RobotInterface {
   public:
    RobotInterface(ros::NodeHandle& nh, const std::string& robot_name,
                   const VecXd& q0, const VecXd& v0,
                   const ocs2::scalar_t& smoothing_factor)
        : q_(q0), v_(v0), velocity_smoother_(smoothing_factor, v0) {
        feedback_sub_ = nh.subscribe(robot_name + "_joint_states", 1,
                                     &RobotInterface::feedback_cb, this);

        command_pub_ = nh.advertise<std_msgs::Float64MultiArray>(
            robot_name + "_cmd_vel", 1, true);
    }

    void feedback_cb(const sensor_msgs::JointState& msg) {
        // Joint name order not necessarily the same as config vector: remap it.
        for (int i = 0; i < msg.name.size(); ++i) {
            size_t j = JOINT_INDEX_MAP.at(msg.name[i]);
            q_[j] = msg.position[i];
            v_[j] = msg.velocity[i];
        }

        feedback_received_ = true;

        velocity_smoother_.update(v_);
    }

    bool has_received_feedback() { return feedback_received_; }

    void send_command(const VecXd& cmd) {
        std_msgs::Float64MultiArray msg;
        // TODO layout?
        // The command has the joints in the correct order (i.e. the one
        // specified in the config file), as opposed to the re-ordered version
        // from the joint state controller feedback above.
        msg.data = std::vector<double>(cmd.data(), cmd.data() + cmd.size());
        command_pub_.publish(msg);
    }

    VecXd get_joint_position() { return q_; }

    VecXd get_joint_velocity_raw() { return v_; }

    VecXd get_joint_velocity_smoothed() {
        return velocity_smoother_.get_estimate();
    }

   private:
    VecXd q_;
    VecXd v_;

    bool feedback_received_ = false;

    ros::Subscriber feedback_sub_;
    ros::Publisher command_pub_;

    ExponentialSmoother velocity_smoother_;
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

    // parse the control settings
    ros::service::waitForService("parse_control_settings");
    upright_ros_interface::ParseControlSettings settings_srv;
    settings_srv.request.config_path = config_path;
    if (!ros::service::call("parse_control_settings", settings_srv)) {
        throw std::runtime_error("Service call for control settings failed.");
    }

    // parse the target trajectory
    ros::service::waitForService("parse_target_trajectory");
    upright_ros_interface::ParseTargetTrajectory target_srv;
    target_srv.request.config_path = config_path;
    if (!ros::service::call("parse_target_trajectory", target_srv)) {
        throw std::runtime_error("Service call for control settings failed.");
    }

    // controller interface
    ControllerSettings settings = parse_control_settings(settings_srv.response);
    std::cout << settings << std::endl;
    ControllerInterface interface(settings);

    // MRT
    ocs2::MRT_ROS_Interface mrt(robot_name);
    mrt.initRollout(&interface.getRollout());
    mrt.launchNodes(nh);

    // nominal initial state and interface to the real robot
    VecXd x0 = interface.getInitialState();
    VecXd q0 = x0.head(settings.dims.q);
    VecXd v0 = x0.segment(settings.dims.q, settings.dims.v);
    RobotInterface robot(nh, robot_name, q0, v0, 0.9);

    ocs2::scalar_t timestep = 1.0 / settings.rate;
    ros::Rate rate(settings.rate);

    // wait until we get feedback from the robot
    while (ros::ok() && ros::master::check()) {
        ros::spinOnce();
        if (robot.has_received_feedback()) {
            break;
        }
        rate.sleep();
    }
    std::cout << "Received feedback from robot." << std::endl;

    // update to the real initial state
    q0 = robot.get_joint_position();
    x0.head(settings.dims.q) = q0;

    // MatXd Kp = settings.Kp;
    // std::cout << "Kp = " << Kp << std::endl;

    ocs2::SystemObservation initial_observation;
    initial_observation.state = x0;
    initial_observation.input.setZero(settings.dims.u);
    initial_observation.time = ros::Time::now().toSec();

    // reset MPC
    ocs2::TargetTrajectories target =
        ocs2::ros_msg_conversions::readTargetTrajectoriesMsg(
            target_srv.response.target_trajectory);
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
        VecXd q = robot.get_joint_position();
        // VecXd v = robot.get_joint_velocity_raw();

        // Integrate our internal model to get velocity and acceleration
        // "feedback"
        std::tie(v_ff, a_ff) = double_integrate(v_ff, a_ff, u, dt);

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
        robot.send_command(v_ff);

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

    // send zero-velocity command to stop the robot when we're done
    v0 = VecXd::Zero(settings.dims.v);
    robot.send_command(v0);

    // Successful exit
    return 0;
}
