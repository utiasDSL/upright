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

// Get feedback and send commands to the robot
class RobotInterface {
   public:
    RobotInterface(ros::NodeHandle& nh, const VecXd& q0, const VecXd& v0,
                   const ocs2::scalar_t& smoothing_factor)
        : q_(q0), v_(v0), velocity_smoother_(smoothing_factor, v0) {
        // TODO avoid fixed names
        feedback_sub_ = nh.subscribe("/ur10_joint_states", 1,
                                     &RobotInterface::feedback_cb, this);

        command_pub_ =
            nh.advertise<std_msgs::Float64MultiArray>("/ur10_cmd_vel", 1, true);
    }

    void feedback_cb(const sensor_msgs::JointState& msg) {
        std::vector<double> qvec = msg.position;
        std::vector<double> vvec = msg.velocity;
        q_ = parse_vector(qvec);
        v_ = parse_vector(vvec);

        velocity_smoother_.update(v_);
    }

    void send_command(const VecXd& cmd) {
        std_msgs::Float64MultiArray msg;
        // TODO layout?
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

    ros::Subscriber feedback_sub_;
    ros::Publisher command_pub_;

    ExponentialSmoother velocity_smoother_;
};

int main(int argc, char** argv) {
    const std::string robotName = "mobile_manipulator";

    if (argc < 2) {
        throw std::runtime_error("Config path is required.");
    }

    // Initialize ros node
    ros::init(argc, argv, robotName + "_mrt");
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
    ocs2::MRT_ROS_Interface mrt(robotName);
    mrt.initRollout(&interface.getRollout());
    mrt.launchNodes(nh);

    // initial state
    VecXd x0 = interface.getInitialState();
    ocs2::SystemObservation initial_observation;
    initial_observation.state = x0;
    initial_observation.input.setZero(settings.dims.u);
    initial_observation.time = ros::Time::now().toSec();

    // interface to the real robot
    VecXd q0 = x0.head(settings.dims.q);
    VecXd v0 = x0.segment(settings.dims.q, settings.dims.v);
    RobotInterface robot(nh, q0, v0, 0.9);

    ros::Rate rate(settings.rate);

    MatXd Kp = settings.Kp;
    std::cout << "Kp = " << Kp << std::endl;

    // Reset MPC
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

    // TODO experimental
    ros::Time now = ros::Time::now();
    ros::Time last_policy_update_time = now;
    ros::Duration policy_update_delay(0.1);

    while (ros::ok() && ros::master::check()) {
        now = ros::Time::now();
        ocs2::scalar_t t = now.toSec();

        // Robot feedback
        VecXd q = robot.get_joint_position();
        VecXd v = robot.get_joint_velocity_raw();

        // Get new policy messages and update the policy if available
        mrt.spinMRT();
        if (now - last_policy_update_time >= policy_update_delay) {
            mrt.updatePolicy();
            last_policy_update_time = now;
        }

        // Current state is built from robot feedback for q and v; for
        // acceleration we just assume we are tracking well since we don't get
        // good feedback on this
        // NOTE: this should only affect u, not xd
        x.head(settings.dims.q) = q;
        x.segment(settings.dims.q, settings.dims.v) = xd.segment(settings.dims.q, settings.dims.v);
        x.tail(settings.dims.v) = xd.tail(settings.dims.v);

        // Compute optimal state and input using current policy
        mrt.evaluatePolicy(t, x, xd, u, mode);

        // PD controller to generate robot command
        VecXd qd = xd.head(settings.dims.q);
        VecXd vd = xd.segment(settings.dims.q, settings.dims.v);
        VecXd v_cmd = Kp * (qd - q) + vd;
        robot.send_command(v_cmd);

        // Also assume velocity is fine
        // TODO why does this lead to more jittering?
        // the sim is actually just slow!
        x.segment(settings.dims.q, settings.dims.v) = vd;

        // Send observation to MPC
        observation.time = t;
        observation.state = x;
        observation.input = u;
        mrt.setCurrentObservation(observation);

        ros::spinOnce();
        rate.sleep();
    }

    // Successful exit
    return 0;
}
