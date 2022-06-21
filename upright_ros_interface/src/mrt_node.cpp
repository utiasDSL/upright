#include <iostream>

#include <ros/init.h>
#include <ros/package.h>
#include <sensor_msgs/JointState.h>
#include <std_msgs/Float64MultiArray.h>

#include <ocs2_mpc/SystemObservation.h>
#include <ocs2_ros_interfaces/common/RosMsgConversions.h>
#include <ocs2_ros_interfaces/mrt/MRT_ROS_Interface.h>

#include <upright_control/MobileManipulatorInterface.h>
#include <upright_ros_interface/ParseControlSettings.h>
#include <upright_ros_interface/ParseTargetTrajectory.h>
#include <upright_ros_interface/parsing.h>

using namespace upright;

// Get feedback and send commands to the robot
class RobotInterface {
   public:
    RobotInterface(ros::NodeHandle& nh, const VecXd& q0, const VecXd& v0) {
        q = q0;
        v = v0;

        feedback_sub_ = nh.subscribe("/ur10_joint_states", 1,
                                     &RobotInterface::feedback_cb, this);

        command_pub_ =
            nh.advertise<std_msgs::Float64MultiArray>("/ur10_cmd_vel", 1, true);
    }

    void feedback_cb(const sensor_msgs::JointState& msg) {
        std::vector<double> qvec = msg.position;
        std::vector<double> vvec = msg.velocity;
        q = parse_vector(qvec);
        v = parse_vector(vvec);
    }

    void send_command(const VecXd& cmd) {
        std_msgs::Float64MultiArray msg;
        // TODO layout?
        msg.data = std::vector<double>(cmd.data(), cmd.data() + cmd.size());
        command_pub_.publish(msg);
    }

    VecXd q;
    VecXd v;

   private:
    ros::Subscriber feedback_sub_;
    ros::Publisher command_pub_;
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
    MobileManipulatorInterface interface(settings);

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
    RobotInterface robot(nh, q0, v0);

    ros::Rate rate(125);

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

    std::cout << "Received first policy." << std::endl;


    MatXd Kp = MatXd::Identity(settings.dims.q, settings.dims.q);

    VecXd x = x0;
    VecXd xd = VecXd::Zero(x.size());
    VecXd u = VecXd::Zero(settings.dims.u);
    size_t mode = 0;
    ocs2::SystemObservation observation = initial_observation;

    while (ros::ok() && ros::master::check()) {
        // Get new policy messages and update the policy if available
        mrt.spinMRT();
        mrt.updatePolicy();

        // Current state is built from robot feedback for q and v; for
        // acceleration we just assume we are tracking well since we don't get
        // good feedback on this
        x.head(settings.dims.q) = robot.q;
        x.segment(settings.dims.q, settings.dims.v) = robot.v;
        x.tail(settings.dims.v) = xd.tail(settings.dims.v);

        // Compute optimal state and input using current policy
        ocs2::scalar_t now = ros::Time::now().toSec();
        mrt.evaluatePolicy(now, x, xd, u, mode);

        // PD controller to generate robot command
        VecXd qd = xd.head(settings.dims.q);
        VecXd vd = xd.segment(settings.dims.q, settings.dims.v);
        VecXd v_cmd = Kp * (qd - robot.q) + vd;
        robot.send_command(v_cmd);

        // Send observation to MPC
        observation.time = now;
        observation.state = x;
        observation.input = u;
        mrt.setCurrentObservation(observation);

        ros::spinOnce();
        rate.sleep();
    }

    // Successful exit
    return 0;
}
