#include <controller_manager/controller_manager.h>
#include <geometry_msgs/Twist.h>
#include <hardware_interface/joint_command_interface.h>
#include <hardware_interface/joint_state_interface.h>
#include <hardware_interface/robot_hw.h>
#include <nav_msgs/Odometry.h>
#include <ros/ros.h>
#include <tf/transform_datatypes.h>

const size_t NUM_JOINTS = 3;
const std::vector<std::string> JOINT_NAMES = {"x", "y", "angle"};
const double HZ = 100.0;

class TwistInterface : public hardware_interface::RobotHW {
   public:
    TwistInterface(ros::NodeHandle& nh) {
        // connect and register the joint state interface
        for (int i = 0; i < NUM_JOINTS; ++i) {
            hardware_interface::JointStateHandle state_handle(
                JOINT_NAMES[i], &pos[i], &vel[i], &eff[i]);
            jnt_state_interface.registerHandle(state_handle);
        }
        registerInterface(&jnt_state_interface);

        // connect and register the joint velocity interface
        for (int i = 0; i < NUM_JOINTS; ++i) {
            hardware_interface::JointHandle vel_handle(
                jnt_state_interface.getHandle(JOINT_NAMES[i]), &cmd[i]);
            jnt_vel_interface.registerHandle(vel_handle);
        }
        registerInterface(&jnt_vel_interface);

        // Efforts are always zero, since we have no feedback here.
        for (int i = 0; i < NUM_JOINTS; ++i) {
            eff[i] = 0;
        }

        cmd_pub = nh.advertise<geometry_msgs::Twist>("/cmd_vel", 1);

        odom_sub = nh.subscribe("/odometry/filtered", 1,
                                &TwistInterface::odom_cb, this);
    }

    void odom_cb(const nav_msgs::Odometry& msg) {
        pos[0] = msg.pose.pose.position.x;
        pos[1] = msg.pose.pose.position.y;
        pos[2] = tf::getYaw(msg.pose.pose.orientation);

        vel[0] = msg.twist.twist.linear.x;
        vel[1] = msg.twist.twist.linear.y;
        vel[2] = msg.twist.twist.angular.z;
    }

    void publish_cmd() {
        geometry_msgs::Twist msg;
        msg.linear.x = eff[0];
        msg.linear.y = eff[1];
        msg.angular.z = eff[2];
        cmd_pub.publish(msg);
    }

   private:
    hardware_interface::JointStateInterface jnt_state_interface;
    hardware_interface::VelocityJointInterface jnt_vel_interface;
    double cmd[NUM_JOINTS];
    double pos[NUM_JOINTS];
    double vel[NUM_JOINTS];
    double eff[NUM_JOINTS];

    // Publish commands
    ros::Publisher cmd_pub;

    // Subscribe to odometry for localization
    ros::Subscriber odom_sub;
};

int main(int argc, char** argv) {
    ros::init(argc, argv, "twist_interface_node");
    ros::NodeHandle nh;

    TwistInterface robot(nh);
    controller_manager::ControllerManager cm(&robot);
    ros::Rate rate(HZ);

    ros::Time last_time = ros::Time::now();

    while (ros::ok()) {
        ros::spinOnce();
        ros::Time now = ros::Time::now();
        cm.update(now, now - last_time);
        last_time = now;
        robot.publish_cmd();
        rate.sleep();
    }
}
