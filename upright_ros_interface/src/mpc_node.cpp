#include <vector>

#include <ros/ros.h>

#include <ocs2_mpc/MPC_BASE.h>
#include <ocs2_ros_interfaces/mpc/MPC_ROS_Interface.h>
#include <ocs2_ros_interfaces/synchronized_module/RosReferenceManager.h>

#include <upright_control/controller_settings.h>
#include <upright_control/controller_interface.h>

#include <upright_ros_interface/parsing.h>

using namespace upright;

int main(int argc, char** argv) {
    const std::string robot_name = "mobile_manipulator";

    if (argc < 2) {
        throw std::runtime_error("Config path is required.");
    }

    // Initialize ros node
    ros::init(argc, argv, "mpc_node");
    ros::NodeHandle nh;
    std::string config_path = std::string(argv[1]);

    // Robot interface
    py::scoped_interpreter guard{};
    ControllerSettings settings = parse_control_settings(config_path);
    std::cout << settings << std::endl;
    ControllerInterface interface(settings);

    // ROS ReferenceManager
    std::shared_ptr<ocs2::RosReferenceManager> ros_reference_manager_ptr(
        new ocs2::RosReferenceManager(robot_name,
                                      interface.getReferenceManagerPtr()));
    ros_reference_manager_ptr->subscribe(nh);

    // MPC
    std::unique_ptr<ocs2::MPC_BASE> mpc_ptr = interface.get_mpc();
    mpc_ptr->getSolverPtr()->setReferenceManager(ros_reference_manager_ptr);

    // Launch MPC ROS node
    ocs2::MPC_ROS_Interface mpc_node(*mpc_ptr, robot_name);
    mpc_node.launchNodes(nh);

    // Successful exit
    return 0;
}
