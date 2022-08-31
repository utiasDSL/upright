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
    const std::string robotName = "mobile_manipulator";

    if (argc < 2) {
        throw std::runtime_error("Config path is required.");
    }

    // Initialize ros node
    ros::init(argc, argv, "mpc_node");
    ros::NodeHandle nodeHandle;
    std::string config_path = std::string(argv[1]);

    // Robot interface
    py::scoped_interpreter guard{};
    ControllerSettings settings = parse_control_settings(config_path);
    std::cout << settings << std::endl;
    ControllerInterface interface(settings);

    // ROS ReferenceManager
    std::shared_ptr<ocs2::RosReferenceManager> rosReferenceManagerPtr(
        new ocs2::RosReferenceManager(robotName,
                                      interface.getReferenceManagerPtr()));
    rosReferenceManagerPtr->subscribe(nodeHandle);

    // MPC
    std::unique_ptr<ocs2::MPC_BASE> mpcPtr = interface.getMpc();
    mpcPtr->getSolverPtr()->setReferenceManager(rosReferenceManagerPtr);

    // Launch MPC ROS node
    ocs2::MPC_ROS_Interface mpcNode(*mpcPtr, robotName);
    mpcNode.launchNodes(nodeHandle);

    // Successful exit
    return 0;
}
