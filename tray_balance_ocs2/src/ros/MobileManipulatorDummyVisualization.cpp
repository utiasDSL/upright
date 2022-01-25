/******************************************************************************
Copyright (c) 2020, Farbod Farshidian. All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

 * Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

 * Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

 * Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 ******************************************************************************/

#include <pinocchio/fwd.hpp>

#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/kinematics.hpp>

#include <ros/package.h>
#include <tf/tf.h>
#include <urdf/model.h>
#include <kdl_parser/kdl_parser.hpp>

#include <geometry_msgs/PoseArray.h>
#include <visualization_msgs/MarkerArray.h>
// #include <sensor_msgs/JointState.h>

#include <ocs2_ros_interfaces/common/RosMsgHelpers.h>

#include <tray_balance_ocs2/MobileManipulatorDummyVisualization.h>
#include <tray_balance_ocs2/MobileManipulatorInterface.h>
#include <tray_balance_ocs2/MobileManipulatorReferenceTrajectory.h>
#include <tray_balance_ocs2/definitions.h>
#include <tray_balance_ocs2/util.h>

#include <tray_balance_msgs/TrayBalanceControllerInfo.h>

namespace ocs2 {
namespace mobile_manipulator {

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
Eigen::VectorXd getArmJointPositions(Eigen::VectorXd state) {
    return state.segment<6>(3);
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
Eigen::Vector3d getBasePosition(Eigen::VectorXd state) {
    Eigen::Vector3d position;
    position << state(0), state(1), 0.0;
    return position;
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
Eigen::Quaterniond getBaseOrientation(Eigen::VectorXd state) {
    return Eigen::Quaterniond(
        Eigen::AngleAxisd(state(2), Eigen::Vector3d::UnitZ()));
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
template <typename It>
void assignHeader(It firstIt, It lastIt, const std_msgs::Header& header) {
    for (; firstIt != lastIt; ++firstIt) {
        firstIt->header = header;
    }
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
template <typename It>
void assignIncreasingId(It firstIt, It lastIt, int startId = 0) {
    for (; firstIt != lastIt; ++firstIt) {
        firstIt->id = startId++;
    }
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
void MobileManipulatorDummyVisualization::launchVisualizerNode(
    ros::NodeHandle& nodeHandle, const std::string& taskFile) {
    // load a kdl-tree from the urdf robot description and initialize the robot
    // state publisher
    const std::string urdfName = "robot_description";
    urdf::Model model;
    if (!model.initParam(urdfName)) {
        ROS_ERROR("URDF model load was NOT successful");
    }
    KDL::Tree tree;
    if (!kdl_parser::treeFromUrdfModel(model, tree)) {
        ROS_ERROR("Failed to extract kdl tree from xml robot description");
    }

    robotStatePublisherPtr_.reset(
        new robot_state_publisher::RobotStatePublisher(tree));
    robotStatePublisherPtr_->publishFixedTransforms(true);

    stateOptimizedPublisher_ =
        nodeHandle.advertise<visualization_msgs::MarkerArray>(
            "/mobile_manipulator/optimizedStateTrajectory", 1);
    stateOptimizedPosePublisher_ =
        nodeHandle.advertise<geometry_msgs::PoseArray>(
            "/mobile_manipulator/optimizedPoseTrajectory", 1);
    constrolInfoPublisher_ =
        nodeHandle.advertise<tray_balance_msgs::TrayBalanceControllerInfo>(
            "/mm/control_info", 1);

    std::string robot_urdf_path, obstacle_urdf_path;
    std::tie(robot_urdf_path, obstacle_urdf_path) = load_urdf_paths(taskFile);

    PinocchioInterface pinocchioInterface =
        MobileManipulatorInterface::buildPinocchioInterface(robot_urdf_path,
                                                            obstacle_urdf_path);
    // TODO(perry) get the collision pairs from the task.info file to match the
    // current mpc setup
    // TODO need to get the real geom interface
    // PinocchioGeometryInterface geomInterface(pinocchioInterface,
    //                                          {{1, 4}, {1, 6}});

    PinocchioGeometryInterface geometryInterface(pinocchioInterface);
    pinocchio::GeometryModel obs_geom_model =
        MobileManipulatorInterface::build_geometry_model(obstacle_urdf_path);
    geometryInterface.addGeometryObjects(obs_geom_model);

    geometryVisualization_.reset(new GeometryInterfaceVisualization(
        std::move(pinocchioInterface), geometryInterface, nodeHandle));
}

/******************************************************************************************************/ /******************************************************************************************************/
/******************************************************************************************************/
void MobileManipulatorDummyVisualization::update(
    const SystemObservation& observation, const PrimalSolution& policy,
    const CommandData& command) {
    const ros::Time timeStamp = ros::Time::now();

    publishObservation(timeStamp, observation);
    publishTargetTrajectories(timeStamp, command.mpcTargetTrajectories_);
    publishOptimizedTrajectory(timeStamp, policy);
    geometryVisualization_->publishDistances(observation.state);
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
void MobileManipulatorDummyVisualization::publishObservation(
    const ros::Time& timeStamp, const SystemObservation& observation) {
    // publish world -> base transform
    const auto position = getBasePosition(observation.state);
    const auto orientation = getBaseOrientation(observation.state);

    geometry_msgs::TransformStamped base_tf;
    base_tf.header.stamp = timeStamp;
    base_tf.header.frame_id = "world";
    base_tf.child_frame_id = "base_link";
    base_tf.transform.translation = ros_msg_helpers::getVectorMsg(position);
    base_tf.transform.rotation =
        ros_msg_helpers::getOrientationMsg(orientation);
    tfBroadcaster_.sendTransform(base_tf);

    // publish joints transforms
    const auto j_arm = getArmJointPositions(observation.state);
    std::map<std::string, scalar_t> jointPositions{
        {"ur10_arm_shoulder_pan_joint", j_arm(0)},
        {"ur10_arm_shoulder_lift_joint", j_arm(1)},
        {"ur10_arm_elbow_joint", j_arm(2)},
        {"ur10_arm_wrist_1_joint", j_arm(3)},
        {"ur10_arm_wrist_2_joint", j_arm(4)},
        {"ur10_arm_wrist_3_joint", j_arm(5)}};
    robotStatePublisherPtr_->publishTransforms(jointPositions, timeStamp);

    // const pinocchio::ReferenceFrame rf =
    //     pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED;
    // const auto& model = pinocchioInterface_.getModel();
    // auto& data = pinocchioInterface_.getData();
    //
    // pinocchio::forwardKinematics(
    //     model, data, observation.state.head<NUM_DOFS>(),
    //     observation.state.tail<NUM_DOFS>(), observation.input);
    // pinocchio::updateFramePlacements(model, data);
    // const auto eeIndex = model.getBodyId("thing_tool");  // WRIST_2
    //
    // const Mat3<scalar_t> C_we = data.oMf[eeIndex].rotation();
    // const Vec3<scalar_t> angular_vel =
    //     pinocchio::getFrameVelocity(model, data, eeIndex, rf).angular();
    // const Vec3<scalar_t> linear_acc =
    //     pinocchio::getFrameClassicalAcceleration(model, data, eeIndex, rf).linear();
    // const Vec3<scalar_t> angular_acc =
    //     pinocchio::getFrameClassicalAcceleration(model, data, eeIndex, rf).angular();
    //
    // BalancedObject<scalar_t> tray = build_tray_object<scalar_t>();
    //
    // vector_t zmp = compute_zmp(C_we, angular_vel, linear_acc, angular_acc, tray);
    //
    // outfile << "t = " << observation.time << std::endl;
    // outfile << "x = " << observation.state.transpose() << std::endl;
    // outfile << "u = " << observation.input.transpose() << std::endl;
    // outfile << "C_we = " << C_we << std::endl;
    // outfile << "angular_vel = " << angular_vel.transpose() << std::endl;
    // outfile << "linear_acc = " << linear_acc.transpose() << std::endl;
    // outfile << "angular_acc = " << angular_acc.transpose() << std::endl;
    // outfile << "zmp = " << zmp.transpose() << std::endl << std::endl;

    // publish info about the controller
    // TODO it would be better to have this in some interface instead
    // Eigen::VectorXd q = observation.state.head<NUM_DOFS>();
    // Eigen::VectorXd v = observation.state.tail<NUM_DOFS>();
    // Eigen::VectorXd u = observation.input;
    //
    // tray_balance_msgs::TrayBalanceControllerInfo control_info_msg;
    //
    // control_info_msg.header.stamp = ros::Time::now();
    // control_info_msg.joints.name = {"base_x", "base_y", "base_theta",
    //                                 "arm_1",  "arm_2",  "arm_3",
    //                                 "arm_4",  "arm_5",  "arm_6"};
    // control_info_msg.joints.position =
    //     std::vector<double>(q.data(), q.data() + q.size());
    // control_info_msg.joints.velocity =
    //     std::vector<double>(v.data(), v.data() + v.size());
    // control_info_msg.command =
    //     std::vector<double>(u.data(), u.data() + u.size());
    //
    // constrolInfoPublisher_.publish(control_info_msg);
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
void MobileManipulatorDummyVisualization::publishTargetTrajectories(
    const ros::Time& timeStamp, const TargetTrajectories& targetTrajectories) {
    // publish command transform
    const Eigen::Vector3d eeDesiredPosition =
        get_target_position(targetTrajectories.stateTrajectory.back());
    Eigen::Quaterniond eeDesiredOrientation =
        get_target_orientation(targetTrajectories.stateTrajectory.back());
    geometry_msgs::TransformStamped command_tf;
    command_tf.header.stamp = timeStamp;
    command_tf.header.frame_id = "world";
    command_tf.child_frame_id = "command";
    command_tf.transform.translation =
        ros_msg_helpers::getVectorMsg(eeDesiredPosition);
    command_tf.transform.rotation =
        ros_msg_helpers::getOrientationMsg(eeDesiredOrientation);
    tfBroadcaster_.sendTransform(command_tf);
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
void MobileManipulatorDummyVisualization::publishOptimizedTrajectory(
    const ros::Time& timeStamp, const PrimalSolution& policy) {
    const scalar_t TRAJECTORYLINEWIDTH = 0.005;
    const std::array<scalar_t, 3> red{0.6350, 0.0780, 0.1840};
    const std::array<scalar_t, 3> blue{0, 0.4470, 0.7410};
    const auto& mpcStateTrajectory = policy.stateTrajectory_;

    visualization_msgs::MarkerArray markerArray;

    // Base trajectory
    std::vector<geometry_msgs::Point> baseTrajectory;
    baseTrajectory.reserve(mpcStateTrajectory.size());
    geometry_msgs::PoseArray poseArray;
    poseArray.poses.reserve(mpcStateTrajectory.size());

    // End effector trajectory
    const auto& model = pinocchioInterface_.getModel();
    auto& data = pinocchioInterface_.getData();

    std::vector<geometry_msgs::Point> endEffectorTrajectory;
    endEffectorTrajectory.reserve(mpcStateTrajectory.size());
    std::for_each(
        mpcStateTrajectory.begin(), mpcStateTrajectory.end(),
        [&](const Eigen::VectorXd& state) {
            pinocchio::forwardKinematics(model, data, state.head<NUM_DOFS>());
            pinocchio::updateFramePlacements(model, data);
            const auto eeIndex = model.getBodyId("thing_tool");  // WRIST_2
            const vector_t eePosition = data.oMf[eeIndex].translation();
            endEffectorTrajectory.push_back(
                ros_msg_helpers::getPointMsg(eePosition));
        });

    markerArray.markers.emplace_back(ros_msg_helpers::getLineMsg(
        std::move(endEffectorTrajectory), blue, TRAJECTORYLINEWIDTH));
    markerArray.markers.back().ns = "EE Trajectory";

    // Extract base pose from state
    std::for_each(mpcStateTrajectory.begin(), mpcStateTrajectory.end(),
                  [&](const vector_t& state) {
                      geometry_msgs::Pose pose;
                      pose.position =
                          ros_msg_helpers::getPointMsg(getBasePosition(state));
                      pose.orientation = ros_msg_helpers::getOrientationMsg(
                          getBaseOrientation(state));
                      baseTrajectory.push_back(pose.position);
                      poseArray.poses.push_back(std::move(pose));
                  });

    markerArray.markers.emplace_back(ros_msg_helpers::getLineMsg(
        std::move(baseTrajectory), red, TRAJECTORYLINEWIDTH));
    markerArray.markers.back().ns = "Base Trajectory";

    assignHeader(markerArray.markers.begin(), markerArray.markers.end(),
                 ros_msg_helpers::getHeaderMsg("world", timeStamp));
    assignIncreasingId(markerArray.markers.begin(), markerArray.markers.end());
    poseArray.header = ros_msg_helpers::getHeaderMsg("world", timeStamp);

    stateOptimizedPublisher_.publish(markerArray);
    stateOptimizedPosePublisher_.publish(poseArray);
}

}  // namespace mobile_manipulator
}  // namespace ocs2