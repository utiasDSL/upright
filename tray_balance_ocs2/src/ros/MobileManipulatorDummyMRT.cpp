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

#include <tray_balance_ocs2/MobileManipulatorDummyVisualization.h>
#include <tray_balance_ocs2/MobileManipulatorInterface.h>
#include <tray_balance_ocs2/MobileManipulatorReferenceTrajectory.h>

#include <ocs2_mpc/SystemObservation.h>
#include <ocs2_ros_interfaces/mrt/MRT_ROS_Dummy_Loop.h>
#include <ocs2_ros_interfaces/mrt/MRT_ROS_Interface.h>

#include <sensor_msgs/JointState.h>

#include <ros/init.h>

using namespace ocs2;
using namespace mobile_manipulator;

class MRTUpdater {
   public:
    MRTUpdater(MRT_ROS_Interface& mrt) : mrt_(mrt) {}

    void subscribe(::ros::NodeHandle& nh) {
        current_state_sub_ = nh.subscribe("/mm/current_state", 1,
                                          &MRTUpdater::current_state_cb, this);
    }

    void current_state_cb(const sensor_msgs::JointState& msg) {
        SystemObservation observation;
        observation.time = msg.header.stamp.toSec();
        observation.state.setZero(STATE_DIM);
        observation.input.setZero(INPUT_DIM);
        for (int i = 0; i < NUM_DOFS; ++i) {
            observation.state(i) = msg.position[i];
            observation.state(i + NUM_DOFS) = msg.velocity[i];
        }
        mrt_.setCurrentObservation(observation);
    }

   private:
    MRT_ROS_Interface& mrt_;
    ros::Subscriber current_state_sub_;
};

int main(int argc, char** argv) {
    const std::string robotName = "mobile_manipulator";

    // task files
    std::vector<std::string> programArgs{};
    ::ros::removeROSArgs(argc, argv, programArgs);
    if (programArgs.size() <= 2) {
        throw std::runtime_error("No task file specified. Aborting.");
    }
    std::string taskFile = std::string(programArgs[1]);
    std::string libraryFolder = std::string(programArgs[2]);

    // Initialize ros node
    ros::init(argc, argv, robotName + "_mrt");
    ros::NodeHandle nodeHandle;

    // Robot Interface
    mobile_manipulator::MobileManipulatorInterface interface(taskFile,
                                                             libraryFolder);

    // MRT
    MRT_ROS_Interface mrt(robotName);
    // mrt.initRollout(&interface.getRollout());
    mrt.launchNodes(nodeHandle);

    // Subscribe for new updates to the robot state.
    MRTUpdater updater(mrt);
    updater.subscribe(nodeHandle);

    // Visualization
    std::shared_ptr<mobile_manipulator::MobileManipulatorDummyVisualization>
        dummyVisualization(
            new mobile_manipulator::MobileManipulatorDummyVisualization(
                nodeHandle, interface, taskFile));

    // Dummy MRT
    MRT_ROS_Dummy_Loop dummy(mrt, interface.mpcSettings().mrtDesiredFrequency_,
                             interface.mpcSettings().mpcDesiredFrequency_);
    dummy.subscribeObservers({dummyVisualization});

    // initial state
    SystemObservation initObservation;
    initObservation.state = interface.getInitialState();
    initObservation.input.setZero(mobile_manipulator::INPUT_DIM);
    initObservation.time = ros::Time::now().toSec();

    // initial target pose
    // NOTE: Eigen quaternions are constructed using (w, x, y, z) but coeffs()
    // returns (x, y, z, w).
    vector_t initTarget =
        make_target(Eigen::Vector3d(0, -2, 1.5), Eigen::Quaterniond(1, 0, 0, 0),
                    Eigen::Vector3d(1, -5, 1));

    vector_t initTarget2 =
        make_target(Eigen::Vector3d(0, -2, 1.5), Eigen::Quaterniond(1, 0, 0, 0),
                    Eigen::Vector3d(1, -5, 1));

    const vector_t zeroInput = vector_t::Zero(mobile_manipulator::INPUT_DIM);
    const TargetTrajectories initTargetTrajectories(
        {initObservation.time, initObservation.time + 0},
        {initTarget, initTarget2}, {zeroInput, zeroInput});

    // Run dummy (loops while ros is ok)
    dummy.run(initObservation, initTargetTrajectories);

    // Successful exit
    return 0;
}