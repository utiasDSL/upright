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

#include <ros/init.h>

#include <sensor_msgs/JointState.h>
#include <tray_balance_msgs/TrayBalanceControllerInfo.h>

#include <ocs2_mpc/MPC_MRT_Interface.h>
#include <ocs2_ros_interfaces/mpc/MPC_ROS_Interface.h>
#include <ocs2_ros_interfaces/mrt/MRT_ROS_Dummy_Loop.h>
#include <ocs2_ros_interfaces/mrt/MRT_ROS_Interface.h>
#include <ocs2_ros_interfaces/synchronized_module/RosReferenceManager.h>

#include "ocs2_mobile_manipulator_modified/MobileManipulatorDummyVisualization.h"
#include "ocs2_mobile_manipulator_modified/MobileManipulatorInterface.h"

using namespace ocs2;
using namespace mobile_manipulator;

class MRTLoop {
   public:
    MRTLoop(MPC_MRT_Interface& mpc_interface, scalar_t frequency,
            SystemObservation& init_observation)
        : mpc_interface_(mpc_interface),
          frequency_(frequency),
          observation_(init_observation) {
        mpc_interface_.setCurrentObservation(observation_);
    }

    void init(::ros::NodeHandle& nh) {
        current_state_sub_ =
            nh.subscribe("/mm/current_state", 1, &MRTLoop::current_state_cb, this);

        control_info_pub_ =
            nh.advertise<tray_balance_msgs::TrayBalanceControllerInfo>(
                "/mm/control_info", 1);
    }

    void current_state_cb(const sensor_msgs::JointState& msg) {
        // Record new state
        // TODO could it be that *updating* the observation like this is bad?
        observation_.time = msg.header.stamp.toSec();
        for (int i = 0; i < NUM_DOFS; ++i) {
            observation_.state(i) = msg.position[i];
            observation_.state(i + NUM_DOFS) = msg.velocity[i];
        }
        state_received_ = false;
    }

    void publish_control_info(const ros::Time& time,
                              const Eigen::VectorXd& state,
                              const Eigen::VectorXd& input) {
        Eigen::VectorXd q = state.head<NUM_DOFS>();
        Eigen::VectorXd v = state.tail<NUM_DOFS>();

        tray_balance_msgs::TrayBalanceControllerInfo control_info_msg;

        control_info_msg.header.stamp = time;
        control_info_msg.joints.name = {"base_x", "base_y", "base_theta",
                                        "arm_1",  "arm_2",  "arm_3",
                                        "arm_4",  "arm_5",  "arm_6"};
        control_info_msg.joints.position =
            std::vector<double>(q.data(), q.data() + q.size());
        control_info_msg.joints.velocity =
            std::vector<double>(v.data(), v.data() + v.size());
        control_info_msg.command =
            std::vector<double>(input.data(), input.data() + input.size());

        control_info_pub_.publish(control_info_msg);
    }

    void loop(const TargetTrajectories& init_target_trajectory) {
        ros::Rate rate(frequency_);
        double dt = 1. / frequency_;

        // observation_.time = ros::Time::now().toSec();
        observation_.time = 0;

        mpc_interface_.resetMpcNode(init_target_trajectory);

        while (ros::ok()) {
            ros::spinOnce();

            // Update controller state.
            mpc_interface_.setCurrentObservation(observation_);

            // run MPC
            mpc_interface_.advanceMpc();

            if (mpc_interface_.initialPolicyReceived()) {
                size_t mode;
                vector_t optimalState, optimalInput;

                // Re-compute policy based on latest state measurement.
                mpc_interface_.updatePolicy();

                // TODO take out when using sim time
                observation_.time += dt;

                // Evaluate the current policy at the current state and time.
                // ros::Time now = ros::Time::now();
                mpc_interface_.evaluatePolicy(observation_.time, observation_.state,
                                              optimalState, optimalInput, mode);

                // We need to update time to keep in sync, even if we don't
                // receive anything from the subscriber. Otherwise, if we
                // receive something in the future, it'll be outside the MPC
                // horizon.
                // observation_.time = now.toSec();

                // For testing without receiving state updates, we can just use
                // the optimal state.
                observation_.state = optimalState;

                publish_control_info(ros::Time(observation_.time), optimalState, optimalInput);

                // matrix_t K;
                // mpc_interface_.getLinearFeedbackGain(now,

                for (auto& observer : observers_) {
                    observer->update(observation_, mpc_interface_.getPolicy(),
                                     mpc_interface_.getCommand());
                }
            }

            rate.sleep();
        }
    }

    void subscribeObservers(
        const std::vector<std::shared_ptr<DummyObserver>>& observers) {
        observers_ = observers;
    }

   private:
    scalar_t frequency_;

    MPC_MRT_Interface& mpc_interface_;
    SystemObservation observation_;
    std::vector<std::shared_ptr<DummyObserver>> observers_;

    ros::Subscriber current_state_sub_;
    ros::Publisher control_info_pub_;

    bool state_received_ = false;
};

int main(int argc, char** argv) {
    const std::string robot_name = "mobile_manipulator";

    // task files
    std::vector<std::string> programArgs{};
    ::ros::removeROSArgs(argc, argv, programArgs);
    if (programArgs.size() <= 1) {
        throw std::runtime_error("No task file specified. Aborting.");
    }
    std::string taskFileFolderName = std::string(programArgs[1]);

    // Initialize ROS node
    ros::init(argc, argv, robot_name + "_mpc_mrt");
    ros::NodeHandle nodeHandle;

    // Robot interface
    MobileManipulatorInterface interface(taskFileFolderName);

    // ROS ReferenceManager
    std::shared_ptr<ocs2::RosReferenceManager> rosReferenceManagerPtr(
        new ocs2::RosReferenceManager(robot_name,
                                      interface.getReferenceManagerPtr()));
    rosReferenceManagerPtr->subscribe(nodeHandle);

    auto mpcPtr = interface.getMpc();
    mpcPtr->getSolverPtr()->setReferenceManager(rosReferenceManagerPtr);

    MPC_MRT_Interface mpcInterface(*mpcPtr);
    mpcInterface.initRollout(&interface.getRollout());

    // auto time = ros::Time::now().toSec();
    scalar_t time = 0;
    auto frequency = interface.mpcSettings().mrtDesiredFrequency_;

    // initial state
    SystemObservation init_observation;
    init_observation.time = time;
    init_observation.state = interface.getInitialState();
    init_observation.input.setZero(INPUT_DIM);

    // initial target pose
    vector_t initTarget(7);
    initTarget.head(3) << 2, 0, 1;
    initTarget.tail(4) << Eigen::Quaternion<scalar_t>(1, 0, 0, 0).coeffs();
    const vector_t zeroInput = vector_t::Zero(mobile_manipulator::INPUT_DIM);
    const TargetTrajectories init_target_trajectory(
        {init_observation.time}, {initTarget}, {zeroInput});

    // TODO API is kind of all over the place atm
    MRTLoop control_loop(mpcInterface, frequency, init_observation);
    control_loop.init(nodeHandle);

    // Visualization
    std::shared_ptr<mobile_manipulator::MobileManipulatorDummyVisualization>
        dummyVisualization(
            new mobile_manipulator::MobileManipulatorDummyVisualization(
                nodeHandle, interface));
    control_loop.subscribeObservers({dummyVisualization});

    control_loop.loop(init_target_trajectory);

    return 0;
}
