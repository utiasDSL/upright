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

#include <vector>

#include <geometry_msgs/Vector3.h>
#include <ros/ros.h>
#include <trajectory_msgs/JointTrajectory.h>
#include <trajectory_msgs/JointTrajectoryPoint.h>

#include <ocs2_mpc/MPC_BASE.h>
#include <ocs2_mpc/MPC_DDP.h>
#include <ocs2_msgs/mpc_flattened_controller.h>
#include <ocs2_msgs/reset.h>
#include <ocs2_ros_interfaces/common/RosMsgConversions.h>
#include <ocs2_ros_interfaces/mpc/MPC_ROS_Interface.h>
#include <ocs2_ros_interfaces/synchronized_module/RosReferenceManager.h>

#include <upright_control/ControllerSettings.h>
#include <upright_control/controller_interface.h>
#include <upright_control/dynamics/Dimensions.h>
#include <upright_control/types.h>
#include <upright_msgs/FloatArray.h>

#include <upright_ros_interface/ParseControlSettings.h>
#include <upright_ros_interface/parsing.h>

using namespace upright;


class Upright_MPC_ROS_Interface : public ocs2::MPC_ROS_Interface {
   public:
    Upright_MPC_ROS_Interface(ocs2::MPC_BASE& mpc, std::string topicPrefix,
                              const RobotDimensions& dims)
        : dims_(dims), ocs2::MPC_ROS_Interface(mpc, topicPrefix) {}

    void launchNodes(ros::NodeHandle& nodeHandle) {
        ROS_INFO_STREAM("MPC node is setting up ...");

        // Observation subscriber
        mpcObservationSubscriber_ = nodeHandle.subscribe(
            topicPrefix_ + "_mpc_observation", 1,
            &Upright_MPC_ROS_Interface::mpcObservationCallback, this,
            ::ros::TransportHints().tcpNoDelay());

        // MPC publisher
        mpcPolicyPublisher_ =
            nodeHandle.advertise<ocs2_msgs::mpc_flattened_controller>(
                topicPrefix_ + "_mpc_policy", 1, true);

        // Joint trajectory publisher
        jointTrajectoryPublisher_ =
            nodeHandle.advertise<trajectory_msgs::JointTrajectory>(
                topicPrefix_ + "_joint_trajectory", 1, true);

        // MPC reset service server
        mpcResetServiceServer_ =
            nodeHandle.advertiseService<Upright_MPC_ROS_Interface,
                                        ocs2_msgs::reset::Request,
                                        ocs2_msgs::reset::Response>(
                topicPrefix_ + "_mpc_reset",
                &Upright_MPC_ROS_Interface::resetMpcCallback, this);

        // display
#ifdef PUBLISH_THREAD
        ROS_INFO_STREAM("Publishing SLQ-MPC messages on a separate thread.");
#endif

        ROS_INFO_STREAM("MPC node is ready.");

        // spin
        spin();
    }

    void mpcObservationCallback(
        const ocs2_msgs::mpc_observation::ConstPtr& msg) {
        std::lock_guard<std::mutex> resetLock(resetMutex_);

        if (!resetRequestedEver_.load()) {
            ROS_WARN_STREAM(
                "MPC should be reset first. Either call "
                "MPC_ROS_Interface::reset() or use the reset service.");
            return;
        }

        // current time, state, input, and subsystem
        const auto currentObservation =
            ocs2::ros_msg_conversions::readObservationMsg(*msg);

        // measure the delay in running MPC
        mpcTimer_.startTimer();

        // run MPC
        bool controllerIsUpdated =
            mpc_.run(currentObservation.time, currentObservation.state);
        if (!controllerIsUpdated) {
            return;
        }
        copyToBuffer(currentObservation);

        // measure the delay for sending ROS messages
        mpcTimer_.endTimer();

        // check MPC delay and solution window compatibility
        ocs2::scalar_t timeWindow = mpc_.settings().solutionTimeWindow_;
        if (mpc_.settings().solutionTimeWindow_ < 0) {
            timeWindow =
                mpc_.getSolverPtr()->getFinalTime() - currentObservation.time;
        }
        if (timeWindow < 2.0 * mpcTimer_.getAverageInMilliseconds() * 1e-3) {
            std::cerr << "WARNING: The solution time window might be shorter "
                         "than the MPC delay!\n";
        }

        // display
        if (mpc_.settings().debugPrint_) {
            std::cerr << '\n';
            std::cerr << "\n### MPC_ROS Benchmarking";
            std::cerr << "\n###   Maximum : "
                      << mpcTimer_.getMaxIntervalInMilliseconds() << "[ms].";
            std::cerr << "\n###   Average : "
                      << mpcTimer_.getAverageInMilliseconds() << "[ms].";
            std::cerr << "\n###   Latest  : "
                      << mpcTimer_.getLastIntervalInMilliseconds() << "[ms]."
                      << std::endl;
        }

#ifdef PUBLISH_THREAD
        std::unique_lock<std::mutex> lk(publisherMutex_);
        readyToPublish_ = true;
        lk.unlock();
        msgReady_.notify_one();

#else
        ocs2_msgs::mpc_flattened_controller mpcPolicyMsg =
            createMpcPolicyMsg(*bufferPrimalSolutionPtr_, *bufferCommandPtr_,
                               *bufferPerformanceIndicesPtr_);
        mpcPolicyPublisher_.publish(mpcPolicyMsg);
#endif

        // TODO this may be more accurate if the currentObservation time is
        // used, so that the controller can account for lag
        ros::Time stamp(currentObservation.time);
        trajectory_msgs::JointTrajectory joint_trajectory_msg =
            create_joint_trajectory_msg(stamp, *bufferPrimalSolutionPtr_);
        jointTrajectoryPublisher_.publish(joint_trajectory_msg);
    }

    void shutdownNode() {
#ifdef PUBLISH_THREAD
        ROS_INFO_STREAM("Shutting down workers ...");

        std::unique_lock<std::mutex> lk(publisherMutex_);
        terminateThread_ = true;
        lk.unlock();

        msgReady_.notify_all();

        if (publisherWorker_.joinable()) {
            publisherWorker_.join();
        }

        ROS_INFO_STREAM("All workers are shut down.");
#endif

        // shutdown publishers
        mpcPolicyPublisher_.shutdown();
        jointTrajectoryPublisher_.shutdown();
    }

    trajectory_msgs::JointTrajectory create_joint_trajectory_msg(
        const ros::Time& stamp, const ocs2::PrimalSolution& primalSolution) {
        trajectory_msgs::JointTrajectory msg;

        // TODO msg.joint_names
        msg.header.stamp = stamp;
        ocs2::scalar_t t0 = msg.header.stamp.toSec();

        size_t N = primalSolution.timeTrajectory_.size();
        for (int i = 0; i < N; ++i) {
            ocs2::scalar_t t = primalSolution.timeTrajectory_[i];

            // Don't include multiple points with the same timestamp. This also
            // filters out points where the time actually decreases, but this
            // should not happen.
            if ((i < N - 1) && (t >= primalSolution.timeTrajectory_[i + 1])) {
                continue;
            }

            VecXd x = primalSolution.stateTrajectory_[i];
            VecXd u = primalSolution.inputTrajectory_[i];

            VecXd q = x.head(dims_.q);
            VecXd v = x.segment(dims_.q, dims_.v);
            VecXd a = x.tail(dims_.v);

            trajectory_msgs::JointTrajectoryPoint point;
            // relative to the header timestamp
            point.time_from_start = ros::Duration(t - t0);
            point.positions =
                std::vector<double>(q.data(), q.data() + q.size());
            point.velocities =
                std::vector<double>(v.data(), v.data() + v.size());
            point.accelerations =
                std::vector<double>(a.data(), a.data() + a.size());
            point.effort = std::vector<double>(
                u.data(), u.data() + u.size());  // jerk (input)
            msg.points.push_back(point);
        }

        return msg;
    }

   protected:
    ::ros::Publisher jointTrajectoryPublisher_;
    RobotDimensions dims_;
};

int main(int argc, char** argv) {
    const std::string robotName = "mobile_manipulator";

    if (argc < 2) {
        throw std::runtime_error("Config path is required.");
    }

    // Initialize ros node
    ros::init(argc, argv, "mpc_node");
    ros::NodeHandle nodeHandle;

    ros::service::waitForService("parse_control_settings");
    upright_ros_interface::ParseControlSettings settings_srv;
    settings_srv.request.config_path = std::string(argv[1]);
    if (!ros::service::call("parse_control_settings", settings_srv)) {
        throw std::runtime_error("Service call for control settings failed.");
    }

    // Robot interface
    ControllerSettings settings = parse_control_settings(settings_srv.response);
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
    Upright_MPC_ROS_Interface mpcNode(*mpcPtr, robotName, settings.dims);
    mpcNode.launchNodes(nodeHandle);

    // Successful exit
    return 0;
}
