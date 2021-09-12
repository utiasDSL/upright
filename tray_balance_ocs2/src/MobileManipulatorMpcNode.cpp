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

#include <ocs2_ros_interfaces/mpc/MPC_ROS_Interface.h>
#include <ocs2_ros_interfaces/synchronized_module/RosReferenceManager.h>

#include "ocs2_mobile_manipulator_modified/MobileManipulatorInterface.h"

#include <tray_balance_constraints/inequality_constraints.h>

using namespace ocs2;
using namespace mobile_manipulator;

int main(int argc, char** argv) {
    const std::string robotName = "mobile_manipulator";

    // task file
    std::vector<std::string> programArgs{};
    ::ros::removeROSArgs(argc, argv, programArgs);
    if (programArgs.size() <= 1) {
        throw std::runtime_error("No task file specified. Aborting.");
    }
    std::string taskFileFolderName = std::string(programArgs[1]);

    // Initialize ros node
    ros::init(argc, argv, robotName + "_mpc");
    ros::NodeHandle nodeHandle;

    // Robot interface
    MobileManipulatorInterface interface(taskFileFolderName);

    // Ros ReferenceManager
    std::shared_ptr<ocs2::RosReferenceManager> rosReferenceManagerPtr(
        new ocs2::RosReferenceManager(robotName,
                                      interface.getReferenceManagerPtr()));
    rosReferenceManagerPtr->subscribe(nodeHandle);

    // scalar_t obj_mass(1.0);
    // scalar_t obj_mu(0.5);
    // scalar_t obj_com_height(0.2);
    // scalar_t obj_zmp_margin(0.01);
    // Eigen::Vector2d obj_support_offset = Eigen::Vector2d::Zero();
    //
    // // cuboid-specific params
    // Eigen::Vector3d cuboid_side_lengths(0.2, 0.2, obj_com_height * 2);
    // Eigen::Matrix3d cuboid_inertia =
    //     cuboid_inertia_matrix(obj_mass, cuboid_side_lengths);
    // // NOTE: this assumes that the cuboid is -0.05 offset
    // Eigen::Vector3d cuboid_com(-0.05, 0, 0.25);
    // RigidBody<scalar_t> cuboid_body(obj_mass, cuboid_inertia, cuboid_com);
    // scalar_t cuboid_r_tau = circle_r_tau(cuboid_side_lengths(0) * 0.5);  //
    // TODO
    //
    // std::vector<Eigen::Vector2d> vertices =
    //     cuboid_support_vertices(cuboid_side_lengths);
    // PolygonSupportArea<scalar_t> cuboid_support_area(
    //     vertices, obj_support_offset, obj_zmp_margin);
    // // CircleSupportArea<ad_scalar_t> cuboid_support_area(
    // //     ad_scalar_t(0.1), obj_support_offset, obj_zmp_margin);
    //
    // BalancedObject<scalar_t> cuboid(
    //     cuboid_body, obj_com_height, cuboid_support_area, cuboid_r_tau,
    //     obj_mu);
    //
    // // vector_t constraints = balancing_constraints<scalar_t>(
    // //     C_we, angular_vel, linear_acc, angular_acc, {cuboid});
    // Eigen::Vector2d zmp;
    // zmp << 0.01, 0;
    // vector_t zmp_cons = cuboid.support_area_ptr->zmp_constraints(zmp);
    //
    // std::cout << "cuboid ZMP constraints = " << zmp_cons << std::endl;
    // throw std::runtime_error("stop!");

    // scalar_t w = 0.1;
    // scalar_t h = 0.1;
    // scalar_t r = 0.5 * w;
    // std::cout << "circle r_tau = " << circle_r_tau(r) << std::endl;
    // std::cout << "rectangle r_tau = " << rectangle_r_tau(w, h) << std::endl;
    // std::cout << "rectangle r_tau = " << rectangle_r_tau(h, w) << std::endl;

    // std::cout << "loc 1 = " << equilateral_triangle_cup_location(0.2, 0.08, 0) << std::endl;
    // std::cout << "loc 2 = " << equilateral_triangle_cup_location(0.2, 0.08, 1) << std::endl;
    // std::cout << "loc 3 = " << equilateral_triangle_cup_location(0.2, 0.08, 2) << std::endl;

    // throw std::runtime_error("stop!");

    // Launch MPC ROS node
    auto mpcPtr = interface.getMpc();
    mpcPtr->getSolverPtr()->setReferenceManager(rosReferenceManagerPtr);
    MPC_ROS_Interface mpcNode(*mpcPtr, robotName);
    mpcNode.launchNodes(nodeHandle);

    return 0;
}
