#pragma once

#include <ocs2_pinocchio_interface/PinocchioEndEffectorKinematics.h>
#include <ocs2_pinocchio_interface/PinocchioInterface.h>
#include <upright_control/controller_interface.h>
#include <upright_control/dynamics/system_pinocchio_mapping.h>
#include <upright_core/types.h>

#include <iostream>
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/fwd.hpp>

namespace upright {

class SafetyMonitor {
   public:
    SafetyMonitor(const ControllerSettings& settings,
                  const ocs2::PinocchioInterface& pinocchio_interface)
        : settings_(settings), pinocchio_interface_(pinocchio_interface) {
        SystemPinocchioMapping<TripleIntegratorPinocchioMapping<ocs2::scalar_t>,
                               ocs2::scalar_t>
            mapping(settings.dims);
        kinematics_ptr_.reset(new ocs2::PinocchioEndEffectorKinematics(
            pinocchio_interface, mapping, {settings.end_effector_link_name}));
        kinematics_ptr_->setPinocchioInterface(pinocchio_interface_);
    }

    bool state_limits_violated(const VecXd& x) const {
        VecXd x_robot = x.head(settings_.dims.robot.x);

        if (((x_robot - settings_.state_limit_lower).array() <
             -settings_.tracking.state_violation_margin)
                .any()) {
            std::cout << "x = " << x_robot.transpose() << std::endl;
            std::cout << "State violated lower limits!" << std::endl;
            return true;
        }
        if (((settings_.state_limit_upper - x_robot).array() <
             -settings_.tracking.state_violation_margin)
                .any()) {
            std::cout << "x = " << x_robot.transpose() << std::endl;
            std::cout << "State violated upper limits!" << std::endl;
            return true;
        }
        return false;
    }

    bool input_limits_violated(const VecXd& u) const {
        VecXd u_robot = u.head(settings_.dims.robot.u);

        if (((u_robot - settings_.input_limit_lower).array() <
             -settings_.tracking.input_violation_margin)
                .any()) {
            std::cout << "u = " << u_robot.transpose() << std::endl;
            std::cout << "Input violated lower limits!" << std::endl;
            return true;
        }
        if (((settings_.input_limit_upper - u_robot).array() <
             -settings_.tracking.input_violation_margin)
                .any()) {
            std::cout << "u = " << u_robot.transpose() << std::endl;
            std::cout << "Input violated upper limits!" << std::endl;
            return true;
        }
        return false;
    }

    bool end_effector_position_violated(const ocs2::TargetTrajectories& target,
                                        ocs2::scalar_t t, const VecXd& x) {
        VecXd q = VecXd::Zero(settings_.dims.q());
        q.tail(settings_.dims.robot.q) = x.head(settings_.dims.robot.q);

        pinocchio::forwardKinematics(pinocchio_interface_.getModel(),
                                     pinocchio_interface_.getData(), q);
        pinocchio::updateFramePlacements(pinocchio_interface_.getModel(),
                                         pinocchio_interface_.getData());
        Vec3d actual_position = kinematics_ptr_->getPosition(x).front();
        Vec3d desired_position = interpolate_end_effector_pose(t, target).first;

        VecXd position_constraint(6);
        position_constraint
            << desired_position + settings_.xyz_upper - actual_position,
            actual_position - (desired_position + settings_.xyz_lower);

        if (position_constraint.minCoeff() <
            -settings_.tracking.ee_position_violation_margin) {
            std::cerr << "Controller violated position limits!" << std::endl;
            std::cerr << "Controller position = " << actual_position.transpose()
                      << std::endl;
            std::cerr << "Desired position = " << desired_position.transpose()
                      << std::endl;
            std::cerr << "q = " << q.transpose() << std::endl;
            return true;
        }
        return false;
    }

   private:
    ControllerSettings settings_;
    std::unique_ptr<ocs2::PinocchioEndEffectorKinematics> kinematics_ptr_;
    ocs2::PinocchioInterface pinocchio_interface_;
};

}  // namespace upright
