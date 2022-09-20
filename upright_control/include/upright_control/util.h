#pragma once

#include <pinocchio/multibody/model.hpp>

#include <ocs2_pinocchio_interface/PinocchioInterface.h>
#include <ocs2_pinocchio_interface/urdf.h>

#include <upright_control/dynamics/base_type.h>

namespace upright {

ocs2::PinocchioInterface build_pinocchio_interface(
    const std::string& urdf_path, const RobotBaseType base_type) {
    if (base_type == RobotBaseType::Omnidirectional) {
        // add 3 DOF for wheelbase
        pinocchio::JointModelComposite root_joint(3);
        root_joint.addJoint(pinocchio::JointModelPX());
        root_joint.addJoint(pinocchio::JointModelPY());
        root_joint.addJoint(pinocchio::JointModelRZ());
        // root_joint.addJoint(pinocchio::JointModelRUBZ());

        return ocs2::getPinocchioInterfaceFromUrdfFile(urdf_path, root_joint);
    }
    // Fixed base
    return ocs2::getPinocchioInterfaceFromUrdfFile(urdf_path);
}

}  // namespace upright
