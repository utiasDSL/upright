#pragma once

#include <pinocchio/multibody/model.hpp>
#include <pinocchio/parsers/urdf.hpp>
#include <pinocchio/algorithm/model.hpp>

#include <ocs2_pinocchio_interface/PinocchioInterface.h>
#include <ocs2_pinocchio_interface/urdf.h>
#include <urdf_parser/urdf_parser.h>

#include <upright_control/dynamics/base_type.h>

namespace upright {

ocs2::PinocchioInterface build_pinocchio_interface(
    const std::string& urdf_path, const RobotBaseType base_type,
    const std::map<std::string, ocs2::scalar_t>& locked_joints) {
    pinocchio::Model model;

    // Load the URDF
    ::urdf::ModelInterfaceSharedPtr urdf_tree =
        ::urdf::parseURDFFile(urdf_path);
    if (urdf_tree == nullptr) {
        throw std::invalid_argument("The file " + urdf_path +
                                    " does not contain a valid URDF model.");
    }

    if (base_type == RobotBaseType::Omnidirectional) {
        // add 3 DOF for wheelbase
        pinocchio::JointModelComposite root_joint(3);
        root_joint.addJoint(pinocchio::JointModelPX());
        root_joint.addJoint(pinocchio::JointModelPY());
        root_joint.addJoint(pinocchio::JointModelRZ());
        // root_joint.addJoint(pinocchio::JointModelRUBZ());

        pinocchio::urdf::buildModel(urdf_tree, root_joint, model);
        // return ocs2::getPinocchioInterfaceFromUrdfFile(urdf_path,
        // root_joint);
    } else {
        // Fixed base
        // return ocs2::getPinocchioInterfaceFromUrdfFile(urdf_path);
        pinocchio::urdf::buildModel(urdf_tree, model);
    }

    // TODO need to deal with fixed joints
    if (locked_joints.size() > 0) {
        VecXd q = VecXd::Zero(model.nq);
        std::vector<pinocchio::JointIndex> joint_ids_to_lock;
        for (const auto& kv : locked_joints) {
            pinocchio::JointIndex joint_idx = model.getJointId(kv.first);
            joint_ids_to_lock.push_back(joint_idx);
            auto q_idx = model.idx_qs[joint_idx];
            q(q_idx) = kv.second;
        }
        pinocchio::Model reduced_model;
        pinocchio::buildReducedModel(model, joint_ids_to_lock, q, reduced_model);
        model = reduced_model;
    }
    return ocs2::PinocchioInterface(model, urdf_tree);
}

}  // namespace upright
