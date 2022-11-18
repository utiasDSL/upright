#pragma once

#include <pinocchio/algorithm/model.hpp>
#include <pinocchio/multibody/model.hpp>
#include <pinocchio/parsers/urdf.hpp>

#include <ocs2_pinocchio_interface/PinocchioInterface.h>
#include <ocs2_pinocchio_interface/urdf.h>
#include <urdf_parser/urdf_parser.h>

#include <upright_control/dynamics/base_type.h>

namespace upright {

ocs2::PinocchioInterface build_pinocchio_interface(
    const std::string& urdf_path, const RobotBaseType base_type,
    const std::map<std::string, ocs2::scalar_t>& locked_joints,
    const Vec3d& base_pose) {
    // Load the URDF
    ::urdf::ModelInterfaceSharedPtr urdf_tree =
        ::urdf::parseURDFFile(urdf_path);
    if (urdf_tree == nullptr) {
        throw std::invalid_argument("The file " + urdf_path +
                                    " does not contain a valid URDF model.");
    }

    pinocchio::Model model;
    pinocchio::JointModelComposite root_joint(3);
    root_joint.addJoint(pinocchio::JointModelPX());
    root_joint.addJoint(pinocchio::JointModelPY());
    root_joint.addJoint(pinocchio::JointModelRZ());
    pinocchio::urdf::buildModel(urdf_tree, root_joint, model);

    // Lock the pose of the base at the desired location
    if (base_type == RobotBaseType::Fixed) {
        pinocchio::JointIndex joint_idx = model.getJointId("root_joint");
        std::vector<pinocchio::JointIndex> joint_ids_to_lock = {joint_idx};
        auto q_idx = model.idx_qs[joint_idx];

        VecXd q = VecXd::Zero(model.nq);
        q.segment(q_idx, 3) = base_pose;

        pinocchio::Model reduced_model;
        pinocchio::buildReducedModel(model, joint_ids_to_lock, q,
                                     reduced_model);
        model = reduced_model;
    }

    // Lock joints if applicable
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
        pinocchio::buildReducedModel(model, joint_ids_to_lock, q,
                                     reduced_model);
        model = reduced_model;
    }
    return ocs2::PinocchioInterface(model, urdf_tree);
}

}  // namespace upright
