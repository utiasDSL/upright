#pragma once

#include <pinocchio/multibody/model.hpp>

#include <ocs2_core/PreComputation.h>
#include <ocs2_core/Types.h>
#include <ocs2_pinocchio_interface/PinocchioEndEffectorKinematicsCppAd.h>
#include <ocs2_pinocchio_interface/urdf.h>

#include <upright_control/constraint/bounded_balancing_constraints.h>
#include <upright_control/controller_interface.h>
#include <upright_control/dynamics/combined_pinocchio_mapping.h>
// #include <upright_control/dynamics/fixed_base_pinocchio_mapping.h>
// #include <upright_control/dynamics/omnidirectional_pinocchio_mapping.h>

namespace upright {

// Wrapper to enable access to constraint Jacobian through bindings.
class BalancingConstraintWrapper {
   public:
    BalancingConstraintWrapper(const ControllerSettings& settings) {
        ocs2::PinocchioInterface interface(buildPinocchioInterface(
            settings, settings.robot_urdf_path, settings.obstacle_urdf_path));

        std::unique_ptr<ocs2::PinocchioStateInputMapping<ocs2::ad_scalar_t>>
            pinocchio_mapping_ptr(new CombinedPinocchioMapping<
                                  IntegratorPinocchioMapping<ocs2::ad_scalar_t>,
                                  ocs2::ad_scalar_t>(settings.dims));

        // std::unique_ptr<ocs2::PinocchioStateInputMapping<ocs2::ad_scalar_t>>
        //     pinocchio_mapping_ptr;
        // if (settings.robot_base_type == RobotBaseType::Omnidirectional) {
        //     pinocchio_mapping_ptr.reset(
        //         new OmnidirectionalPinocchioMapping<ocs2::ad_scalar_t>(
        //             settings.dims));
        // } else {
        //     pinocchio_mapping_ptr.reset(
        //         new FixedBasePinocchioMapping<ocs2::ad_scalar_t>(
        //             settings.dims));
        // }

        bool recompileLibraries = true;

        ocs2::PinocchioEndEffectorKinematicsCppAd end_effector_kinematics(
            interface, *pinocchio_mapping_ptr,
            {settings.end_effector_link_name}, settings.dims.x(),
            settings.dims.u(), "end_effector_kinematics", settings.lib_folder,
            recompileLibraries, false);

        constraints_.reset(new BoundedBalancingConstraints(
            end_effector_kinematics, settings.balancing_settings,
            settings.gravity, settings.dims, recompileLibraries));
    }

    ocs2::PinocchioInterface buildPinocchioInterface(
        const ControllerSettings& settings, const std::string& urdfPath,
        const std::string& obstacle_urdfPath) {
        if (settings.robot_base_type == RobotBaseType::Omnidirectional) {
            // add 3 DOF for wheelbase
            pinocchio::JointModelComposite rootJoint(3);
            rootJoint.addJoint(pinocchio::JointModelPX());
            rootJoint.addJoint(pinocchio::JointModelPY());
            rootJoint.addJoint(pinocchio::JointModelRZ());

            return ocs2::getPinocchioInterfaceFromUrdfFile(urdfPath, rootJoint);
        }
        // Fixed base
        return ocs2::getPinocchioInterfaceFromUrdfFile(urdfPath);
    }

    ocs2::VectorFunctionLinearApproximation getLinearApproximation(
        ocs2::scalar_t time, const VecXd& state, const VecXd& input) {
        return constraints_->getLinearApproximation(time, state, input,
                                                    precomputation_);
    }

   private:
    std::unique_ptr<BoundedBalancingConstraints> constraints_;
    ocs2::PreComputation precomputation_;
};

}  // namespace upright
