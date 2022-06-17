#pragma once

#include <ocs2_core/PreComputation.h>
#include <ocs2_core/Types.h>
#include <ocs2_pinocchio_interface/PinocchioEndEffectorKinematicsCppAd.h>
#include <ocs2_pinocchio_interface/urdf.h>
#include <tray_balance_ocs2/constraint/BoundedBalancingConstraints.h>
#include <tray_balance_ocs2/dynamics/FixedBasePinocchioMapping.h>
#include <tray_balance_ocs2/dynamics/MobileManipulatorPinocchioMapping.h>
#include <tray_balance_ocs2/util.h>
#include <pinocchio/multibody/model.hpp>

namespace ocs2 {
namespace mobile_manipulator {

// Wrapper to enable access to constraint Jacobian through bindings.
class BalancingConstraintWrapper {
   public:
    BalancingConstraintWrapper(const ControllerSettings& settings) {
        PinocchioInterface interface(buildPinocchioInterface(
            settings, settings.robot_urdf_path, settings.obstacle_urdf_path));

        std::unique_ptr<PinocchioStateInputMapping<ad_scalar_t>>
            pinocchio_mapping_ptr;
        if (settings.robot_base_type == RobotBaseType::Omnidirectional) {
            pinocchio_mapping_ptr.reset(
                new MobileManipulatorPinocchioMapping<ad_scalar_t>(
                    settings.dims));
        } else {
            pinocchio_mapping_ptr.reset(
                new FixedBasePinocchioMapping<ad_scalar_t>(settings.dims));
        }

        bool recompileLibraries = true;

        PinocchioEndEffectorKinematicsCppAd end_effector_kinematics(
            interface, *pinocchio_mapping_ptr,
            {settings.end_effector_link_name}, settings.dims.x, settings.dims.u,
            "end_effector_kinematics", settings.lib_folder, recompileLibraries,
            false);

        constraints_.reset(new BoundedBalancingConstraints(
            end_effector_kinematics, settings.tray_balance_settings,
            settings.gravity, settings.dims, recompileLibraries));
    }

    PinocchioInterface buildPinocchioInterface(
        const ControllerSettings& settings, const std::string& urdfPath,
        const std::string& obstacle_urdfPath) {
        if (settings.robot_base_type == RobotBaseType::Omnidirectional) {
            // add 3 DOF for wheelbase
            pinocchio::JointModelComposite rootJoint(3);
            rootJoint.addJoint(pinocchio::JointModelPX());
            rootJoint.addJoint(pinocchio::JointModelPY());
            rootJoint.addJoint(pinocchio::JointModelRZ());

            return getPinocchioInterfaceFromUrdfFile(urdfPath, rootJoint);
        }
        // Fixed base
        return getPinocchioInterfaceFromUrdfFile(urdfPath);
    }

    VectorFunctionLinearApproximation getLinearApproximation(
        scalar_t time, const vector_t& state, const vector_t& input) {
        return constraints_->getLinearApproximation(time, state, input,
                                                    precomputation_);
    }

   private:
    std::unique_ptr<BoundedBalancingConstraints> constraints_;
    PreComputation precomputation_;
};

}  // namespace mobile_manipulator
}  // namespace ocs2
