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
        ocs2::PinocchioInterface interface(
            buildPinocchioInterface(settings, settings.robot_urdf_path));

        CombinedPinocchioMapping<IntegratorPinocchioMapping<ocs2::ad_scalar_t>,
                                 ocs2::ad_scalar_t>
            mapping(settings.dims);

        bool recompile_libraries = true;

        ocs2::PinocchioEndEffectorKinematicsCppAd end_effector_kinematics(
            interface, mapping, {settings.end_effector_link_name},
            settings.dims.x(), settings.dims.u(), "end_effector_kinematics",
            settings.lib_folder, recompile_libraries, false);

        constraints_.reset(new BoundedBalancingConstraints(
            end_effector_kinematics, settings.balancing_settings,
            settings.gravity, settings.dims, recompile_libraries));
    }

    ocs2::PinocchioInterface buildPinocchioInterface(
        const ControllerSettings& settings, const std::string& urdf_path) {
        if (settings.robot_base_type == RobotBaseType::Omnidirectional) {
            // add 3 DOF for wheelbase
            pinocchio::JointModelComposite root_joint(3);
            root_joint.addJoint(pinocchio::JointModelPX());
            root_joint.addJoint(pinocchio::JointModelPY());
            root_joint.addJoint(pinocchio::JointModelRZ());

            return ocs2::getPinocchioInterfaceFromUrdfFile(urdf_path,
                                                           root_joint);
        }
        // Fixed base
        return ocs2::getPinocchioInterfaceFromUrdfFile(urdf_path);
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
