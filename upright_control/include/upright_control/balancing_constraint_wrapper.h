#pragma once

#include <pinocchio/multibody/model.hpp>

#include <ocs2_core/PreComputation.h>
#include <ocs2_pinocchio_interface/PinocchioEndEffectorKinematicsCppAd.h>

#include <upright_control/constraint/bounded_balancing_constraints.h>
#include <upright_control/controller_settings.h>
#include <upright_control/dynamics/system_pinocchio_mapping.h>
#include <upright_control/types.h>
#include <upright_control/util.h>

namespace upright {

// Wrapper to enable access to constraint Jacobian through bindings.
class BalancingConstraintWrapper {
   public:
    BalancingConstraintWrapper(const ControllerSettings& settings) {
        ocs2::PinocchioInterface interface(build_pinocchio_interface(
            settings.robot_urdf_path, settings.robot_base_type,
            settings.locked_joints, settings.base_pose));

        SystemPinocchioMapping<
            TripleIntegratorPinocchioMapping<ocs2::ad_scalar_t>,
            ocs2::ad_scalar_t>
            mapping(settings.dims);

        bool recompile_libraries = true;

        ocs2::PinocchioEndEffectorKinematicsCppAd end_effector_kinematics(
            interface, mapping, {settings.end_effector_link_name},
            settings.dims.x(), settings.dims.u(), "end_effector_kinematics",
            settings.lib_folder, recompile_libraries, false);

        constraints_.reset(new NominalBalancingConstraints(
            end_effector_kinematics, settings.balancing_settings,
            settings.gravity, settings.dims, recompile_libraries));
    }

    ocs2::VectorFunctionLinearApproximation getLinearApproximation(
        ocs2::scalar_t time, const VecXd& state, const VecXd& input) {
        return constraints_->getLinearApproximation(time, state, input,
                                                    precomputation_);
    }

   private:
    std::unique_ptr<NominalBalancingConstraints> constraints_;
    ocs2::PreComputation precomputation_;
};

}  // namespace upright
