#pragma once

#include <ocs2_core/PreComputation.h>
#include <ocs2_pinocchio_interface/PinocchioEndEffectorKinematicsCppAd.h>
#include <upright_control/constraint/bounded_balancing_constraints.h>
#include <upright_control/controller_settings.h>
#include <upright_control/dynamics/system_pinocchio_mapping.h>
#include <upright_control/types.h>
#include <upright_control/util.h>

#include <pinocchio/multibody/model.hpp>

namespace upright {

// Wrapper to enable access to constraint Jacobian through bindings.
class BalancingConstraintWrapper {
   public:
    BalancingConstraintWrapper(const ControllerSettings& settings) {
        if (!settings.balancing_settings.enabled) {
            throw std::runtime_error("Balancing settings not enabled.");
        }
        if (!settings.balancing_settings.use_force_constraints) {
            throw std::runtime_error("Contact force constraints not enabled.");
        }

        ocs2::PinocchioInterface interface(build_pinocchio_interface(
            settings.robot_urdf_path, settings.robot_base_type,
            settings.locked_joints, settings.base_pose));

        SystemPinocchioMapping<
            TripleIntegratorPinocchioMapping<ocs2::ad_scalar_t>,
            ocs2::ad_scalar_t>
            mapping(settings.dims);

        ocs2::PinocchioEndEffectorKinematicsCppAd end_effector_kinematics(
            interface, mapping, {settings.end_effector_link_name},
            settings.dims.x(), settings.dims.u(), "end_effector_kinematics",
            settings.lib_folder, settings.recompile_libraries, false);

        contact_constraints_.reset(new ContactForceBalancingConstraints(
            end_effector_kinematics, settings.balancing_settings,
            settings.gravity, settings.dims, settings.recompile_libraries));
        dynamics_constraints_.reset(new ObjectDynamicsConstraints(
            end_effector_kinematics, settings.balancing_settings,
            settings.gravity, settings.dims, settings.recompile_libraries));
    }

    ocs2::VectorFunctionLinearApproximation getLinearApproximation(
        ocs2::scalar_t time, const VecXd& state, const VecXd& input) {
        ocs2::VectorFunctionLinearApproximation a =
            contact_constraints_->getLinearApproximation(time, state, input,
                                                         precomputation_);
        ocs2::VectorFunctionLinearApproximation b =
            dynamics_constraints_->getLinearApproximation(time, state, input,
                                                          precomputation_);

        // Concatenate the two constraint linearizations.
        const size_t nv = a.f.size() + b.f.size();
        ocs2::VectorFunctionLinearApproximation approx(nv, state.size(), 0);
        approx.f << a.f, b.f;
        approx.dfdx << a.dfdx, b.dfdx;
        return approx;
    }

   private:
    std::unique_ptr<ContactForceBalancingConstraints> contact_constraints_;
    std::unique_ptr<ObjectDynamicsConstraints> dynamics_constraints_;
    ocs2::PreComputation precomputation_;
};

}  // namespace upright
