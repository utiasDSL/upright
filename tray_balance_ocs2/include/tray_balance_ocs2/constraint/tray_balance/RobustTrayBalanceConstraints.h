#pragma once

#include <memory>

#include <ocs2_pinocchio_interface/PinocchioEndEffectorKinematicsCppAd.h>
#include <ocs2_robotic_tools/common/SkewSymmetricMatrix.h>
#include <ocs2_robotic_tools/end_effector/EndEffectorKinematics.h>
#include <tray_balance_ocs2/definitions.h>

#include <ocs2_core/constraint/StateInputConstraintCppAd.h>
#include <ocs2_oc/synchronized_module/ReferenceManager.h>

#include <tray_balance_constraints/robust.h>

namespace ocs2 {
namespace mobile_manipulator {

class RobustTrayBalanceConstraints final : public StateInputConstraintCppAd {
   public:
    using ad_quaternion_t =
        PinocchioEndEffectorKinematicsCppAd::ad_quaternion_t;
    using ad_rotmat_t = PinocchioEndEffectorKinematicsCppAd::ad_rotmat_t;

    using ad_vec2_t = Eigen::Matrix<ad_scalar_t, 2, 1>;
    using ad_vec3_t = Eigen::Matrix<ad_scalar_t, 3, 1>;
    using ad_mat3_t = Eigen::Matrix<ad_scalar_t, 3, 3>;

    RobustTrayBalanceConstraints(
        const PinocchioEndEffectorKinematicsCppAd& pinocchioEEKinematics,
        const RobustParameterSet<scalar_t>& params, bool recompileLibraries)
        : StateInputConstraintCppAd(ConstraintOrder::Linear),
          pinocchioEEKinPtr_(pinocchioEEKinematics.clone()),
          params_(params) {
        if (pinocchioEEKinematics.getIds().size() != 1) {
            throw std::runtime_error(
                "[EndEffectorConstraint] endEffectorKinematics has wrong "
                "number of end effector IDs.");
        }
        // initialize everything, mostly the CppAD interface
        // TODO may wish to use parameters here too
        initialize(STATE_DIM, INPUT_DIM, 0, "robust_tray_balance_constraints",
                   "/tmp/ocs2", recompileLibraries, true);
    }

    RobustTrayBalanceConstraints* clone() const override {
        return new RobustTrayBalanceConstraints(*pinocchioEEKinPtr_, params_,
                                                false);
    }

    size_t getNumConstraints(scalar_t time) const override {
        return params_.balls.size() * 3;
    }

    size_t getNumConstraints() const { return getNumConstraints(0); }

   protected:
    ad_vector_t constraintFunction(
        ad_scalar_t time, const ad_vector_t& state, const ad_vector_t& input,
        const ad_vector_t& parameters) const override {
        ad_rotmat_t C_we = pinocchioEEKinPtr_->getOrientationCppAd(state);
        ad_vector_t angular_vel =
            pinocchioEEKinPtr_->getAngularVelocityCppAd(state, input);
        ad_vector_t angular_acc =
            pinocchioEEKinPtr_->getAngularAccelerationCppAd(state, input);
        ad_vector_t linear_acc =
            pinocchioEEKinPtr_->getAccelerationCppAd(state, input);

        std::cerr << "Using robust constraints" << std::endl;

        RobustParameterSet<ad_scalar_t> params = params_.cast<ad_scalar_t>();
        ad_vector_t constraints = robust_balancing_constraints<ad_scalar_t>(
            C_we, angular_vel, linear_acc, angular_acc, params);
        return constraints;
    }

   private:
    RobustTrayBalanceConstraints(const RobustTrayBalanceConstraints& other) =
        default;

    std::unique_ptr<PinocchioEndEffectorKinematicsCppAd> pinocchioEEKinPtr_;
    RobustParameterSet<scalar_t> params_;
};

}  // namespace mobile_manipulator
}  // namespace ocs2
