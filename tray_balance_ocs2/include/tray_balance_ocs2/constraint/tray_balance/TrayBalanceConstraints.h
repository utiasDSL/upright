#pragma once

#include <memory>

#include <ocs2_pinocchio_interface/PinocchioEndEffectorKinematicsCppAd.h>
#include <ocs2_robotic_tools/common/SkewSymmetricMatrix.h>
#include <ocs2_robotic_tools/end_effector/EndEffectorKinematics.h>
#include <tray_balance_ocs2/definitions.h>

#include <ocs2_core/constraint/StateInputConstraintCppAd.h>
#include <ocs2_oc/synchronized_module/ReferenceManager.h>

#include <tray_balance_constraints/inequality_constraints.h>

namespace ocs2 {
namespace mobile_manipulator {

class TrayBalanceConstraints final : public StateInputConstraintCppAd {
   public:
    using ad_quaternion_t =
        PinocchioEndEffectorKinematicsCppAd::ad_quaternion_t;
    using ad_rotmat_t = PinocchioEndEffectorKinematicsCppAd::ad_rotmat_t;

    using ad_vec2_t = Eigen::Matrix<ad_scalar_t, 2, 1>;
    using ad_vec3_t = Eigen::Matrix<ad_scalar_t, 3, 1>;
    using ad_mat3_t = Eigen::Matrix<ad_scalar_t, 3, 3>;

    TrayBalanceConstraints(
        const PinocchioEndEffectorKinematicsCppAd& pinocchioEEKinematics,
        const TrayBalanceConfiguration& config, bool recompileLibraries)
        : StateInputConstraintCppAd(ConstraintOrder::Linear),
          pinocchioEEKinPtr_(pinocchioEEKinematics.clone()),
          config_(config),
          params_(config.num_parameters()) {
        if (pinocchioEEKinematics.getIds().size() != 1) {
            throw std::runtime_error(
                "[TrayBalanaceConstraint] endEffectorKinematics has wrong "
                "number of end effector IDs.");
        }

        // pre-compute parameter values (since they don't currently change
        // during execution)
        size_t index = 0;
        for (const auto& obj : config_.objects) {
            // std::cout << obj.body.inertia << std::endl;
            vector_t p = obj.get_parameters();
            size_t n = p.size();
            params_.segment(index, n) = p;
            index += n;
            param_sizes_.push_back(n);
        }

        // initialize everything, mostly the CppAD interface (compile the
        // library)
        initialize(STATE_DIM, INPUT_DIM, config_.num_parameters(),
                   "tray_balance_constraints", "/tmp/ocs2", recompileLibraries,
                   true);
    }

    TrayBalanceConstraints* clone() const override {
        // Always pass recompileLibraries = false to avoid recompiling the same
        // library just because this object is cloned.
        return new TrayBalanceConstraints(*pinocchioEEKinPtr_, config_, false);
    }

    size_t getNumConstraints(scalar_t time) const override {
        return config_.num_constraints();
    }

    size_t getNumConstraints() const { return getNumConstraints(0); }

    vector_t getParameters(scalar_t time) const override { return params_; }

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

        // for debugging purposes: we can skip using the parameters and just
        // hardcode the values into the AD library
        // std::vector<BalancedObject<ad_scalar_t>> objects;
        // for (auto& obj : config_.objects) {
        //     objects.push_back(obj.cast<ad_scalar_t>());
        // }

        // Reconstruct the objects from the parameters
        std::vector<BalancedObject<ad_scalar_t>> objects;
        size_t index = 0;
        for (int i = 0; i < param_sizes_.size(); ++i) {
            size_t n = param_sizes_[i];
            auto obj = BalancedObject<ad_scalar_t>::from_parameters(
                parameters.segment(index, n));
            objects.push_back(obj);
            index += n;
        }

        ad_vector_t constraints = balancing_constraints<ad_scalar_t>(
            C_we, angular_vel, linear_acc, angular_acc, objects, config_.enabled);

        return constraints;
    }

   private:
    TrayBalanceConstraints(const TrayBalanceConstraints& other) = default;

    std::unique_ptr<PinocchioEndEffectorKinematicsCppAd> pinocchioEEKinPtr_;
    TrayBalanceConfiguration config_;

    std::vector<size_t> param_sizes_;
    vector_t params_;
};

}  // namespace mobile_manipulator
}  // namespace ocs2
