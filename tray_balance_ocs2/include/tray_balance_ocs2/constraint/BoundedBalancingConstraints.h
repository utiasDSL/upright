#pragma once

#include <memory>

#include <ocs2_core/constraint/StateInputConstraintCppAd.h>
#include <ocs2_oc/synchronized_module/ReferenceManager.h>
#include <ocs2_pinocchio_interface/PinocchioEndEffectorKinematicsCppAd.h>
#include <ocs2_robotic_tools/common/SkewSymmetricMatrix.h>
#include <ocs2_robotic_tools/end_effector/EndEffectorKinematics.h>

#include <tray_balance_constraints/bounded.h>

#include <tray_balance_ocs2/types.h>
#include <tray_balance_ocs2/dynamics/Dimensions.h>
#include <tray_balance_ocs2/constraint/ConstraintType.h>

namespace ocs2 {
namespace mobile_manipulator {

struct TrayBalanceSettings {
    bool enabled = false;
    BalanceConstraintsEnabled constraints_enabled;
    BoundedBalancedObjects<scalar_t> objects;

    ConstraintType constraint_type = ConstraintType::Soft;
    scalar_t mu = 1e-2;
    scalar_t delta = 1e-3;
};

// TODO: this can be split into a .cpp file, too
class BoundedTrayBalanceConstraints final : public StateInputConstraintCppAd {
   public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    BoundedTrayBalanceConstraints(
        const PinocchioEndEffectorKinematicsCppAd& pinocchioEEKinematics,
        const TrayBalanceSettings& settings, const Vec3d& gravity,
        const RobotDimensions& dims, bool recompileLibraries)
        : StateInputConstraintCppAd(ConstraintOrder::Linear),
          pinocchioEEKinPtr_(pinocchioEEKinematics.clone()),
          gravity_(gravity),
          settings_(settings),
          dims_(dims) {
        if (pinocchioEEKinematics.getIds().size() != 1) {
            throw std::runtime_error(
                "[TrayBalanaceConstraint] endEffectorKinematics has wrong "
                "number of end effector IDs.");
        }

        // compile the CppAD library
        initialize(dims.x, dims.u, 0, "bounded_tray_balance_constraints",
                   "/tmp/ocs2", recompileLibraries, true);
    }

    BoundedTrayBalanceConstraints* clone() const override {
        // Always pass recompileLibraries = false to avoid recompiling the same
        // library just because this object is cloned.
        return new BoundedTrayBalanceConstraints(*pinocchioEEKinPtr_, settings_,
                                                 gravity_, dims_, false);
    }

    size_t getNumConstraints(scalar_t time) const override {
        return settings_.objects.num_constraints();
    }

    size_t getNumConstraints() const { return getNumConstraints(0); }

    vector_t getParameters(scalar_t time) const override {
        // Parameters are constant for now
        // return params_;
        return vector_t(0);
    }

   protected:
    ad_vector_t constraintFunction(
        ad_scalar_t time, const VecXad& state, const VecXad& input,
        const VecXad& parameters) const override {
        Mat3ad C_we = pinocchioEEKinPtr_->getOrientationCppAd(state);
        Vec3ad angular_vel =
            pinocchioEEKinPtr_->getAngularVelocityCppAd(state, input);
        Vec3ad angular_acc =
            pinocchioEEKinPtr_->getAngularAccelerationCppAd(state, input);
        Vec3ad linear_acc =
            pinocchioEEKinPtr_->getAccelerationCppAd(state, input);

        // Cast to AD scalar type
        Vec3ad ad_gravity = gravity_.template cast<ad_scalar_t>();
        BoundedBalancedObjects<ad_scalar_t> ad_objects =
            settings_.objects.template cast<ad_scalar_t>();

        return ad_objects.balancing_constraints(
            ad_gravity, settings_.constraints_enabled, C_we, angular_vel,
            linear_acc, angular_acc);
    }

   private:
    BoundedTrayBalanceConstraints(const BoundedTrayBalanceConstraints& other) =
        default;

    std::unique_ptr<PinocchioEndEffectorKinematicsCppAd> pinocchioEEKinPtr_;
    TrayBalanceSettings settings_;
    RobotDimensions dims_;
    Vec3d gravity_;
    // vector_t params_;  // TODO unused
};

}  // namespace mobile_manipulator
}  // namespace ocs2
