#pragma once

#include <memory>

#include <ocs2_core/constraint/StateInputConstraintCppAd.h>
#include <ocs2_oc/synchronized_module/ReferenceManager.h>
#include <ocs2_pinocchio_interface/PinocchioEndEffectorKinematicsCppAd.h>
#include <ocs2_robotic_tools/end_effector/EndEffectorKinematics.h>

#include <tray_balance_constraints/bounded.h>

#include <tray_balance_ocs2/constraint/ConstraintType.h>
#include <tray_balance_ocs2/dynamics/Dimensions.h>
#include <tray_balance_ocs2/types.h>

namespace ocs2 {
namespace mobile_manipulator {

struct TrayBalanceSettings {
    bool enabled = false;
    BalanceConstraintsEnabled constraints_enabled;
    std::vector<BoundedBalancedObject<scalar_t>> objects;

    ConstraintType constraint_type = ConstraintType::Soft;
    scalar_t mu = 1e-2;
    scalar_t delta = 1e-3;
};

std::ostream& operator<<(std::ostream& out,
                         const TrayBalanceSettings& settings);

// TODO: this can be split into a .cpp file, too
class BoundedTrayBalanceConstraints final : public StateInputConstraintCppAd {
   public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    BoundedTrayBalanceConstraints(
        const PinocchioEndEffectorKinematicsCppAd& pinocchioEEKinematics,
        const TrayBalanceSettings& settings, const Vec3d& gravity,
        const RobotDimensions& dims, bool recompileLibraries);

    BoundedTrayBalanceConstraints* clone() const override {
        // Always pass recompileLibraries = false to avoid recompiling the same
        // library just because this object is cloned.
        return new BoundedTrayBalanceConstraints(*pinocchioEEKinPtr_, settings_,
                                                 gravity_, dims_, false);
    }

    size_t getNumConstraints(scalar_t time) const override {
        return num_constraints_;
    }

    size_t getNumConstraints() const { return getNumConstraints(0); }

    vector_t getParameters(scalar_t time) const override {
        // Parameters are constant for now
        return vector_t(0);
    }

   protected:
    ad_vector_t constraintFunction(ad_scalar_t time, const VecXad& state,
                                   const VecXad& input,
                                   const VecXad& parameters) const override;

   private:
    BoundedTrayBalanceConstraints(const BoundedTrayBalanceConstraints& other) =
        default;

    std::unique_ptr<PinocchioEndEffectorKinematicsCppAd> pinocchioEEKinPtr_;
    TrayBalanceSettings settings_;
    RobotDimensions dims_;
    Vec3d gravity_;
    size_t num_constraints_;
};

}  // namespace mobile_manipulator
}  // namespace ocs2
