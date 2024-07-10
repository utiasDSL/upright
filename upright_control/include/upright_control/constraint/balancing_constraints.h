#pragma once

#include <ocs2_core/constraint/StateInputConstraintCppAd.h>
#include <ocs2_pinocchio_interface/PinocchioEndEffectorKinematicsCppAd.h>
#include <upright_control/dimensions.h>
#include <upright_control/types.h>
#include <upright_core/contact.h>
#include <upright_core/rigid_body.h>

namespace upright {

struct BalancingSettings {
    bool enabled = false;

    // Name of the arrangement is used to produce different CppAD libraries,
    // which can then be re-used later to save startup time
    std::string arrangement_name;

    // BalanceConstraintsEnabled constraints_enabled;
    std::map<std::string, RigidBody<ocs2::scalar_t>> bodies;

    std::vector<ContactPoint<ocs2::scalar_t>> contacts;

    // Weight on the contact forces in the optimization problem
    ocs2::scalar_t force_weight = 0.01;
};

std::ostream& operator<<(std::ostream& out, const BalancingSettings& settings);

// Balancing constraints based on contact forces between objects.
class ContactForceBalancingConstraints final
    : public ocs2::StateInputConstraintCppAd {
   public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    ContactForceBalancingConstraints(
        const ocs2::PinocchioEndEffectorKinematicsCppAd& pinocchioEEKinematics,
        const BalancingSettings& settings, const Vec3d& gravity,
        const OptimizationDimensions& dims, bool recompileLibraries);

    ContactForceBalancingConstraints* clone() const override {
        // Always pass recompileLibraries = false to avoid recompiling the same
        // library just because this object is cloned.
        return new ContactForceBalancingConstraints(
            *pinocchioEEKinPtr_, settings_, gravity_, dims_, false);
    }

    size_t getNumConstraints(ocs2::scalar_t time) const override {
        return num_constraints_;
    }

    size_t getNumConstraints() const { return getNumConstraints(0); }

    VecXd getParameters(ocs2::scalar_t time) const override {
        // Parameters are constant for now
        return VecXd(0);
    }

   protected:
    VecXad constraintFunction(ocs2::ad_scalar_t time, const VecXad& state,
                              const VecXad& input,
                              const VecXad& parameters) const override;

   private:
    ContactForceBalancingConstraints(
        const ContactForceBalancingConstraints& other) = default;

    std::unique_ptr<ocs2::PinocchioEndEffectorKinematicsCppAd>
        pinocchioEEKinPtr_;
    BalancingSettings settings_;
    OptimizationDimensions dims_;
    Vec3d gravity_;
    size_t num_constraints_;
};

// Equality constraints that ensure contact forces on the object are consistent
// with sticking to the EE.
class ObjectDynamicsConstraints final : public ocs2::StateInputConstraintCppAd {
   public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    ObjectDynamicsConstraints(
        const ocs2::PinocchioEndEffectorKinematicsCppAd& pinocchioEEKinematics,
        const BalancingSettings& settings, const Vec3d& gravity,
        const OptimizationDimensions& dims, bool recompileLibraries);

    ObjectDynamicsConstraints* clone() const override {
        // Always pass recompileLibraries = false to avoid recompiling the same
        // library just because this object is cloned.
        return new ObjectDynamicsConstraints(*pinocchioEEKinPtr_, settings_,
                                             gravity_, dims_, false);
    }

    size_t getNumConstraints(ocs2::scalar_t time) const override {
        return num_constraints_;
    }

    size_t getNumConstraints() const { return getNumConstraints(0); }

    VecXd getParameters(ocs2::scalar_t time) const override;

   protected:
    VecXad constraintFunction(ocs2::ad_scalar_t time, const VecXad& state,
                              const VecXad& input,
                              const VecXad& parameters) const override;

   private:
    ObjectDynamicsConstraints(const ObjectDynamicsConstraints& other) = default;

    std::unique_ptr<ocs2::PinocchioEndEffectorKinematicsCppAd>
        pinocchioEEKinPtr_;
    BalancingSettings settings_;
    OptimizationDimensions dims_;
    VecXd parameters_;
    Vec3d gravity_;
    size_t num_constraints_;
};

}  // namespace upright
