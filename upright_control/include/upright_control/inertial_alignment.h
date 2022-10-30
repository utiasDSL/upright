#pragma once

#include <memory>

#include <ocs2_core/constraint/StateInputConstraintCppAd.h>
#include <ocs2_core/cost/StateInputCostCppAd.h>
#include <ocs2_pinocchio_interface/PinocchioEndEffectorKinematicsCppAd.h>

#include <upright_control/dimensions.h>
#include <upright_control/types.h>
#include <upright_core/util.h>

namespace upright {

struct InertialAlignmentSettings {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    InertialAlignmentSettings() {
        contact_plane_normal = Vec3d::UnitZ();
        com = Vec3d::Zero();
    }

    // True to enable alignment cost/constraints, false to disable them.
    bool enabled = false;

    // True to constrain the alignment; false to create a cost to encourage
    // alignment
    bool use_constraint = true;

    // Whether to include acceleration resulting from the rotation of the
    // object's CoM in the cost/constraint
    bool use_angular_acceleration = false;

    // Bound for the constraint: n.dot(a) >= constraint_bound * ||a||. Must be
    // in >= 0. Setting this to 0 means the constraint is an equality.
    ocs2::scalar_t alpha = 0;

    // Cost weighting (no effect if use_constraint = true)
    ocs2::scalar_t cost_weight = 1.0;

    // Contact plane normal and span (in the end effector frame)
    Vec3d contact_plane_normal;
    Mat23d contact_plane_span;

    // Center of mass w.r.t. the end effector (no effect if
    // use_angular_acceleration = false)
    Vec3d com;
};

inline std::ostream& operator<<(std::ostream& out,
                                const InertialAlignmentSettings& settings) {
    out << "enabled = " << settings.enabled << std::endl
        << "use_constraint = " << settings.use_constraint << std::endl
        << "use_angular_acceleration = " << settings.use_angular_acceleration
        << std::endl
        << "cost_weight = " << settings.cost_weight << std::endl
        << "contact_plane_normal = "
        << settings.contact_plane_normal.transpose() << std::endl
        << "com = " << settings.com.transpose() << std::endl;
    return out;
}

// Constraint form of inertial alignment
class InertialAlignmentConstraint final
    : public ocs2::StateInputConstraintCppAd {
   public:
    InertialAlignmentConstraint(
        const ocs2::PinocchioEndEffectorKinematicsCppAd& kinematics,
        const InertialAlignmentSettings& settings, const Vec3d& gravity,
        const OptimizationDimensions& dims, bool recompile_libraries)
        : ocs2::StateInputConstraintCppAd(ocs2::ConstraintOrder::Linear),
          kinematics_ptr_(kinematics.clone()),
          settings_(settings),
          gravity_(gravity),
          dims_(dims) {
        initialize(dims.x(), dims.u(), 0, "inertial_alignment_constraint",
                   "/tmp/ocs2", recompile_libraries, true);
    }

    InertialAlignmentConstraint* clone() const override {
        // Always pass recompileLibraries = false to avoid recompiling the same
        // library just because this object is cloned.
        return new InertialAlignmentConstraint(*kinematics_ptr_, settings_,
                                               gravity_, dims_, false);
    }

    size_t getNumConstraints(ocs2::scalar_t time) const override { return 5; }

    size_t getNumConstraints() const { return getNumConstraints(0); }

   protected:
    VecXad constraintFunction(ocs2::ad_scalar_t time, const VecXad& state,
                              const VecXad& input,
                              const VecXad& parameters) const override {
        Mat3ad C_we = kinematics_ptr_->getOrientationCppAd(state);
        Vec3ad linear_acc = kinematics_ptr_->getAccelerationCppAd(state, input);
        Vec3ad gravity = gravity_.cast<ocs2::ad_scalar_t>();

        // In the EE frame
        Vec3ad a = C_we.transpose() * (linear_acc - gravity);

        // Take into account object center of mass, if available
        if (settings_.use_angular_acceleration) {
            Vec3ad angular_vel =
                kinematics_ptr_->getAngularVelocityCppAd(state, input);
            Vec3ad angular_acc =
                kinematics_ptr_->getAngularAccelerationCppAd(state, input);

            Mat3ad ddC_we =
                dC_dtt<ocs2::ad_scalar_t>(C_we, angular_vel, angular_acc);
            Vec3ad com = settings_.com.cast<ocs2::ad_scalar_t>();

            a += ddC_we * com;
        }

        Vec3ad n = settings_.contact_plane_normal.cast<ocs2::ad_scalar_t>();
        Mat23ad S = settings_.contact_plane_span.cast<ocs2::ad_scalar_t>();

        ocs2::ad_scalar_t a_n = n.dot(a);
        Vec2ad a_t = S * a;

        // constrain the normal acc to be non-negative
        VecXad constraints(getNumConstraints(0));
        constraints(0) = a_n;

        // linearized version: the quadratic cone does not play well with the
        // optimizer
        constraints(1) = settings_.alpha * a_n - a_t(0) - a_t(1);
        constraints(2) = settings_.alpha * a_n - a_t(0) + a_t(1);
        constraints(3) = settings_.alpha * a_n + a_t(0) - a_t(1);
        constraints(4) = settings_.alpha * a_n + a_t(0) + a_t(1);
        return constraints;
    }

   private:
    InertialAlignmentConstraint(const InertialAlignmentConstraint& other) =
        default;

    std::unique_ptr<ocs2::PinocchioEndEffectorKinematicsCppAd> kinematics_ptr_;
    InertialAlignmentSettings settings_;
    Vec3d gravity_;
    OptimizationDimensions dims_;
};

// Constraint form of inertial alignment
// Given the constraint form c = 0, the cost form computes
// 0.5 * w * c.T * c, where w > 0 is a scalar weight
class InertialAlignmentCost final : public ocs2::StateInputCostCppAd {
   public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    InertialAlignmentCost(
        const ocs2::PinocchioEndEffectorKinematicsCppAd& kinematics,
        const InertialAlignmentSettings& settings, const Vec3d& gravity,
        const OptimizationDimensions& dims, bool recompile_libraries)
        : kinematics_ptr_(kinematics.clone()),
          settings_(settings),
          gravity_(gravity),
          dims_(dims) {
        initialize(dims.x(), dims.u(), 0, "inertial_alignment_cost",
                   "/tmp/ocs2", recompile_libraries, true);
    }

    InertialAlignmentCost* clone() const override {
        return new InertialAlignmentCost(*kinematics_ptr_, settings_, gravity_,
                                         dims_, false);
    }

   protected:
    ocs2::ad_scalar_t costFunction(ocs2::ad_scalar_t time, const VecXad& state,
                                   const VecXad& input,
                                   const VecXad& parameters) const {
        Mat3ad C_we = kinematics_ptr_->getOrientationCppAd(state);
        Vec3ad linear_acc = kinematics_ptr_->getAccelerationCppAd(state, input);

        Vec3ad gravity = gravity_.cast<ocs2::ad_scalar_t>();
        Vec3ad total_acc = linear_acc - gravity;

        if (settings_.use_angular_acceleration) {
            Vec3ad angular_vel =
                kinematics_ptr_->getAngularVelocityCppAd(state, input);
            Vec3ad angular_acc =
                kinematics_ptr_->getAngularAccelerationCppAd(state, input);

            Mat3ad ddC_we =
                dC_dtt<ocs2::ad_scalar_t>(C_we, angular_vel, angular_acc);
            Vec3ad com = settings_.com.cast<ocs2::ad_scalar_t>();

            total_acc += ddC_we * com;
        }

        Vec3ad n = settings_.contact_plane_normal.cast<ocs2::ad_scalar_t>();

        // Scaling by inverse of gravity makes the numerics better
        Vec3ad err = (C_we * n).cross(total_acc) / gravity.norm();
        return 0.5 * settings_.cost_weight * err.dot(err);
    }

   private:
    InertialAlignmentCost(const InertialAlignmentCost& other) = default;

    std::unique_ptr<ocs2::PinocchioEndEffectorKinematicsCppAd> kinematics_ptr_;
    InertialAlignmentSettings settings_;
    Vec3d gravity_;
    OptimizationDimensions dims_;
};

}  // namespace upright
