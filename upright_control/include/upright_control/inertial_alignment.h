#pragma once

#include <memory>

#include <ocs2_core/automatic_differentiation/CppAdInterface.h>
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
        contact_plane_span << 1, 0, 0, 0, 1, 0;
        com = Vec3d::Zero();
    }

    // True to enable the alignment cost
    bool cost_enabled = false;

    // True to enable the alignment constraint
    bool constraint_enabled = false;

    // Whether to include acceleration resulting from the rotation of the
    // object's CoM in the cost/constraint
    bool use_angular_acceleration = false;

    bool align_with_fixed_vector = false;

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
    out << "cost_enabled = " << settings.cost_enabled << std::endl
        << "constraint_enabled = " << settings.constraint_enabled << std::endl
        << "use_angular_acceleration = " << settings.use_angular_acceleration
        << std::endl
        << "cost_weight = " << settings.cost_weight << std::endl
        << "contact_plane_normal = "
        << settings.contact_plane_normal.transpose() << std::endl
        << "contact_plane_span = " << settings.contact_plane_span << std::endl
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
                              const VecXad& parameters) const override;

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
                                   const VecXad& parameters) const override;

   private:
    InertialAlignmentCost(const InertialAlignmentCost& other) = default;

    std::unique_ptr<ocs2::PinocchioEndEffectorKinematicsCppAd> kinematics_ptr_;
    InertialAlignmentSettings settings_;
    Vec3d gravity_;
    OptimizationDimensions dims_;
};

// Use Gauss-Newton approximation of the cost to avoid ill-conditioned (i.e.
// indefinite) Hessian
class InertialAlignmentCostGaussNewton final : public ocs2::StateInputCost {
   public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    InertialAlignmentCostGaussNewton(
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

    InertialAlignmentCostGaussNewton* clone() const override {
        return new InertialAlignmentCostGaussNewton(*kinematics_ptr_, settings_,
                                                    gravity_, dims_, false);
    }

    VecXd getParameters(ocs2::scalar_t, const ocs2::TargetTrajectories&) const {
        return VecXd(0);
    };

    void initialize(size_t nx, size_t nu, size_t np,
                    const std::string& modelName,
                    const std::string& modelFolder, bool recompileLibraries,
                    bool verbose) {
        // Compile arbitrary vector function
        auto ad_func = [=](const VecXad& taped_vars, const VecXad& p,
                           VecXad& y) {
            assert(taped_vars.rows() == 1 + nx + nu);
            const ocs2::ad_scalar_t t = taped_vars(0);
            const VecXad x = taped_vars.segment(1, nx);
            const VecXad u = taped_vars.tail(nu);
            y = this->function(t, x, u, p);
        };
        ad_interface_ptr_.reset(new ocs2::CppAdInterface(
            ad_func, 1 + nx + nu, np, modelName, modelFolder));

        if (recompileLibraries) {
            ad_interface_ptr_->createModels(
                ocs2::CppAdInterface::ApproximationOrder::First, verbose);
        } else {
            ad_interface_ptr_->loadModelsIfAvailable(
                ocs2::CppAdInterface::ApproximationOrder::First, verbose);
        }
    }

    ocs2::scalar_t getValue(ocs2::scalar_t time, const VecXd& state,
                            const VecXd& input,
                            const ocs2::TargetTrajectories& target,
                            const ocs2::PreComputation&) const {
        VecXd tapedTimeStateInput(1 + state.rows() + input.rows());
        tapedTimeStateInput << time, state, input;
        const VecXd f = ad_interface_ptr_->getFunctionValue(
            tapedTimeStateInput, getParameters(time, target));
        return 0.5 * settings_.cost_weight * f.dot(f);
    }

    ocs2::ScalarFunctionQuadraticApproximation getQuadraticApproximation(
        ocs2::scalar_t time, const VecXd& state, const VecXd& input,
        const ocs2::TargetTrajectories& target,
        const ocs2::PreComputation&) const {
        const size_t nx = state.rows();
        const size_t nu = input.rows();
        const VecXd params = getParameters(time, target);
        VecXd tapedTimeStateInput(1 + state.rows() + input.rows());
        tapedTimeStateInput << time, state, input;

        const VecXd e =
            ad_interface_ptr_->getFunctionValue(tapedTimeStateInput, params);
        const MatXd J =
            ad_interface_ptr_->getJacobian(tapedTimeStateInput, params);

        MatXd dedx = J.middleCols(1, nx);
        MatXd dedu = J.rightCols(nu);

        ocs2::ScalarFunctionQuadraticApproximation cost;
        cost.f = 0.5 * settings_.cost_weight * e.dot(e);

        // Final transpose is because dfdx and dfdu are stored as (column)
        // vectors (i.e. the gradients of the cost)
        cost.dfdx = settings_.cost_weight * (e.transpose() * dedx).transpose();
        cost.dfdu = settings_.cost_weight * (e.transpose() * dedu).transpose();

        // Gauss-Newton approximation to the Hessian
        cost.dfdxx = settings_.cost_weight * dedx.transpose() * dedx;
        cost.dfdux = settings_.cost_weight * dedu.transpose() * dedx;
        cost.dfduu = settings_.cost_weight * dedu.transpose() * dedu;

        return cost;
    }

   protected:
    VecXad function(ocs2::ad_scalar_t time, const VecXad& state,
                    const VecXad& input, const VecXad& parameters) const {
        Mat3ad C_we = kinematics_ptr_->getOrientationCppAd(state);
        Vec3ad linear_acc = kinematics_ptr_->getAccelerationCppAd(state, input);

        Vec3ad gravity = gravity_.cast<ocs2::ad_scalar_t>();
        Vec3ad total_acc = linear_acc - gravity;

        Mat23ad S = settings_.contact_plane_span.cast<ocs2::ad_scalar_t>();
        return S * (C_we.transpose() * total_acc) / total_acc.norm();
    }

   private:
    InertialAlignmentCostGaussNewton(
        const InertialAlignmentCostGaussNewton& other) = default;

    std::unique_ptr<ocs2::PinocchioEndEffectorKinematicsCppAd> kinematics_ptr_;
    InertialAlignmentSettings settings_;
    Vec3d gravity_;
    OptimizationDimensions dims_;

    std::unique_ptr<ocs2::CppAdInterface> ad_interface_ptr_;
};

}  // namespace upright
