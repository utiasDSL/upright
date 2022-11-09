#include <upright_control/types.h>

#include "upright_control/inertial_alignment.h"

namespace upright {

VecXad InertialAlignmentConstraint::constraintFunction(
    ocs2::ad_scalar_t time, const VecXad& state, const VecXad& input,
    const VecXad& parameters) const {
    Mat3ad C_we = kinematics_ptr_->getOrientationCppAd(state);
    Vec3ad linear_acc = kinematics_ptr_->getAccelerationCppAd(state, input);
    Vec3ad gravity = gravity_.cast<ocs2::ad_scalar_t>();
    Vec3ad n = settings_.contact_plane_normal.cast<ocs2::ad_scalar_t>();
    Mat23ad S = settings_.contact_plane_span.cast<ocs2::ad_scalar_t>();

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
    } else if (settings_.align_with_fixed_vector) {
        // In this case we just try to stay aligned with whatever the
        // original normal was.
        a = C_we.transpose() * n;
    }

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

ocs2::ad_scalar_t InertialAlignmentCost::costFunction(
    ocs2::ad_scalar_t time, const VecXad& state, const VecXad& input,
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

    if (settings_.align_with_fixed_vector) {
        // Negative because we want to maximize
        Vec3ad n = settings_.contact_plane_normal.cast<ocs2::ad_scalar_t>();
        return -settings_.cost_weight * n.dot(C_we.transpose() * n);
    } else {
        Mat23ad S = settings_.contact_plane_span.cast<ocs2::ad_scalar_t>();
        Vec2ad e = S * (C_we.transpose() * total_acc) / gravity.norm();
        return 0.5 * settings_.cost_weight * e.dot(e);
    }
}

void InertialAlignmentCostGaussNewton::initialize(
    size_t nx, size_t nu, size_t np, const std::string& modelName,
    const std::string& modelFolder, bool recompileLibraries, bool verbose) {
    // Compile arbitrary vector function
    auto ad_func = [=](const VecXad& taped_vars, const VecXad& p, VecXad& y) {
        assert(taped_vars.rows() == 1 + nx + nu);
        const ocs2::ad_scalar_t t = taped_vars(0);
        const VecXad x = taped_vars.segment(1, nx);
        const VecXad u = taped_vars.tail(nu);
        y = this->function(t, x, u, p);
    };
    ad_interface_ptr_.reset(new ocs2::CppAdInterface(ad_func, 1 + nx + nu, np,
                                                     modelName, modelFolder));

    if (recompileLibraries) {
        ad_interface_ptr_->createModels(
            ocs2::CppAdInterface::ApproximationOrder::First, verbose);
    } else {
        ad_interface_ptr_->loadModelsIfAvailable(
            ocs2::CppAdInterface::ApproximationOrder::First, verbose);
    }
}

ocs2::scalar_t InertialAlignmentCostGaussNewton::getValue(
    ocs2::scalar_t time, const VecXd& state, const VecXd& input,
    const ocs2::TargetTrajectories& target, const ocs2::PreComputation&) const {
    VecXd tapedTimeStateInput(1 + state.rows() + input.rows());
    tapedTimeStateInput << time, state, input;
    const VecXd f = ad_interface_ptr_->getFunctionValue(
        tapedTimeStateInput, getParameters(time, target));
    return 0.5 * settings_.cost_weight * f.dot(f);
}

ocs2::ScalarFunctionQuadraticApproximation
InertialAlignmentCostGaussNewton::getQuadraticApproximation(
    ocs2::scalar_t time, const VecXd& state, const VecXd& input,
    const ocs2::TargetTrajectories& target, const ocs2::PreComputation&) const {
    const size_t nx = state.rows();
    const size_t nu = input.rows();
    const VecXd params = getParameters(time, target);
    VecXd tapedTimeStateInput(1 + state.rows() + input.rows());
    tapedTimeStateInput << time, state, input;

    const VecXd e =
        ad_interface_ptr_->getFunctionValue(tapedTimeStateInput, params);
    const MatXd J = ad_interface_ptr_->getJacobian(tapedTimeStateInput, params);

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

VecXad InertialAlignmentCostGaussNewton::function(
    ocs2::ad_scalar_t time, const VecXad& state, const VecXad& input,
    const VecXad& parameters) const {
    Mat3ad C_we = kinematics_ptr_->getOrientationCppAd(state);
    Vec3ad linear_acc = kinematics_ptr_->getAccelerationCppAd(state, input);

    Vec3ad gravity = gravity_.cast<ocs2::ad_scalar_t>();
    Vec3ad total_acc = linear_acc - gravity;

    Mat23ad S = settings_.contact_plane_span.cast<ocs2::ad_scalar_t>();
    return S * (C_we.transpose() * total_acc) / gravity.norm();
}

}  // namespace upright
