#pragma once

#include <memory>

#include <ocs2_mobile_manipulator_modified/constraint/TrayBalanceUtil.h>
#include <ocs2_mobile_manipulator_modified/definitions.h>
#include <ocs2_pinocchio_interface/PinocchioEndEffectorKinematicsCppAd.h>
#include <ocs2_robotic_tools/common/SkewSymmetricMatrix.h>
#include <ocs2_robotic_tools/end_effector/EndEffectorKinematics.h>

#include <ocs2_core/constraint/StateInputConstraintCppAd.h>
#include <ocs2_oc/synchronized_module/ReferenceManager.h>

namespace ocs2 {
namespace mobile_manipulator {

class TrayBalanceConstraints final : public StateInputConstraintCppAd {
   public:
    // using vector3_t = Eigen::Matrix<scalar_t, 3, 1>;
    // using quaternion_t = Eigen::Quaternion<scalar_t>;
    using ad_quaternion_t =
        PinocchioEndEffectorKinematicsCppAd::ad_quaternion_t;
    using ad_rotmat_t = PinocchioEndEffectorKinematicsCppAd::ad_rotmat_t;
    // using ad_quaternion_t = Eigen::Quaternion<ad_scalar_t>;

    TrayBalanceConstraints(
        const PinocchioEndEffectorKinematicsCppAd& pinocchioEEKinematics)
        : StateInputConstraintCppAd(ConstraintOrder::Linear),
          pinocchioEEKinPtr_(pinocchioEEKinematics.clone()) {
        if (pinocchioEEKinematics.getIds().size() != 1) {
            throw std::runtime_error(
                "[EndEffectorConstraint] endEffectorKinematics has wrong "
                "number of "
                "end effector IDs.");
        }
        // initialize everything, mostly the CppAD interface
        std::cerr << "about to initialize tray balance" << std::endl;
        initialize(STATE_DIM, INPUT_DIM, 0, "tray_balance_constraints",
                   "/tmp/ocs2", true, true);
        std::cerr << "done initialize tray balance" << std::endl;
    }

    // TrayBalanceConstraints() override = default;

    TrayBalanceConstraints* clone() const override {
        return new TrayBalanceConstraints(*pinocchioEEKinPtr_);
    }

    // NOTE: this should be implemented by StateInputConstraintCppAd
    // VectorFunctionLinearApproximation getLinearApproximation(
    //     scalar_t time, const vector_t& state,
    //     const PreComputation& preComputation) const override;

    size_t getNumConstraints(scalar_t time) const override { return 4; }

    size_t getNumConstraints() const { return getNumConstraints(0); }

   protected:
    ad_vector_t constraintFunction(
        ad_scalar_t time, const ad_vector_t& state, const ad_vector_t& input,
        const ad_vector_t& parameters) const override {
        // TODO probably construct these elsewhere later
        ad_rotmat_t It = cylinder_inertia_matrix<ad_scalar_t>(
            ad_scalar_t(TRAY_MASS), ad_scalar_t(TRAY_RADIUS),
            ad_scalar_t(2 * TRAY_COM_HEIGHT));

        // TODO placeholder, need to see what this value should actually be
        // TODO r_te_e should be passed in as a parameter perhaps
        ad_vector_t r_te_e(3);
        r_te_e << ad_scalar_t(0), ad_scalar_t(0), ad_scalar_t(0.2);

        ad_rotmat_t C_we = pinocchioEEKinPtr_->getOrientationCppAd(state);
        ad_rotmat_t C_ew = C_we.transpose();

        ad_vector_t angular_vel =
            pinocchioEEKinPtr_->getAngularVelocityCppAd(state, input);
        ad_vector_t angular_acc =
            pinocchioEEKinPtr_->getAngularAccelerationCppAd(state, input);
        ad_vector_t linear_acc =
            pinocchioEEKinPtr_->getAccelerationCppAd(state, input);

        ad_matrix_t S_angular_vel =
            skewSymmetricMatrix<ad_scalar_t>(angular_vel);
        ad_matrix_t S_angular_acc =
            skewSymmetricMatrix<ad_scalar_t>(angular_acc);
        ad_matrix_t ddC_we =
            (S_angular_acc + S_angular_vel * S_angular_vel) * C_we;

        ad_vector_t g(3);
        g << ad_scalar_t(0), ad_scalar_t(0), ad_scalar_t(-GRAVITY);

        ad_vector_t alpha =
            ad_scalar_t(TRAY_MASS) * C_ew * (linear_acc + ddC_we * r_te_e - g);

        ad_matrix_t Iw = C_we * It * C_ew;
        ad_vector_t beta =
            C_ew * S_angular_vel * Iw * angular_vel + It * C_ew * angular_acc;

        ad_matrix_t S(2, 2);
        S << ad_scalar_t(0), ad_scalar_t(1), ad_scalar_t(-1), ad_scalar_t(0);

        ad_scalar_t rz = ad_scalar_t(-TRAY_COM_HEIGHT);
        ad_scalar_t r = equilateral_triangle_inscribed_radius<ad_scalar_t>(
            ad_scalar_t(EE_TRIANGLE_SIDE_LENGTH));

        ad_vector_t gamma =
            rz * S.transpose() * angular_acc.head<2>() - beta.head<2>();

        // friction constraint(s)
        ad_scalar_t eps2 = ad_scalar_t(0.01);  // TODO still needed?
        ad_scalar_t h1 =
            ad_scalar_t(TRAY_MU) * alpha(2) -
            CppAD::sqrt(alpha(0) * alpha(0) + alpha(1) * alpha(1) + eps2);
        ad_scalar_t h1a = h1 + beta(2) / r;
        ad_scalar_t h1b = h1 - beta(2) / r;

        // normal constraint
        ad_scalar_t h2 = alpha(2);

        // tipping constraint
        ad_scalar_t h3 = r * r * alpha(2) * alpha(2) - gamma(0) * gamma(0) -
                         gamma(1) * gamma(1);

        ad_vector_t constraints(getNumConstraints());
        constraints << h1a, h1b, h2, h3;
        // constraints << ad_scalar_t(1), ad_scalar_t(1), ad_scalar_t(1),
        //     ad_scalar_t(1);
        return constraints;
    }

   private:
    TrayBalanceConstraints(const TrayBalanceConstraints& other) = default;

    std::unique_ptr<PinocchioEndEffectorKinematicsCppAd> pinocchioEEKinPtr_;

    // TODO may not need this
    // std::unique_ptr<EndEffectorKinematics<scalar_t>>
    // endEffectorKinematicsPtr_;
    //
    // const ReferenceManager* referenceManagerPtr_;
};

}  // namespace mobile_manipulator
}  // namespace ocs2
