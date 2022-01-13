#pragma once

#include <memory>

#include <ocs2_pinocchio_interface/PinocchioEndEffectorKinematicsCppAd.h>
#include <ocs2_robotic_tools/common/SkewSymmetricMatrix.h>
#include <ocs2_robotic_tools/end_effector/EndEffectorKinematics.h>
#include <tray_balance_ocs2/definitions.h>

#include <ocs2_core/constraint/StateInputConstraintCppAd.h>
#include <ocs2_oc/synchronized_module/ReferenceManager.h>

#include <tray_balance_constraints/inequality_constraints.h>
#include <tray_balance_constraints/robust.h>
#include <tray_balance_ocs2/constraint/tray_balance/TrayBalanceConfigurations.h>

namespace ocs2 {
namespace mobile_manipulator {

// required to convert from config-specified <scalar_t> values to required
// <ad_scalar_t> values
ParameterSet<ad_scalar_t> cast_parameter_set_to_ad(
    ParameterSet<scalar_t> config) {
    ParameterSet<ad_scalar_t> ad_config;
    ad_config.min_support_dist = config.min_support_dist;
    ad_config.min_mu = config.min_mu;
    ad_config.min_r_tau = config.min_r_tau;
    ad_config.max_radius = config.max_radius;

    for (auto ball : config.balls) {
        Vec3<ad_scalar_t> ad_center = ball.center.cast<ad_scalar_t>();
        Ball<ad_scalar_t> ad_ball(ad_center, ad_scalar_t(ball.radius));
        ad_config.balls.push_back(ad_ball);
    }
    return ad_config;
}

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
        const ParameterSet<scalar_t>& params)
        : StateInputConstraintCppAd(ConstraintOrder::Linear),
          pinocchioEEKinPtr_(pinocchioEEKinematics.clone()),
          params_(params) {
        if (pinocchioEEKinematics.getIds().size() != 1) {
            throw std::runtime_error(
                "[EndEffectorConstraint] endEffectorKinematics has wrong "
                "number of end effector IDs.");
        }
        // initialize everything, mostly the CppAD interface
        initialize(STATE_DIM, INPUT_DIM, 0, "robust_tray_balance_constraints",
                   "/tmp/ocs2", true, true);
    }

    RobustTrayBalanceConstraints* clone() const override {
        return new RobustTrayBalanceConstraints(*pinocchioEEKinPtr_, params_);
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

        ParameterSet<ad_scalar_t> params = cast_parameter_set_to_ad(params_);
        ad_vector_t constraints = robust_balancing_constraints<ad_scalar_t>(
            C_we, angular_vel, linear_acc, angular_acc, params);

        // ad_scalar_t min_support_dist(0.05);
        // ad_scalar_t min_mu(0.5);
        // ad_scalar_t min_r_tau = circle_r_tau(min_support_dist);
        // ad_scalar_t max_radius;
        // std::vector<Ball<ad_scalar_t>> balls;
        //
        // if (config_.arrangement ==
        //     TrayBalanceConfiguration::Arrangement::Flat) {
        //     Vec3<ad_scalar_t> center_flat(ad_scalar_t(0), ad_scalar_t(0),
        //                                   ad_scalar_t(0.02 + 0.02 + 0.075));
        //     ad_scalar_t radius_flat(0.1);
        //     Ball<ad_scalar_t> ball_flat(center_flat, radius_flat);
        //     balls.push_back(ball_flat);
        //
        //     max_radius = radius_flat;
        //
        // } else {
        //     // bottom ball
        //     Vec3<ad_scalar_t> center1(ad_scalar_t(0), ad_scalar_t(0),
        //                               ad_scalar_t(0.1));
        //     ad_scalar_t radius1(0.12);
        //     Ball<ad_scalar_t> ball1(center1, radius1);
        //     balls.push_back(ball1);
        //
        //     // top ball
        //     Vec3<ad_scalar_t> center2(ad_scalar_t(0), ad_scalar_t(0),
        //                               ad_scalar_t(0.3));
        //     ad_scalar_t radius2(0.12);
        //     Ball<ad_scalar_t> ball2(center2, radius2);
        //     balls.push_back(ball2);
        //
        //     // Note that this isn't really correct (but the correct value is so
        //     // conservative that we can get away with this)
        //     // TODO this is wrong: should be half this value (this is the
        //     // diameter)
        //     max_radius = (center2 - center1).norm() + radius1 + radius2;
        // }
        //
        // ParameterSet<ad_scalar_t> param_set(balls, min_support_dist, min_mu,
        //                                     min_r_tau, max_radius);
        // ad_vector_t constraints = robust_balancing_constraints<ad_scalar_t>(
        //     C_we, angular_vel, linear_acc, angular_acc, param_set);

        return constraints;
    }

   private:
    RobustTrayBalanceConstraints(const RobustTrayBalanceConstraints& other) =
        default;

    std::unique_ptr<PinocchioEndEffectorKinematicsCppAd> pinocchioEEKinPtr_;
    ParameterSet<scalar_t> params_;
};

}  // namespace mobile_manipulator
}  // namespace ocs2
