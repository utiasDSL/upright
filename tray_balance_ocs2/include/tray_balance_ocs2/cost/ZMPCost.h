#pragma once

#include <memory>

#include <ocs2_pinocchio_interface/PinocchioEndEffectorKinematicsCppAd.h>
#include <ocs2_robotic_tools/common/SkewSymmetricMatrix.h>
#include <ocs2_robotic_tools/end_effector/EndEffectorKinematics.h>
#include <tray_balance_ocs2/definitions.h>

#include <ocs2_core/cost/StateInputCostCppAd.h>
#include <ocs2_oc/synchronized_module/ReferenceManager.h>

#include <tray_balance_constraints/nominal.h>

namespace ocs2 {
namespace mobile_manipulator {

class ZMPCost final : public StateInputCostCppAd {
   public:
    using ad_quaternion_t =
        PinocchioEndEffectorKinematicsCppAd::ad_quaternion_t;
    using ad_rotmat_t = PinocchioEndEffectorKinematicsCppAd::ad_rotmat_t;

    using ad_vec2_t = Eigen::Matrix<ad_scalar_t, 2, 1>;
    using ad_vec3_t = Eigen::Matrix<ad_scalar_t, 3, 1>;
    using ad_mat2_t = Eigen::Matrix<ad_scalar_t, 2, 2>;
    using ad_mat3_t = Eigen::Matrix<ad_scalar_t, 3, 3>;

    ZMPCost(const PinocchioEndEffectorKinematicsCppAd& pinocchioEEKinematics)
        : pinocchioEEKinPtr_(pinocchioEEKinematics.clone()) {
        initialize(STATE_DIM, INPUT_DIM, 0, "zmp_cost", "/tmp/ocs2", true,
                   true);
    }

    ZMPCost* clone() const override { return new ZMPCost(*pinocchioEEKinPtr_); }

   protected:
    ad_scalar_t costFunction(ad_scalar_t time, const ad_vector_t& state,
                             const ad_vector_t& input,
                             const ad_vector_t& parameters) const {
        ad_rotmat_t C_we = pinocchioEEKinPtr_->getOrientationCppAd(state);
        ad_vector_t angular_vel =
            pinocchioEEKinPtr_->getAngularVelocityCppAd(state, input);
        ad_vector_t angular_acc =
            pinocchioEEKinPtr_->getAngularAccelerationCppAd(state, input);
        ad_vector_t linear_acc =
            pinocchioEEKinPtr_->getAccelerationCppAd(state, input);

        ad_scalar_t m(0.5);
        ad_scalar_t mu(0.5);
        ad_scalar_t r(0.5);
        ad_scalar_t r_tau(0);

        ad_mat3_t It = ad_mat3_t::Zero();
        It.diagonal() << m * r * r, m * r * r, ad_scalar_t(0);

        ad_vec3_t com;
        com << ad_scalar_t(0), ad_scalar_t(0), r;

        RigidBody<ad_scalar_t> body(m, It, com);

        std::vector<Vec2<ad_scalar_t>> vertices;
        vertices.push_back(Vec2<ad_scalar_t>::Zero());
        PolygonSupportArea<ad_scalar_t> support_area(vertices);

        BalancedObject<ad_scalar_t> object(body, r, support_area, r_tau, mu);

        ad_vector_t zmp = compute_zmp<ad_scalar_t>(
            C_we, angular_vel, linear_acc, angular_acc, object);

        return zmp.transpose() * zmp;
    }

   private:
    ZMPCost(const ZMPCost& other) = default;

    std::unique_ptr<PinocchioEndEffectorKinematicsCppAd> pinocchioEEKinPtr_;
};

}  // namespace mobile_manipulator
}  // namespace ocs2
