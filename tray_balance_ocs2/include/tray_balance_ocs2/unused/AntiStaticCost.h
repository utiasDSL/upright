#pragma once

#include <memory>

#include <ocs2_pinocchio_interface/PinocchioEndEffectorKinematicsCppAd.h>
#include <ocs2_robotic_tools/end_effector/EndEffectorKinematics.h>
#include <ocs2_core/cost/StateCostCppAd.h>

namespace ocs2 {
namespace mobile_manipulator {

class AntiStaticCost final : public StateCostCppAd {
   public:
    using ad_quaternion_t =
        PinocchioEndEffectorKinematicsCppAd::ad_quaternion_t;
    using ad_rotmat_t = PinocchioEndEffectorKinematicsCppAd::ad_rotmat_t;

    using ad_vec2_t = Eigen::Matrix<ad_scalar_t, 2, 1>;
    using ad_vec3_t = Eigen::Matrix<ad_scalar_t, 3, 1>;
    using ad_mat2_t = Eigen::Matrix<ad_scalar_t, 2, 2>;
    using ad_mat3_t = Eigen::Matrix<ad_scalar_t, 3, 3>;

    AntiStaticCost(const PinocchioEndEffectorKinematicsCppAd& pinocchioEEKinematics,
            const RobotDimensions& dims)
        : pinocchioEEKinPtr_(pinocchioEEKinematics.clone()), dims_(dims) {
        initialize(dims.x, 0, "anti_static_cost", "/tmp/ocs2", true, true);
    }

    AntiStaticCost* clone() const override {
        return new AntiStaticCost(*pinocchioEEKinPtr_, dims_);
    }

   protected:
    ad_scalar_t costFunction(ad_scalar_t time, const ad_vector_t& state,
                             const ad_vector_t& parameters) const {
        ad_rotmat_t C_we = pinocchioEEKinPtr_->getOrientationCppAd(state);
        ad_vec3_t z = ad_vec3_t::UnitZ();
        return 0.1 * z.dot(C_we * z);
    }

   private:
    AntiStaticCost(const AntiStaticCost& other) = default;

    std::unique_ptr<PinocchioEndEffectorKinematicsCppAd> pinocchioEEKinPtr_;
    RobotDimensions dims_;
};

}  // namespace mobile_manipulator
}  // namespace ocs2
