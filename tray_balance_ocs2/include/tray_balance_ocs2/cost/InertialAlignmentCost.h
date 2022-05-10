#pragma once

#include <memory>

#include <ocs2_core/cost/StateInputCostCppAd.h>
#include <ocs2_pinocchio_interface/PinocchioEndEffectorKinematicsCppAd.h>
#include <ocs2_robotic_tools/end_effector/EndEffectorKinematics.h>

#include <tray_balance_constraints/bounded.h>
#include <tray_balance_ocs2/dynamics/Dimensions.h>

namespace ocs2 {
namespace mobile_manipulator {

struct InertialAlignmentSettings {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    InertialAlignmentSettings() {
        gravity = -9.81 * Vec3<scalar_t>::UnitZ();
        r_oe_e = Vec3<scalar_t>::Zero();
    }

    bool enabled = false;
    bool use_angular_acceleration = false;
    scalar_t weight = 1.0;
    Vec3<scalar_t> r_oe_e;  // center of mass
    Vec3<scalar_t> gravity;
};

class InertialAlignmentCost final : public StateInputCostCppAd {
   public:
    using ad_quaternion_t =
        PinocchioEndEffectorKinematicsCppAd::ad_quaternion_t;
    using ad_rotmat_t = PinocchioEndEffectorKinematicsCppAd::ad_rotmat_t;

    using ad_vec2_t = Eigen::Matrix<ad_scalar_t, 2, 1>;
    using ad_vec3_t = Eigen::Matrix<ad_scalar_t, 3, 1>;
    using ad_mat2_t = Eigen::Matrix<ad_scalar_t, 2, 2>;
    using ad_mat3_t = Eigen::Matrix<ad_scalar_t, 3, 3>;

    InertialAlignmentCost(
        const PinocchioEndEffectorKinematicsCppAd& pinocchioEEKinematics,
        const InertialAlignmentSettings& settings,
        const RobotDimensions& dims, bool recompileLibraries)
        : pinocchioEEKinPtr_(pinocchioEEKinematics.clone()),
          settings_(settings),
          dims_(dims) {
        initialize(dims.x, dims.u, 0, "inertial_alignment_cost", "/tmp/ocs2",
                   recompileLibraries, true);
    }

    InertialAlignmentCost* clone() const override {
        return new InertialAlignmentCost(*pinocchioEEKinPtr_, settings_, dims_,
                                         false);
    }

   protected:
    ad_scalar_t costFunction(ad_scalar_t time, const ad_vector_t& state,
                             const ad_vector_t& input,
                             const ad_vector_t& parameters) const {
        ad_rotmat_t C_we = pinocchioEEKinPtr_->getOrientationCppAd(state);
        ad_vector_t linear_acc =
            pinocchioEEKinPtr_->getAccelerationCppAd(state, input);

        ad_vec3_t gravity = settings_.gravity.cast<ad_scalar_t>();
        ad_vec3_t total_acc = linear_acc - gravity;  // + ddC_we * r_oe_e;

        if (settings_.use_angular_acceleration) {
            ad_vector_t angular_vel =
                pinocchioEEKinPtr_->getAngularVelocityCppAd(state, input);
            ad_vector_t angular_acc =
                pinocchioEEKinPtr_->getAngularAccelerationCppAd(state, input);

            ad_rotmat_t ddC_we = rotation_matrix_second_derivative<ad_scalar_t>(
                C_we, angular_vel, angular_acc);
            ad_vec3_t r_oe_e = settings_.r_oe_e.cast<ad_scalar_t>();

            total_acc += ddC_we * r_oe_e;
        }

        ad_vec3_t total_acc_dir = total_acc / total_acc.norm();

        ad_vec3_t z = ad_vec3_t::UnitZ();
        ad_vec3_t err = z.cross(C_we.transpose() * total_acc_dir);
        // return -z.dot(C_we.transpose() * total_acc_dir);

        // ad_vec3_t err = C_we * z - total_acc_dir;
        return 0.5 * settings_.weight * err.dot(err);
    }

   private:
    InertialAlignmentCost(const InertialAlignmentCost& other) = default;

    std::unique_ptr<PinocchioEndEffectorKinematicsCppAd> pinocchioEEKinPtr_;
    InertialAlignmentSettings settings_;
    RobotDimensions dims_;
};

}  // namespace mobile_manipulator
}  // namespace ocs2
