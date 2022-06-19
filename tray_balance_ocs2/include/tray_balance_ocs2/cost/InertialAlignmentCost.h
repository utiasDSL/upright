#pragma once

#include <memory>

#include <ocs2_core/cost/StateInputCostCppAd.h>
#include <ocs2_pinocchio_interface/PinocchioEndEffectorKinematicsCppAd.h>
#include <ocs2_robotic_tools/end_effector/EndEffectorKinematics.h>

#include <tray_balance_constraints/bounded.h>
#include <tray_balance_ocs2/dynamics/Dimensions.h>
#include <tray_balance_ocs2/types.h>

namespace upright {

struct InertialAlignmentSettings {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    InertialAlignmentSettings() { r_oe_e = Vec3d::Zero(); }

    bool enabled = false;
    bool use_angular_acceleration = false;
    ocs2::scalar_t weight = 1.0;
    Vec3d r_oe_e;  // center of mass
};

inline std::ostream& operator<<(std::ostream& out,
                                const InertialAlignmentSettings& settings) {
    out << "enabled = " << settings.enabled << std::endl
        << "use_angular_acceleration = " << settings.use_angular_acceleration
        << std::endl
        << "weight = " << settings.weight << std::endl
        << "r_oe_e = " << settings.r_oe_e.transpose() << std::endl;
    return out;
}

class InertialAlignmentCost final : public ocs2::StateInputCostCppAd {
   public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    InertialAlignmentCost(
        const ocs2::PinocchioEndEffectorKinematicsCppAd& pinocchioEEKinematics,
        const InertialAlignmentSettings& settings, const Vec3d& gravity,
        const RobotDimensions& dims, bool recompileLibraries)
        : pinocchioEEKinPtr_(pinocchioEEKinematics.clone()),
          settings_(settings),
          gravity_(gravity),
          dims_(dims) {
        initialize(dims.x, dims.u, 0, "inertial_alignment_cost", "/tmp/ocs2",
                   recompileLibraries, true);
    }

    InertialAlignmentCost* clone() const override {
        return new InertialAlignmentCost(*pinocchioEEKinPtr_, settings_,
                                         gravity_, dims_, false);
    }

   protected:
    ocs2::ad_scalar_t costFunction(ocs2::ad_scalar_t time, const VecXad& state,
                                   const VecXad& input,
                                   const VecXad& parameters) const {
        Mat3ad C_we = pinocchioEEKinPtr_->getOrientationCppAd(state);
        Vec3ad linear_acc =
            pinocchioEEKinPtr_->getAccelerationCppAd(state, input);

        Vec3ad gravity = gravity_.cast<ocs2::ad_scalar_t>();
        Vec3ad total_acc = linear_acc - gravity;

        if (settings_.use_angular_acceleration) {
            Vec3ad angular_vel =
                pinocchioEEKinPtr_->getAngularVelocityCppAd(state, input);
            Vec3ad angular_acc =
                pinocchioEEKinPtr_->getAngularAccelerationCppAd(state, input);

            Mat3ad ddC_we =
                rotation_matrix_second_derivative<ocs2::ad_scalar_t>(
                    C_we, angular_vel, angular_acc);
            Vec3ad r_oe_e = settings_.r_oe_e.cast<ocs2::ad_scalar_t>();

            total_acc += ddC_we * r_oe_e;
        }

        Vec3ad total_acc_dir = total_acc / total_acc.norm();

        Vec3ad z = Vec3ad::UnitZ();
        Vec3ad err = z.cross(C_we.transpose() * total_acc_dir);
        return 0.5 * settings_.weight * err.dot(err);
    }

   private:
    InertialAlignmentCost(const InertialAlignmentCost& other) = default;

    std::unique_ptr<ocs2::PinocchioEndEffectorKinematicsCppAd>
        pinocchioEEKinPtr_;
    InertialAlignmentSettings settings_;
    Vec3d gravity_;
    RobotDimensions dims_;
};

}  // namespace upright
