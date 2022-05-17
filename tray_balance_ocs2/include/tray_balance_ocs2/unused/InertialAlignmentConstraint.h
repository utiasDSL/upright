#pragma once

#include <memory>

#include <ocs2_core/constraint/StateConstraintCppAd.h>
#include <ocs2_oc/synchronized_module/ReferenceManager.h>
#include <ocs2_pinocchio_interface/PinocchioEndEffectorKinematicsCppAd.h>
#include <ocs2_robotic_tools/common/SkewSymmetricMatrix.h>
#include <ocs2_robotic_tools/end_effector/EndEffectorKinematics.h>

#include <tray_balance_constraints/bounded.h>

#include <tray_balance_ocs2/dynamics/Dimensions.h>

namespace ocs2 {
namespace mobile_manipulator {

class InertialAlignmentConstraint final : public StateConstraintCppAd {
   public:
    using ad_quaternion_t =
        PinocchioEndEffectorKinematicsCppAd::ad_quaternion_t;
    using ad_rotmat_t = PinocchioEndEffectorKinematicsCppAd::ad_rotmat_t;

    using ad_vec2_t = Eigen::Matrix<ad_scalar_t, 2, 1>;
    using ad_vec3_t = Eigen::Matrix<ad_scalar_t, 3, 1>;
    using ad_mat3_t = Eigen::Matrix<ad_scalar_t, 3, 3>;

    InertialAlignmentConstraint(
        const PinocchioEndEffectorKinematicsCppAd& pinocchioEEKinematics,
        const RobotDimensions& dims, bool recompileLibraries)
        : StateConstraintCppAd(ConstraintOrder::Linear),
          pinocchioEEKinPtr_(pinocchioEEKinematics.clone()),
          dims_(dims) {
        if (pinocchioEEKinematics.getIds().size() != 1) {
            throw std::runtime_error(
                "[TrayBalanaceConstraint] endEffectorKinematics has wrong "
                "number of end effector IDs.");
        }

        // compile the CppAD library
        initialize(dims.x, 0, "inertial_alignment_constraint",
                   "/tmp/ocs2", recompileLibraries, true);
    }

    InertialAlignmentConstraint* clone() const override {
        // Always pass recompileLibraries = false to avoid recompiling the same
        // library just because this object is cloned.
        return new InertialAlignmentConstraint(*pinocchioEEKinPtr_, dims_, false);
    }

    size_t getNumConstraints(scalar_t time) const override {
        return 3;
    }

    size_t getNumConstraints() const { return getNumConstraints(0); }

    vector_t getParameters(scalar_t time) const override {
        // Parameters are constant for now
        // return params_;
        return vector_t(0);
    }

   protected:
    ad_vector_t constraintFunction(
        ad_scalar_t time, const ad_vector_t& state,
        const ad_vector_t& parameters) const override {
        ad_rotmat_t C_we = pinocchioEEKinPtr_->getOrientationCppAd(state);
        ad_vector_t input = ad_vector_t::Zero(dims_.u);
        ad_vector_t linear_acc =
            pinocchioEEKinPtr_->getAccelerationCppAd(state, input);

        ad_vec3_t z = ad_vec3_t::UnitZ();
        ad_vec3_t gravity = ad_scalar_t(-9.81) * z;
        ad_vec3_t total_acc = linear_acc - gravity;
        ad_vec3_t total_acc_dir = total_acc / total_acc.norm();
        ad_vec3_t err = z.cross(C_we.transpose() * total_acc_dir);
        return err;
        // ad_vector_t con(1);
        // con << z.dot(C_we.transpose() * total_acc) - total_acc.norm();
        // return con;
    }

   private:
    InertialAlignmentConstraint(const InertialAlignmentConstraint& other) =
        default;

    std::unique_ptr<PinocchioEndEffectorKinematicsCppAd> pinocchioEEKinPtr_;
    RobotDimensions dims_;
};

}  // namespace mobile_manipulator
}  // namespace ocs2
