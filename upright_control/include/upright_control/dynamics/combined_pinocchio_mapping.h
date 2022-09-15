#pragma once

#include <ocs2_pinocchio_interface/PinocchioStateInputMapping.h>

#include <upright_control/dimensions.h>

namespace upright {

// TODO: do we want to make a simplified version of the pinocchio mapping, like
// I've done for the dynamics?

template <typename Mapping, typename Scalar>
class CombinedPinocchioMapping final
    : public ocs2::PinocchioStateInputMapping<Scalar> {
   public:
    using VecXs = VecX<Scalar>;
    using MatXs = MatX<Scalar>;

    CombinedPinocchioMapping(const OptimizationDimensions& dims)
        : dims_(dims), robot_mapping_(dims.robot) {}

    ~CombinedPinocchioMapping() override = default;

    CombinedPinocchioMapping<Scalar>* clone() const override {
        return new CombinedPinocchioMapping<Mapping, Scalar>(*this);
    }

    VecXs getPinocchioJointPosition(const VecXs& state) const override {
        VecXs q_pin(dims_.q());

        // For now, we assume all obstacles go first in list of q, v
        for (int i = 0; i < dims_.o; ++i) {
            VecXs x_obs = state.segment(dims_.robot.x + i * 9, 9);
            q_pin.segment(i * 3, 3) =
                obstacle_mapping_.getPinocchioJointPosition(x_obs);
        }

        // Then we add on the robot q
        VecXs x_robot = state.head(dims.robot.x);
        q_pin.tail(dims_.robot.q) =
            robot_mapping_.getPinocchioJointPosition(x_robot);

        return q_pin;
    }

    VecXs getPinocchioJointVelocity(const VecXs& state,
                                    const VecXs& input) const override {
        VecXs v_pin(dims_.v());
        VecXs u_obs = VecXs::Zero(3);  // Obstacles have no input

        for (int i = 0; i < dims_.o; ++i) {
            VecXs x_obs = state.segment(dims_.robot.x + i * 9, 9);
            v_pin.segment(i * 3, 3) =
                obstacle_mapping_.getPinocchioJointVelocity(x_obs, u_obs);
        }

        // Then we add on the robot v
        VecXs x_robot = state.head(dims.robot.x);
        v_pin.tail(dims_.robot.v) =
            robot_mapping_.getPinocchioJointVelocity(x_robot);

        return v_pin;
    }

    VecXs getPinocchioJointAcceleration(const VecXs& state,
                                        const VecXs& input) const override {
        VecXs a_pin(dims_.v());
        VecXs u_obs = VecXs::Zero(3);  // Obstacles have no input

        for (int i = 0; i < dims_.o; ++i) {
            VecXs x_obs = state.segment(dims_.robot.x + i * 9, 9);
            a_pin.segment(i * 3, 3) =
                obstacle_mapping_.getPinocchioJointAcceleration(x_obs, u_obs);
        }

        // Then we add on the robot a
        VecXs x_robot = state.head(dims.robot.x);
        a_pin.tail(dims_.robot.v) =
            robot_mapping_.getPinocchioJointAcceleration(x_robot);

        return a_pin;
    }

    // NOTE: maps the Jacobians of an arbitrary function f w.r.t q and v
    // (generalized positions and velocities), as provided by Pinocchio as Jq
    // and Jv, to the Jacobian of the state dfdx and Jacobian of the input
    // dfdu.
    std::pair<MatXs, MatXs> getOcs2Jacobian(
        const VecXs& state, const MatXs& Jq_pin,
        const MatXs& Jv_pin) const override {

        const auto output_dim = Jq_pin.rows();
        MatXs dfdx(output_dim, dims_.x());
        MatXs dfdu(output_dim, dims_.u());

        for (int i = 0; i < dims_.o; ++i) {
            VecXs x_obs = state.segment(dims_.robot.x + i * 9, 9);
            MatXs Jq_pin_obs = Jq_pin.middleCols(i * 3, 3);
            MatXs Jv_pin_obs = Jv_pin.middleCols(i * 3, 3);

            MatXs dfdx_obs;
            std::tie(dfdx_obs, std::ignore) =
                obstacle_mapping_.getOcs2Jacobian(x_obs, Jq_pin_obs, Jv_pin_obs);

            // Obstacles have no input, so no dfdu
            dfdx.middleCols(dims_.robot.x + i * 9, 9) = dfdx_i;
        }

        VecXs x_robot = state.head(dims_.robot.x);
        MatXs Jq_pin_robot = Jq_pin.leftCols(dims_.robot.q);
        MatXs Jv_pin_robot = Jv_pin.leftCols(dims_.robot.v);
        MatXs dfdx_robot, dfdu_robot;
        std::tie(dfdx_robot, dfdu_robot) =
            robot_mapping_.getOcs2Jacobian(x_robot, Jq_pin_robot, Jv_pin_robot);

        dfdx.leftCols(dims_.robot.x) = dfdx_robot;
        dfdu.leftCols(dims_.robot.u) = dfdu_robot;

        return {dfdx, dfdu};
    }

   private:
    OptimizationDimensions dims_;
    Mapping<Scalar> robot_mapping_;
    ObstacleMapping<Scalar> obstacle_mapping_;
};

}  // namespace upright
