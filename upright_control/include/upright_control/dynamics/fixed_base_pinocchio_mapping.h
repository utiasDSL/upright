#pragma once

#include <ocs2_pinocchio_interface/PinocchioStateInputMapping.h>

#include <upright_control/dimensions.h>

namespace upright {

template <typename Scalar>
class FixedBasePinocchioMapping final
    : public ocs2::PinocchioStateInputMapping<Scalar> {
   public:
    using Base = ocs2::PinocchioStateInputMapping<Scalar>;
    using VecXd = typename Base::vector_t;
    using MatXd = typename Base::matrix_t;

    FixedBasePinocchioMapping(const RobotDimensions& dims) : dims_(dims) {}

    ~FixedBasePinocchioMapping() override = default;

    FixedBasePinocchioMapping<Scalar>* clone() const override {
        return new FixedBasePinocchioMapping<Scalar>(*this);
    }

    VecXd getPinocchioJointPosition(const VecXd& state) const override {
        return state.head(dims_.q);
    }

    VecXd getPinocchioJointVelocity(const VecXd& state,
                                    const VecXd& input) const override {
        return state.segment(dims_.q, dims_.v);
    }

    VecXd getPinocchioJointAcceleration(const VecXd& state,
                                        const VecXd& input) const override {
        return state.tail(dims_.v);
    }

    // NOTE: maps the Jacobians of an arbitrary function f w.r.t q and v
    // (generalized positions and velocities), as provided by Pinocchio as Jq
    // and Jv, to the Jacobian of the state dfdx and Jacobian of the input
    // dfdu.
    std::pair<MatXd, MatXd> getOcs2Jacobian(const VecXd& state, const MatXd& Jq,
                                            const MatXd& Jv) const override {
        const auto output_dim = Jq.rows();
        MatXd dfdx(output_dim, Jq.cols() + Jv.cols() + dims_.v);
        dfdx << Jq, Jv, MatXd::Zero(output_dim, dims_.v);

        // NOTE: this isn't used for collision avoidance (which is the only
        // place this method is called)
        MatXd dfdu(output_dim, dims_.u);
        dfdu.setZero();

        return {dfdx, dfdu};
    }

   private:
    RobotDimensions dims_;
};

template <typename Scalar>
class ObstaclePinocchioMapping final
    : public ocs2::PinocchioStateInputMapping<Scalar> {
   public:
    using Base = ocs2::PinocchioStateInputMapping<Scalar>;
    using VecXd = typename Base::vector_t;
    using MatXd = typename Base::matrix_t;

    ObstaclePinocchioMapping(const RobotDimensions& dims) : dims_(dims) {}

    ~ObstaclePinocchioMapping() override = default;

    ObstaclePinocchioMapping<Scalar>* clone() const override {
        return new ObstaclePinocchioMapping<Scalar>(*this);
    }

    VecXd getPinocchioJointPosition(const VecXd& state) const override {
        return state.head(dims_.q);
    }

    VecXd getPinocchioJointVelocity(const VecXd& state,
                                    const VecXd& input) const override {
        return state.segment(dims_.q, dims_.v);
    }

    VecXd getPinocchioJointAcceleration(const VecXd& state,
                                        const VecXd& input) const override {
        return state.tail(dims_.v);
    }

    // NOTE: maps the Jacobians of an arbitrary function f w.r.t q and v
    // (generalized positions and velocities), as provided by Pinocchio as Jq
    // and Jv, to the Jacobian of the state dfdx and Jacobian of the input
    // dfdu.
    std::pair<MatXd, MatXd> getOcs2Jacobian(const VecXd& state, const MatXd& Jq,
                                            const MatXd& Jv) const override {
        const auto output_dim = Jq.rows();
        MatXd dfdx(output_dim, Jq.cols() + Jv.cols() + dims_.v);
        dfdx << Jq, Jv, MatXd::Zero(output_dim, dims_.v);

        // NOTE: this isn't used for collision avoidance (which is the only
        // place this method is called)
        MatXd dfdu(output_dim, dims_.u);
        dfdu.setZero();

        return {dfdx, dfdu};
    }

   private:
    RobotDimensions dims_;
};

template <typename Scalar>
class CombinedPinocchioMapping final
    : public ocs2::PinocchioStateInputMapping<Scalar> {
   public:
    using Base = ocs2::PinocchioStateInputMapping<Scalar>;
    using VecXd = typename Base::vector_t;
    using MatXd = typename Base::matrix_t;

    // TODO we need a joint index mapping from q -> q_pin and v -> v_pin
    // could avoid this for now by putting the projectile first
    CombinedPinocchioMapping(
        const OptimizationDimensions& dims,
        const std::vector<ocs2::PinocchioStateInputMapping<Scalar>>& mappings)
        : dims_(dims), mappings_(mappings) {}

    ~CombinedPinocchioMapping() override = default;

    CombinedPinocchioMapping<Scalar>* clone() const override {
        return new CombinedPinocchioMapping<Scalar>(*this);
    }

    VecXd getPinocchioJointPosition(const VecXd& state) const override {
        VecXd q_pin(dims_.q());
        size_t iq = 0;
        size_t ix = 0;
        for (int i = 0; i < mappings_.size(); ++i) {
            const RobotDimensions& robot_dims = dims_.robot(i);
            VecXd x = state.segment(ix, robot_dims.x);
            q_pin.segment(iq, robot_dims.q) =
                mappings_[i].getPinocchioJointPosition(x);

            iq += robot_dims.q;
            ix += robot_dims.x;
        }
        return q_pin;
    }

    VecXd getPinocchioJointVelocity(const VecXd& state,
                                    const VecXd& input) const override {
        VecXd v_pin(dims_.v());
        size_t iv = 0;
        size_t ix = 0;
        size_t iu = 0;
        for (int i = 0; i < mappings_.size(); ++i) {
            const RobotDimensions& robot_dims = dims_.robot(i);
            VecXd x = state.segment(ix, robot_dims.x);
            VecXd u = input.segment(iu, robot_dims.u);
            v_pin.segment(iv, robot_dims.v) =
                mappings_[i].getPinocchioJointVelocity(x, u);

            iv += robot_dims.v;
            ix += robot_dims.x;
            iu += robot_dims.u;
        }
        return v_pin;
    }

    VecXd getPinocchioJointAcceleration(const VecXd& state,
                                        const VecXd& input) const override {
        VecXd a_pin(dims_.v());
        size_t iv = 0;
        size_t ix = 0;
        size_t iu = 0;
        for (int i = 0; i < mappings_.size(); ++i) {
            const RobotDimensions& robot_dims = dims_.robot(i);
            VecXd x = state.segment(ix, robot_dims.x);
            VecXd u = input.segment(iu, robot_dims.u);
            a_pin.segment(iv, robot_dims.v) =
                mappings_[i].getPinocchioJointAcceleration(x, u);

            iv += robot_dims.v;
            ix += robot_dims.x;
            iu += robot_dims.u;
        }
        return a_pin;
    }

    // NOTE: maps the Jacobians of an arbitrary function f w.r.t q and v
    // (generalized positions and velocities), as provided by Pinocchio as Jq
    // and Jv, to the Jacobian of the state dfdx and Jacobian of the input
    // dfdu.
    std::pair<MatXd, MatXd> getOcs2Jacobian(
        const VecXd& state, const MatXd& Jq_pin,
        const MatXd& Jv_pin) const override {
        const auto output_dim = Jq_pin.rows();
        MatXd dfdx(output_dim, dims_.x());
        MatXd dfdu(output_dim, dims_.u());

        size_t iq = 0;
        size_t iv = 0;
        size_t ix = 0;
        size_t iu = 0;
        for (int i = 0; i < mappings_.size(); ++i) {
            const RobotDimensions& robot_dims = dims_.robot(i);
            VecXd x = state.segment(ix, robot_dims.x);
            MatXd Jq_pin_i = Jq_pin.middleCols(iq, robot_dims.q);
            MatXd Jv_pin_i = Jv_pin.middleCols(iv, robot_dims.v);

            std::tie(dfdx_i, dfdu_i) =
                mappings_[i].getOcs2Jacobian(x, Jq_pin_i, Jv_pin_i);
            dfdx.middleCols(ix, robot_dims.x) = dfdx_i;
            dfdu.middleCols(iu, robot_dims.u) = dfdu_i;

            iq += robot_dims.q;
            iv += robot_dims.v;
            ix += robot_dims.x;
            iu += robot_dims.u;
        }
        return {dfdx, dfdu};
    }

   private:
    OptimizationDimensions dims_;
    std::vector<ocs2::PinocchioStateInputMapping<Scalar>> mappings_;
};

}  // namespace upright
