#pragma once

#include <ocs2_pinocchio_interface/PinocchioStateInputMapping.h>

#include <upright_control/dimensions.h>

namespace upright {

template <typename Scalar>
class CombinedPinocchioMapping final
    : public ocs2::PinocchioStateInputMapping<Scalar> {
   public:
    using Base = ocs2::PinocchioStateInputMapping<Scalar>;
    using VecXd = typename Base::vector_t;
    using MatXd = typename Base::matrix_t;

    // q_indices and v_indices contain the indices that each robot q and v
    // (respectively) starts at in the Pinocchio q and v vectors
    CombinedPinocchioMapping(
        const OptimizationDimensions& dims,
        const std::vector<ocs2::PinocchioStateInputMapping<Scalar>>& mappings,
        const std::vector<size_t>& q_indices,
        const std::vector<size_t>& v_indices)
        : dims_(dims),
          mappings_(mappings),
          q_indices_(q_indices),
          v_indices_(v_indices) {}

    ~CombinedPinocchioMapping() override = default;

    CombinedPinocchioMapping<Scalar>* clone() const override {
        return new CombinedPinocchioMapping<Scalar>(*this);
    }

    VecXd getPinocchioJointPosition(const VecXd& state) const override {
        VecXd q_pin(dims_.q());
        size_t ix = 0;
        for (int i = 0; i < mappings_.size(); ++i) {
            const RobotDimensions& robot_dims = dims_.robot(i);
            VecXd x = state.segment(ix, robot_dims.x);
            q_pin.segment(q_indices_[i], robot_dims.q) =
                mappings_[i].getPinocchioJointPosition(x);

            ix += robot_dims.x;
        }
        return q_pin;
    }

    VecXd getPinocchioJointVelocity(const VecXd& state,
                                    const VecXd& input) const override {
        VecXd v_pin(dims_.v());
        size_t ix = 0;
        size_t iu = 0;
        for (int i = 0; i < mappings_.size(); ++i) {
            const RobotDimensions& robot_dims = dims_.robot(i);
            VecXd x = state.segment(ix, robot_dims.x);
            VecXd u = input.segment(iu, robot_dims.u);
            v_pin.segment(v_indices_[i], robot_dims.v) =
                mappings_[i].getPinocchioJointVelocity(x, u);

            ix += robot_dims.x;
            iu += robot_dims.u;
        }
        return v_pin;
    }

    VecXd getPinocchioJointAcceleration(const VecXd& state,
                                        const VecXd& input) const override {
        VecXd a_pin(dims_.v());
        size_t ix = 0;
        size_t iu = 0;
        for (int i = 0; i < mappings_.size(); ++i) {
            const RobotDimensions& robot_dims = dims_.robot(i);
            VecXd x = state.segment(ix, robot_dims.x);
            VecXd u = input.segment(iu, robot_dims.u);
            a_pin.segment(v_indices[i], robot_dims.v) =
                mappings_[i].getPinocchioJointAcceleration(x, u);

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

            MatXd dfdx_i, dfdu_i;
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
    std::vector<size_t> q_indices_;
    std::vector<size_t> v_indices_;
};

}  // namespace upright
