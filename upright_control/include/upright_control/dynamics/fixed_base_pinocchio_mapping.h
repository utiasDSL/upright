#pragma once

#include <ocs2_pinocchio_interface/PinocchioStateInputMapping.h>

#include <upright_control/dynamics/dimensions.h>

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

}  // namespace upright
