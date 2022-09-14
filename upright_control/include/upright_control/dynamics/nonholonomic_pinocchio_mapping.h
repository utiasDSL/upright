/******************************************************************************
Copyright (c) 2021, Farbod Farshidian. All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
******************************************************************************/

#pragma once

#include <ocs2_pinocchio_interface/PinocchioStateInputMapping.h>

#include <upright_control/dimensions.h>

namespace upright {


// Pinocchio mapping for a nonholonomic mobile manipulator.
// State x = [q, v, a], where q starts with the planar (x, y, yaw) pose of the
// base. In contrast, v and a are expressed in the body frame and begin with a
// linear forward component followed by the angular component.
template <typename Scalar>
class NonholonomicPinocchioMapping final
    : public ocs2::PinocchioStateInputMapping<Scalar> {
   public:
    using Base = ocs2::PinocchioStateInputMapping<Scalar>;
    using VecXd = typename Base::vector_t;
    using MatXd = typename Base::matrix_t;

    NonholonomicPinocchioMapping(const RobotDimensions& dims)
        : dims_(dims) {}

    ~NonholonomicPinocchioMapping() override = default;

    NonholonomicPinocchioMapping<Scalar>* clone() const override {
        return new NonholonomicPinocchioMapping<Scalar>(*this);
    }

    VecXd getPinocchioJointPosition(const VecXd& state) const override {
        return state.head(dims_.q);
    }

    VecXd getPinocchioJointVelocity(const VecXd& state,
                                    const VecXd& input) const override {
        ocs2::scalar_t yaw = state(2);
        VecXd v = state.segment(dims_.q, dims_.v);

        VecXd v_pin(dims_.v);
        v_pin << cos(yaw) * v(0), sin(yaw) * v(0), v.tail(dims_.v - 1);
        return v_pin;
    }

    VecXd getPinocchioJointAcceleration(const VecXd& state,
                                        const VecXd& input) const override {
        ocs2::scalar_t yaw = state(2);
        VecXd a = state.tail(dims_.v);

        VecXd a_pin(dims_.v);
        a_pin << cos(yaw) * a(0), sin(yaw) * a(0), a.tail(dims_.v - 1);
        return a_pin;
    }

    // NOTE: maps the Jacobians of an arbitrary function f w.r.t q and v
    // (generalized positions and velocities), as provided by Pinocchio as Jq
    // and Jv, to the Jacobian of the state dfdx and Jacobian of the input
    // dfdu.
    std::pair<MatXd, MatXd> getOcs2Jacobian(const VecXd& state, const MatXd& Jq_pin,
                                            const MatXd& Jv_pin) const override {
        // Jacobian of Pinocchio joint velocities v_pin w.r.t. actual state
        // velocities v: dfdv = dfdv_pin / dv_pin_dv
        ocs2::scalar_t yaw = state(2);
        MatXd dv_pin_dv = MatXd::Zero(dims_.v + 1, dims_.v);
        dv_pin_dv(0, 0) = cos(yaw);
        dv_pin_dv(1, 0) = sin(yaw);
        dv_pin_dv.template bottomRightCorner<dims_.v - 1, dims_.v - 1>().setIdentity();

        // Function output dimension
        const auto nf = Jq_pin.rows();

        MatXd Jq = Jq_pin;
        MatXd Jv = Jv_pin * dv_pin_dv;
        MatXd Ja = MatXd::Zero(nf, Jv.cols());

        // State Jacobian
        MatXd dfdx(nf, Jq.cols() + Jv.cols() + Ja.cols());
        dfdx << Jq, Jv, Ja;

        // Input Jacobian
        MatXd dfdu = MatXd::Zero(nf, dims_.u);

        return {dfdx, dfdu};
    }

   private:
    RobotDimensions dims_;
};

}  // namespace upright
