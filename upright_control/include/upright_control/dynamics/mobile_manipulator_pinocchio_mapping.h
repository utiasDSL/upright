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

#include <iostream>

#include <ocs2_pinocchio_interface/PinocchioStateInputMapping.h>

#include <upright_control/dynamics/dimensions.h>
#include <upright_control/dynamics/util.h>

namespace upright {


template <typename Scalar>
class MobileManipulatorPinocchioMapping final
    : public ocs2::PinocchioStateInputMapping<Scalar> {
   public:
    using Base = ocs2::PinocchioStateInputMapping<Scalar>;
    using VecXd = typename Base::vector_t;
    using MatXd = typename Base::matrix_t;

    MobileManipulatorPinocchioMapping(const RobotDimensions& dims)
        : dims_(dims) {}

    ~MobileManipulatorPinocchioMapping() override = default;

    MobileManipulatorPinocchioMapping<Scalar>* clone() const override {
        return new MobileManipulatorPinocchioMapping<Scalar>(*this);
    }

    VecXd getPinocchioJointPosition(const VecXd& state) const override {
        // VecXd q = state.head(dims_.q);
        // VecXd q_pin(dims_.q + 1);
        // q_pin << q.head(2), cos(q(2)), sin(q(2)), q.tail(dims_.q - 3);
        // return q_pin;
        return state.head(dims_.q);
    }

    VecXd getPinocchioJointVelocity(const VecXd& state,
                                    const VecXd& input) const override {
        Mat2<Scalar> C_wb = base_rotation_matrix(state);

        // convert velocity from body frame to world frame
        VecXd v_body = state.segment(dims_.q, dims_.v);
        VecXd v_world(dims_.v);
        v_world << C_wb * v_body.head(2), v_body.tail(dims_.v - 2);
        return v_world;
    }

    VecXd getPinocchioJointAcceleration(const VecXd& state,
                                        const VecXd& input) const override {
        Mat2<Scalar> C_wb = base_rotation_matrix(state);

        // convert acceleration input from body frame to world frame
        VecXd a_body = state.tail(dims_.v);
        VecXd a_world(dims_.v);
        a_world << C_wb * a_body.head(2), a_body.tail(dims_.v - 2);
        return a_world;
    }

    // NOTE: maps the Jacobians of an arbitrary function f w.r.t q and v
    // (generalized positions and velocities), as provided by Pinocchio as Jq
    // and Jv, to the Jacobian of the state dfdx and Jacobian of the input
    // dfdu.
    std::pair<MatXd, MatXd> getOcs2Jacobian(const VecXd& state, const MatXd& Jq_pin,
                                            const MatXd& Jv_pin) const override {
        // Jacobian of Pinocchio joint velocities v_pin w.r.t. actual state
        // velocities v
        Mat2<Scalar> C_wb = base_rotation_matrix(state);
        MatXd dv_pin_dv = MatXd::Identity(dims_.v, dims_.v);
        dv_pin_dv.template topLeftCorner<2, 2>() = C_wb;

        const auto nf = Jq_pin.rows();

        // MatXd dq_pin_dq = MatXd::Zero(dims_.q + 1, dims_.q);
        // dq_pin_dq.template topLeftCorner<2, 2>() = MatXd::Identity(2, 2);
        // dq_pin_dq(2, 2) = -sin(state(2));
        // dq_pin_dq(3, 2) = cos(state(2));
        // dq_pin_dq.template bottomRightCorner<6, 6>() = MatXd::Identity(6, 6);

        MatXd Jq = Jq_pin;
        // MatXd Jq = Jq_pin * dq_pin_dq;
        MatXd Jv = Jv_pin * dv_pin_dv;
        MatXd Ja = MatXd::Zero(nf, Jv.cols());

        // State Jacobian
        MatXd dfdx(nf, Jq.cols() + Jv.cols() + Ja.cols());
        dfdx << Jq, Jv, Ja;

        // Input Jacobian
        // NOTE: this isn't used for collision avoidance (which is the only
        // place this method is called)
        MatXd dfdu = MatXd::Zero(nf, dims_.u);

        return {dfdx, dfdu};
    }

   private:
    RobotDimensions dims_;
};

}  // namespace upright