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

#include <tray_balance_ocs2/dynamics/Dimensions.h>
#include <tray_balance_ocs2/util.h>

namespace ocs2 {
namespace mobile_manipulator {

template <typename Scalar>
class MobileManipulatorPinocchioMapping final
    : public PinocchioStateInputMapping<Scalar> {
   public:
    using Base = PinocchioStateInputMapping<Scalar>;
    using typename Base::matrix_t;
    using typename Base::vector_t;

    MobileManipulatorPinocchioMapping(const RobotDimensions& dims)
        : dims_(dims) {}

    ~MobileManipulatorPinocchioMapping() override = default;

    MobileManipulatorPinocchioMapping<Scalar>* clone() const override {
        return new MobileManipulatorPinocchioMapping<Scalar>(*this);
    }

    vector_t getPinocchioJointPosition(const vector_t& state) const override {
        return state.head(dims_.q);
    }

    vector_t getPinocchioJointVelocity(const vector_t& state,
                                       const vector_t& input) const override {
        Eigen::Matrix<Scalar, 2, 2> C_wb = base_rotation_matrix(state);

        // convert velocity from body frame to world frame
        vector_t v_body = state.segment(dims_.q, dims_.v);
        vector_t v_world(dims_.v);
        v_world << C_wb * v_body.head(2), v_body.tail(dims_.v - 2);
        return v_world;
    }

    vector_t getPinocchioJointAcceleration(
        const vector_t& state, const vector_t& input) const override {
        Eigen::Matrix<Scalar, 2, 2> C_wb = base_rotation_matrix(state);

        // convert acceleration input from body frame to world frame
        vector_t a_body = state.tail(dims_.v);
        vector_t a_world(dims_.v);
        a_world << C_wb * a_body.head(2), a_body.tail(dims_.v - 2);
        return a_world;
    }

    // NOTE: maps the Jacobians of an arbitrary function f w.r.t q and v
    // (generalized positions and velocities), as provided by Pinocchio as Jq
    // and Jv, to the Jacobian of the state dfdx and Jacobian of the input
    // dfdu.
    std::pair<matrix_t, matrix_t> getOcs2Jacobian(
        const vector_t& state, const matrix_t& Jq,
        const matrix_t& Jv) const override {
        // TODO not correct now

        // Jacobian of Pinocchio joint velocities v_pin w.r.t. actual state
        // velocities v
        Eigen::Matrix<Scalar, 2, 2> C_wb = base_rotation_matrix(state);
        matrix_t dv_pin_dv = matrix_t::Identity(dims_.v, dims_.v);
        dv_pin_dv.template topLeftCorner<2, 2>() = C_wb;

        const auto output_dim = Jq.rows();
        matrix_t dfdx(output_dim, Jq.cols() + Jv.cols());
        dfdx << Jq, Jv * dv_pin_dv;

        // NOTE: this isn't used for collision avoidance (which is the only
        // place this method is called)
        matrix_t dfdu(output_dim, dims_.u);
        dfdu.setZero();

        return {dfdx, dfdu};
    }

   private:
    RobotDimensions dims_;
};

}  // namespace mobile_manipulator
}  // namespace ocs2
