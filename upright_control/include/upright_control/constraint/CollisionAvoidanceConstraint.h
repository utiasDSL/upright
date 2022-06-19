/******************************************************************************
Copyright (c) 2020, Farbod Farshidian. All rights reserved.

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

#include <ocs2_self_collision/SelfCollisionConstraint.h>
#include <upright_control/types.h>

namespace upright {

template <typename Scalar>
struct CollisionSphere {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    // Empty constructor required for binding as opaque vector type.
    CollisionSphere() {
        name = "";
        parent_frame_name = "";
        offset = Vec3<Scalar>::Zero();
        radius = 0;
    }

    CollisionSphere(const std::string& name,
                    const std::string& parent_frame_name,
                    const Eigen::Matrix<Scalar, 3, 1>& offset,
                    const Scalar radius)
        : name(name),
          parent_frame_name(parent_frame_name),
          offset(offset),
          radius(radius) {}

    // Name of this collision sphere.
    std::string name;

    // Name of the robot joint this collision sphere is attached to.
    std::string parent_frame_name;

    // Offset from that joint (in the joint's frame).
    Vec3<Scalar> offset;

    // Radius of this collision sphere.
    Scalar radius;
};

struct CollisionAvoidanceSettings {
    bool enabled = false;

    // List of pairs of collision objects to check
    std::vector<std::pair<std::string, std::string>> collision_link_pairs;

    // Minimum distance allowed between collision objects
    ocs2::scalar_t minimum_distance = 0;

    // Relaxed barrier function parameters
    ocs2::scalar_t mu = 1e-2;
    ocs2::scalar_t delta = 1e-3;

    // Extra collision spheres to attach to the robot body for collision
    // avoidance.
    std::vector<CollisionSphere<ocs2::scalar_t>> extra_spheres;
};

// class CollisionAvoidanceConstraint final
//     : public ocs2::SelfCollisionConstraint {
//    public:
//     CollisionAvoidanceConstraint(
//         const ocs2::PinocchioStateInputMapping<ocs2::scalar_t>& mapping,
//         ocs2::PinocchioGeometryInterface pinocchioGeometryInterface,
//         ocs2::scalar_t minimumDistance)
//         : SelfCollisionConstraint(
//               mapping, std::move(pinocchioGeometryInterface), minimumDistance) {
//     }
//
//     ~CollisionAvoidanceConstraint() override = default;
//
//     CollisionAvoidanceConstraint(const CollisionAvoidanceConstraint& other) =
//         default;
//
//     CollisionAvoidanceConstraint* clone() const {
//         return new CollisionAvoidanceConstraint(*this);
//     }
//
//     const ocs2::PinocchioInterface& getPinocchioInterface(
//         const ocs2::PreComputation& preComputation) const override {
//         return ocs2::cast<ocs2::MobileManipulatorPreComputation>(preComputation)
//             .getPinocchioInterface();
//     }
// };

}  // namespace upright
