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

#include <tray_balance_ocs2/definitions.h>

namespace ocs2 {
namespace mobile_manipulator {

using vector3_t = Eigen::Matrix<scalar_t, 3, 1>;
using quaternion_t = Eigen::Quaternion<scalar_t>;

inline vector_t make_target(const vector3_t& ee_position,
                            const quaternion_t& ee_orientation,
                            const vector3_t& obs_position) {
    vector_t target(10);
    target << ee_position, ee_orientation.coeffs(), obs_position;
    return target;
}

inline void set_target_position(vector_t& target, const vector3_t& position) {
    target.head<3>() = position;
}

inline vector3_t get_target_position(const vector_t& target) {
    return target.head<3>();
}

inline void set_target_orientation(vector_t& target,
                                   const quaternion_t& orientation) {
    target.segment<4>(3) = orientation.coeffs();
}

inline quaternion_t get_target_orientation(const vector_t& target) {
    return quaternion_t(target.segment<4>(3));
}

inline void set_obstacle_position(vector_t& target, const vector3_t& position) {
    target.segment<3>(7) = position;
}

inline vector3_t get_obstacle_position(const vector_t& target) {
    return target.segment<3>(7);
}

inline std::pair<vector_t, quaternion_t> interpolateEndEffectorPose(
    scalar_t time, const TargetTrajectories& targetTrajectories) {
    const auto& timeTrajectory = targetTrajectories.timeTrajectory;
    const auto& stateTrajectory = targetTrajectories.stateTrajectory;

    vector_t position;
    quaternion_t orientation;

    if (stateTrajectory.size() > 1) {
        // Normal interpolation case
        int index;
        scalar_t alpha;
        std::tie(index, alpha) =
            LinearInterpolation::timeSegment(time, timeTrajectory);

        const auto& lhs = stateTrajectory[index];
        const auto& rhs = stateTrajectory[index + 1];
        const quaternion_t q_lhs = get_target_orientation(lhs);
        const quaternion_t q_rhs = get_target_orientation(rhs);

        position = alpha * get_target_position(lhs) +
                   (1.0 - alpha) * get_target_position(rhs);
        orientation = q_lhs.slerp((1.0 - alpha), q_rhs);
    } else {  // stateTrajectory.size() == 1
        position = get_target_position(stateTrajectory.front());
        orientation = get_target_orientation(stateTrajectory.front());
    }

    return {position, orientation};
}

// Interpolate position of obstacle over time.
inline vector3_t interpolate_obstacle_position(
    scalar_t time, const TargetTrajectories& targetTrajectories) {
    const auto& timeTrajectory = targetTrajectories.timeTrajectory;
    const auto& stateTrajectory = targetTrajectories.stateTrajectory;

    vector3_t position;

    if (stateTrajectory.size() > 1) {
        // Normal interpolation case
        int index;
        scalar_t alpha;
        std::tie(index, alpha) =
            LinearInterpolation::timeSegment(time, timeTrajectory);

        const auto& lhs = stateTrajectory[index];
        const auto& rhs = stateTrajectory[index + 1];

        position = alpha * get_obstacle_position(lhs) +
                   (1.0 - alpha) * get_obstacle_position(rhs);
    } else {  // stateTrajectory.size() == 1
        position = get_obstacle_position(stateTrajectory.front());
    }

    return position;
}

}  // namespace mobile_manipulator
}  // namespace ocs2
