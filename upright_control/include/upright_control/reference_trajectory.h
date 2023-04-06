#pragma once

#include <ocs2_core/misc/LinearInterpolation.h>
#include <ocs2_core/reference/TargetTrajectories.h>

#include <upright_control/types.h>

namespace upright {

inline Vec3d get_target_position(const VecXd& target) {
    return target.head<3>();
}

inline Quatd get_target_orientation(const VecXd& target) {
    return Quatd(target.segment<4>(3));
}

inline std::pair<VecXd, Quatd> interpolate_end_effector_pose(
    ocs2::scalar_t time, const ocs2::TargetTrajectories& targetTrajectories) {
    const auto& timeTrajectory = targetTrajectories.timeTrajectory;
    const auto& stateTrajectory = targetTrajectories.stateTrajectory;

    VecXd position;
    Quatd orientation;

    if (stateTrajectory.size() > 1) {
        // Normal interpolation case
        int index;
        ocs2::scalar_t alpha;
        std::tie(index, alpha) =
            ocs2::LinearInterpolation::timeSegment(time, timeTrajectory);

        const auto& lhs = stateTrajectory[index];
        const auto& rhs = stateTrajectory[index + 1];
        const Quatd q_lhs = get_target_orientation(lhs);
        const Quatd q_rhs = get_target_orientation(rhs);

        position = alpha * get_target_position(lhs) +
                   (1.0 - alpha) * get_target_position(rhs);
        orientation = q_lhs.slerp((1.0 - alpha), q_rhs);
    } else {  // stateTrajectory.size() == 1
        position = get_target_position(stateTrajectory.front());
        orientation = get_target_orientation(stateTrajectory.front());
    }

    return {position, orientation};
}

}  // namespace upright
