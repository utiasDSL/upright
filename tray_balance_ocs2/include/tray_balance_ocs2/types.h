#pragma once

#include <ocs2_core/Types.h>
#include <ocs2_core/automatic_differentiation/Types.h>
#include <tray_balance_constraints/types.h>

// Specialize the template types from tray_balance_constraints with the scalars
// from OCS2.

namespace ocs2 {
namespace mobile_manipulator {

// Normal scalar (double)
using VecXd = VecX<scalar_t>;
using MatXd = MatX<scalar_t>;
using Vec2d = Vec2<scalar_t>;
using Mat2d = Mat2<scalar_t>;
using Vec3d = Vec3<scalar_t>;
using Mat3d = Mat3<scalar_t>;

// Auto-diff scalar
using VecXad = VecX<ad_scalar_t>;
using MatXad = MatX<ad_scalar_t>;
using Vec2ad = Vec2<ad_scalar_t>;
using Mat2ad = Mat2<ad_scalar_t>;
using Vec3ad = Vec3<ad_scalar_t>;
using Mat3ad = Mat3<ad_scalar_t>;

}  // namespace mobile_manipulator
}  // namespace ocs2
