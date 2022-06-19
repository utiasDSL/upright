#include <tray_balance_ocs2/dynamics/Dimensions.h>
#include <tray_balance_ocs2/types.h>
#include <tray_balance_ocs2/util.h>

#include "tray_balance_ocs2/dynamics/MobileManipulatorDynamics.h"

namespace upright {

MobileManipulatorDynamics::MobileManipulatorDynamics(
    const std::string& modelName, const RobotDimensions& dims,
    const std::string& modelFolder /*= "/tmp/ocs2"*/,
    bool recompileLibraries /*= true*/, bool verbose /*= true*/)
    : dims_(dims), ocs2::SystemDynamicsBaseAD() {
    initialize(dims.x, dims.u, modelName, modelFolder, recompileLibraries,
               verbose);
}

VecXad MobileManipulatorDynamics::systemFlowMap(
    ocs2::ad_scalar_t time, const VecXad& state, const VecXad& input,
    const VecXad& parameters) const {
    Mat2ad C_wb = base_rotation_matrix(state);

    // convert base velocity from body frame to world frame
    VecXad v_body = state.segment(dims_.q, dims_.v);
    VecXad dqdt(dims_.q);
    dqdt << C_wb * v_body.head(2), v_body.tail(dims_.v - 2);

    VecXad dvdt = state.tail(dims_.v);
    VecXad dadt = input;
    // dvdt(1) = 0;  // nonholonomic

    VecXad dxdt(dims_.x);
    dxdt << dqdt, dvdt, dadt;
    return dxdt;
}

}  // namespace upright
