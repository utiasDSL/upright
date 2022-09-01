#include <upright_control/dynamics/dimensions.h>
#include <upright_control/dynamics/util.h>
#include <upright_control/types.h>

#include "upright_control/dynamics/nonholonomic_dynamics.h"

namespace upright {

NonholonomicDynamics::NonholonomicDynamics(
    const std::string& modelName, const RobotDimensions& dims,
    const std::string& modelFolder /*= "/tmp/ocs2"*/,
    bool recompileLibraries /*= true*/, bool verbose /*= true*/)
    : dims_(dims), ocs2::SystemDynamicsBaseAD() {
    initialize(dims.ox(), dims.ou(), modelName, modelFolder, recompileLibraries,
               verbose);
}

VecXad NonholonomicDynamics::systemFlowMap(
    ocs2::ad_scalar_t time, const VecXad& state, const VecXad& input,
    const VecXad& parameters) const {
    ocs2::ad_scalar_t yaw = state(2);
    VecXad v = state.segment(dims_.q, dims_.v);

    VecXad dqdt(dims_.q);
    dqdt << cos(yaw) * v(0), sin(yaw) * v(0), v.tail(dims_.v - 1);

    VecXad dvdt = state.tail(dims_.v);
    VecXad dadt = input.head(dims_.u);

    VecXad dxdt(dims_.x);
    dxdt << dqdt, dvdt, dadt;
    return dxdt;
}

}  // namespace upright
