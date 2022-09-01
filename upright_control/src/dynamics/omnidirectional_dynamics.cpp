#include <upright_control/dynamics/dimensions.h>
#include <upright_control/dynamics/util.h>
#include <upright_control/types.h>

#include "upright_control/dynamics/omnidirectional_dynamics.h"

namespace upright {

OmnidirectionalDynamics::OmnidirectionalDynamics(
    const std::string& modelName, const RobotDimensions& dims,
    const std::string& modelFolder /*= "/tmp/ocs2"*/,
    bool recompileLibraries /*= true*/, bool verbose /*= true*/)
    : dims_(dims), ocs2::SystemDynamicsBaseAD() {
    initialize(dims.ox(), dims.ou(), modelName, modelFolder, recompileLibraries,
               verbose);
}

VecXad OmnidirectionalDynamics::systemFlowMap(
    ocs2::ad_scalar_t time, const VecXad& state, const VecXad& input,
    const VecXad& parameters) const {
    VecXad dqdt = state.segment(dims_.q, dims_.v);
    VecXad dvdt = state.tail(dims_.v);
    VecXad dadt = input.head(dims_.u);

    VecXad dxdt(dims_.x);
    dxdt << dqdt, dvdt, dadt;
    return dxdt;
}

}  // namespace upright
