#include <upright_control/dynamics/Dimensions.h>
#include <upright_control/types.h>
#include <upright_control/util.h>

#include "upright_control/dynamics/FixedBaseDynamics.h"

namespace upright {

FixedBaseDynamics::FixedBaseDynamics(
    const std::string& modelName, const RobotDimensions& dims,
    const std::string& modelFolder /*= "/tmp/ocs2"*/,
    bool recompileLibraries /*= true*/, bool verbose /*= true*/)
    : dims_(dims), ocs2::SystemDynamicsBaseAD() {
    initialize(dims.x, dims.u, modelName, modelFolder, recompileLibraries,
               verbose);
}

VecXad FixedBaseDynamics::systemFlowMap(ocs2::ad_scalar_t time,
                                        const VecXad& state,
                                        const VecXad& input,
                                        const VecXad& parameters) const {
    VecXad dqdt = state.segment(dims_.q, dims_.v);
    VecXad dvdt = state.tail(dims_.v);
    VecXad dadt = input;

    VecXad dxdt(dims_.x);
    dxdt << dqdt, dvdt, dadt;
    return dxdt;
}

}  // namespace upright
