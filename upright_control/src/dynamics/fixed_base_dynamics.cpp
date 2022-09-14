#include <upright_control/dimensions.h>
#include <upright_control/types.h>

#include "upright_control/dynamics/fixed_base_dynamics.h"

namespace upright {

FixedBaseDynamics::FixedBaseDynamics(
    const std::string& modelName, const OptimizationDimensions& dims,
    const std::string& modelFolder /*= "/tmp/ocs2"*/,
    bool recompileLibraries /*= true*/, bool verbose /*= true*/)
    : dims_(dims), ocs2::SystemDynamicsBaseAD() {
    initialize(dims.x(), dims.u(), modelName, modelFolder, recompileLibraries,
               verbose);
}

VecXad FixedBaseDynamics::systemFlowMap(ocs2::ad_scalar_t time,
                                        const VecXad& state,
                                        const VecXad& input,
                                        const VecXad& parameters) const {
    const RobotDimensions& r = dims_.robot(0);
    VecXad dqdt = state.segment(r.q, r.v);
    VecXad dvdt = state.tail(r.v);
    VecXad dadt = input.head(r.u);

    VecXad dxdt(r.x);
    dxdt << dqdt, dvdt, dadt;
    return dxdt;
}

}  // namespace upright
