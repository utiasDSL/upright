#include <upright_control/dimensions.h>
#include <upright_control/dynamics/util.h>
#include <upright_control/types.h>

#include "upright_control/dynamics/nonholonomic_dynamics.h"

namespace upright {

NonholonomicDynamics::NonholonomicDynamics(
    const std::string& modelName, const OptimizationDimensions& dims,
    const std::string& modelFolder /*= "/tmp/ocs2"*/,
    bool recompileLibraries /*= true*/, bool verbose /*= true*/)
    : dims_(dims), ocs2::SystemDynamicsBaseAD() {
    initialize(dims.x(), dims.u(), modelName, modelFolder, recompileLibraries,
               verbose);
}

VecXad NonholonomicDynamics::systemFlowMap(
    ocs2::ad_scalar_t time, const VecXad& state, const VecXad& input,
    const VecXad& parameters) const {
    const RobotDimensions& r = dims_.robot;

    ocs2::ad_scalar_t yaw = state(2);
    VecXad v = state.segment(r.q, r.v);

    VecXad dqdt(r.q);
    dqdt << cos(yaw) * v(0), sin(yaw) * v(0), v.tail(r.v - 1);

    VecXad dvdt = state.tail(r.v);
    VecXad dadt = input.head(r.u);

    VecXad dxdt(r.x);
    dxdt << dqdt, dvdt, dadt;
    return dxdt;
}

}  // namespace upright
