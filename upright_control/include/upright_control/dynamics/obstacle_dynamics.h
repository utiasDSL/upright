#pragma once

#include <ocs2_core/dynamics/SystemDynamicsBaseAD.h>

#include <upright_control/dimensions.h>
#include <upright_control/types.h>

namespace upright {

class ObstacleDynamics final : public ocs2::SystemDynamicsBaseAD {
   public:
    explicit ObstacleDynamics(const std::string& modelName,
                              const std::string& modelFolder = "/tmp/ocs2",
                              bool recompileLibraries = true,
                              bool verbose = true)
        : ocs2::SystemDynamicsBaseAD() {
        initialize(9, 0, modelName, modelFolder, recompileLibraries, verbose);
    }

    ~ObstacleDynamics() override = default;

    ObstacleDynamics* clone() const override {
        return new ObstacleDynamics(*this);
    }

    VecXad systemFlowMap(ocs2::ad_scalar_t time, const VecXad& state,
                         const VecXad& input,
                         const VecXad& parameters) const override {
        VecXad dqdt = state.segment(3, 3);
        VecXad dvdt = state.tail(3);
        VecXad dadt = VecXad::Zero(3);  // constant acceleration

        VecXad dxdt(9);
        dxdt << dqdt, dvdt, dadt;
        return dxdt;
    }

   private:
    ObstacleDynamics(const ObstacleDynamics& rhs) = default;
};

}  // namespace upright
