#pragma once

#include <ocs2_core/dynamics/SystemDynamicsBaseAD.h>

#include <upright_control/dynamics/dimensions.h>
#include <upright_control/types.h>

namespace upright {

class OmnidirectionalDynamics final : public ocs2::SystemDynamicsBaseAD {
   public:
    explicit OmnidirectionalDynamics(
        const std::string& modelName, const RobotDimensions& dims,
        const std::string& modelFolder = "/tmp/ocs2",
        bool recompileLibraries = true, bool verbose = true);

    ~OmnidirectionalDynamics() override = default;

    OmnidirectionalDynamics* clone() const override {
        return new OmnidirectionalDynamics(*this);
    }

    VecXad systemFlowMap(ocs2::ad_scalar_t time, const VecXad& state,
                         const VecXad& input,
                         const VecXad& parameters) const override;

   private:
    OmnidirectionalDynamics(const OmnidirectionalDynamics& rhs) = default;

    RobotDimensions dims_;
};

}  // namespace upright
