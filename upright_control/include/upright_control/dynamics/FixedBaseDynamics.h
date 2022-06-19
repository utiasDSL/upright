#pragma once

#include <ocs2_core/dynamics/SystemDynamicsBaseAD.h>

#include <upright_control/dynamics/Dimensions.h>
#include <upright_control/types.h>

namespace upright {

class FixedBaseDynamics final : public ocs2::SystemDynamicsBaseAD {
   public:
    explicit FixedBaseDynamics(const std::string& modelName,
                               const RobotDimensions& dims,
                               const std::string& modelFolder = "/tmp/ocs2",
                               bool recompileLibraries = true,
                               bool verbose = true);

    ~FixedBaseDynamics() override = default;

    FixedBaseDynamics* clone() const override {
        return new FixedBaseDynamics(*this);
    }

    VecXad systemFlowMap(ocs2::ad_scalar_t time, const VecXad& state,
                         const VecXad& input,
                         const VecXad& parameters) const override;

   private:
    FixedBaseDynamics(const FixedBaseDynamics& rhs) = default;

    RobotDimensions dims_;
};

}  // namespace upright
