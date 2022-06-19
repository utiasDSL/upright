#pragma once

#include <ocs2_core/dynamics/SystemDynamicsBaseAD.h>

#include <tray_balance_ocs2/dynamics/Dimensions.h>
#include <tray_balance_ocs2/types.h>

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
