#pragma once

#include <ocs2_python_interface/PythonInterface.h>
#include <tray_balance_ocs2/MobileManipulatorInterface.h>
#include <tray_balance_ocs2/ControllerSettings.h>

namespace upright {

class MobileManipulatorPythonInterface final : public ocs2::PythonInterface {
   public:
    explicit MobileManipulatorPythonInterface(const ControllerSettings& settings) {
        MobileManipulatorInterface control_interface(settings);

        // Set the reference manager -- otherwise there are problems with the
        // EndEffectorCost
        std::unique_ptr<ocs2::MPC_BASE> mpcPtr = control_interface.getMpc();
        mpcPtr->getSolverPtr()->setReferenceManager(
            control_interface.getReferenceManagerPtr());
        ocs2::PythonInterface::init(control_interface, std::move(mpcPtr));
    }
};

}  // namespace upright
