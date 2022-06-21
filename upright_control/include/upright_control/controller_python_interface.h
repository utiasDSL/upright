#pragma once

#include <ocs2_python_interface/PythonInterface.h>

#include <upright_control/controller_interface.h>
#include <upright_control/controller_settings.h>

namespace upright {

class ControllerPythonInterface final : public ocs2::PythonInterface {
   public:
    explicit ControllerPythonInterface(const ControllerSettings& settings) {
        ControllerInterface control_interface(settings);

        // Set the reference manager -- otherwise there are problems with the
        // EndEffectorCost
        std::unique_ptr<ocs2::MPC_BASE> mpcPtr = control_interface.getMpc();
        mpcPtr->getSolverPtr()->setReferenceManager(
            control_interface.getReferenceManagerPtr());
        ocs2::PythonInterface::init(control_interface, std::move(mpcPtr));
    }
};

}  // namespace upright
