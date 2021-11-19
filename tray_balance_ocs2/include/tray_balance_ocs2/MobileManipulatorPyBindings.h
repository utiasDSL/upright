#pragma once

#include <tray_balance_ocs2/MobileManipulatorInterface.h>
#include <tray_balance_ocs2/definitions.h>
#include <ocs2_python_interface/PythonInterface.h>

namespace ocs2 {
namespace mobile_manipulator {

class MobileManipulatorPyBindings final : public PythonInterface {
 public:
  explicit MobileManipulatorPyBindings(const std::string& taskFileFolder) {
    MobileManipulatorInterface robot(taskFileFolder);

    // Set the reference manager -- otherwise there are problems with the
    // EndEffectorCost
    std::unique_ptr<MPC_BASE> mpcPtr = robot.getMpc();
    mpcPtr->getSolverPtr()->setReferenceManager(robot.getReferenceManagerPtr());
    PythonInterface::init(robot, std::move(mpcPtr));
  }
};

}  // namespace mobile_manipulator
}  // namespace ocs2
