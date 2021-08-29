#pragma once

#include <ocs2_mobile_manipulator_modified/MobileManipulatorInterface.h>
#include <ocs2_mobile_manipulator_modified/definitions.h>
#include <ocs2_python_interface/PythonInterface.h>

namespace ocs2 {
namespace mobile_manipulator {

class MobileManipulatorPyBindings final : public PythonInterface {
 public:
  explicit MobileManipulatorPyBindings(const std::string& taskFileFolder) {
    MobileManipulatorInterface robot(taskFileFolder);
    PythonInterface::init(robot, robot.getMpc());
  }
};

}  // namespace mobile_manipulator
}  // namespace ocs2
