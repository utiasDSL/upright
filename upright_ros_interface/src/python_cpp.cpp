// Test node for loading stuff into C++ via Python
#include <pybind11/pybind11.h>
#include <pybind11/embed.h>
#include <pybind11/eigen.h>

#include <upright_core/types.h>
#include <upright_control/controller_settings.h>

namespace py = pybind11;

int main() {
    // Note pybind11_catkin uses an older version of pybind, so we use
    // py::module rather than py::module_
    py::scoped_interpreter guard{};
    py::object upright_control = py::module::import("upright_control");
    py::object PyTargetTrajectories = upright_control.attr("wrappers").attr("TargetTrajectories");

    std::string path = "/home/adam/phd/code/projects/tray_balance/catkin_ws/src/tray_balance/upright_cmd/config/ur10/ur10_demo.yaml";
    upright::VecXd x0 = upright::VecXd::Zero(18);
    py::object pyref = PyTargetTrajectories.attr("from_config_file")(path, x0);
    ocs2::TargetTrajectories ref = pyref.cast<ocs2::TargetTrajectories>();

    // std::cout << settings.gravity.transpose() << std::endl;
}
