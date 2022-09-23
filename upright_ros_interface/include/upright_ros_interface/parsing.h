#pragma once

#include <Eigen/Eigen>
#include <vector>

#include <ros/ros.h>

#include <pybind11/eigen.h>
#include <pybind11/embed.h>
#include <pybind11/pybind11.h>

#include <upright_control/controller_settings.h>
#include <upright_control/types.h>

namespace py = pybind11;

namespace upright {

// TODO: ideally, we could just parse everything on C++, instead of doing it in
// Python and casting the object back like this
ControllerSettings parse_control_settings(const std::string& config_path) {
    // Note pybind11_catkin uses an older version of pybind, so we use
    // py::module rather than py::module_
    py::object upright_control = py::module::import("upright_control");
    py::object PyControllerSettings =
        upright_control.attr("wrappers").attr("ControllerSettings");
    return PyControllerSettings.attr("from_config_file")(config_path)
        .cast<ControllerSettings>();
}

ocs2::TargetTrajectories parse_target_trajectory(const std::string& config_path,
                                                 const VecXd& x0) {
    py::object upright_control = py::module::import("upright_control");
    py::object PyTargetTrajectories =
        upright_control.attr("wrappers").attr("TargetTrajectories");
    return PyTargetTrajectories.attr("from_config_file")(config_path, x0)
        .cast<ocs2::TargetTrajectories>();
}

}  // namespace upright
