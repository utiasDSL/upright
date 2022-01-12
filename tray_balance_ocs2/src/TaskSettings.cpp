#include <pybind11/pybind11.h>

#include "tray_balance_ocs2/TaskSettings.h"

using namespace ocs2;
using namespace mobile_manipulator;
namespace py = pybind11;

PYBIND11_MODULE(task_settings, m) {
    m.doc() = "Bindings for task settings.";

    py::class_<TaskSettings>(m, "TaskSettings")
        .def(py::init<>())
        .def_readwrite("tray_balance_enabled",
                       &TaskSettings::tray_balance_enabled)
        .def_readwrite("dynamic_obstacle_enabled",
                       &TaskSettings::dynamic_obstacle_enabled)
        .def_readwrite("collision_avoidance_enabled",
                       &TaskSettings::collision_avoidance_enabled);
}
