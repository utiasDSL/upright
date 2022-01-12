#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <ocs2_core/Types.h>
#include <ocs2_python_interface/PybindMacros.h>

#include "tray_balance_ocs2/MobileManipulatorPyBindings.h"  // TODO rename
#include "tray_balance_ocs2/TaskSettings.h"

// CREATE_ROBOT_PYTHON_BINDINGS(
//     ocs2::mobile_manipulator::MobileManipulatorPyBindings,
//     MobileManipulatorPyBindings)

using MobileManipulatorPythonInterface =
    ocs2::mobile_manipulator::MobileManipulatorPythonInterface;
using TaskSettings = ocs2::mobile_manipulator::TaskSettings;

/* make vector types opaque so they are not converted to python lists */
PYBIND11_MAKE_OPAQUE(ocs2::scalar_array_t)
PYBIND11_MAKE_OPAQUE(ocs2::vector_array_t)
PYBIND11_MAKE_OPAQUE(ocs2::matrix_array_t)

/* create a python module */
PYBIND11_MODULE(MobileManipulatorPythonInterface, m) {
    /* bind vector types so they can be used natively in python */
    VECTOR_TYPE_BINDING(ocs2::scalar_array_t, "scalar_array")
    VECTOR_TYPE_BINDING(ocs2::vector_array_t, "vector_array")
    VECTOR_TYPE_BINDING(ocs2::matrix_array_t, "matrix_array")

    /* bind settings */
    pybind11::class_<TaskSettings>(m, "TaskSettings")
        .def(pybind11::init<>())
        .def_readwrite("tray_balance_enabled",
                       &TaskSettings::tray_balance_enabled)
        .def_readwrite("dynamic_obstacle_enabled",
                       &TaskSettings::dynamic_obstacle_enabled)
        .def_readwrite("collision_avoidance_enabled",
                       &TaskSettings::collision_avoidance_enabled);

    /* bind approximation classes */
    pybind11::class_<ocs2::VectorFunctionLinearApproximation>(
        m, "VectorFunctionLinearApproximation")
        .def_readwrite("f", &ocs2::VectorFunctionLinearApproximation::f)
        .def_readwrite("dfdx", &ocs2::VectorFunctionLinearApproximation::dfdx)
        .def_readwrite("dfdu", &ocs2::VectorFunctionLinearApproximation::dfdu);

    pybind11::class_<ocs2::VectorFunctionQuadraticApproximation>(
        m, "VectorFunctionQuadraticApproximation")
        .def_readwrite("f", &ocs2::VectorFunctionQuadraticApproximation::f)
        .def_readwrite("dfdx",
                       &ocs2::VectorFunctionQuadraticApproximation::dfdx)
        .def_readwrite("dfdu",
                       &ocs2::VectorFunctionQuadraticApproximation::dfdu)
        .def_readwrite("dfdxx",
                       &ocs2::VectorFunctionQuadraticApproximation::dfdxx)
        .def_readwrite("dfdux",
                       &ocs2::VectorFunctionQuadraticApproximation::dfdux)
        .def_readwrite("dfduu",
                       &ocs2::VectorFunctionQuadraticApproximation::dfduu);

    pybind11::class_<ocs2::ScalarFunctionQuadraticApproximation>(
        m, "ScalarFunctionQuadraticApproximation")
        .def_readwrite("f", &ocs2::ScalarFunctionQuadraticApproximation::f)
        .def_readwrite("dfdx",
                       &ocs2::ScalarFunctionQuadraticApproximation::dfdx)
        .def_readwrite("dfdu",
                       &ocs2::ScalarFunctionQuadraticApproximation::dfdu)
        .def_readwrite("dfdxx",
                       &ocs2::ScalarFunctionQuadraticApproximation::dfdxx)
        .def_readwrite("dfdux",
                       &ocs2::ScalarFunctionQuadraticApproximation::dfdux)
        .def_readwrite("dfduu",
                       &ocs2::ScalarFunctionQuadraticApproximation::dfduu);

    /* bind TargetTrajectories class */
    pybind11::class_<ocs2::TargetTrajectories>(m, "TargetTrajectories")
        .def(pybind11::init<ocs2::scalar_array_t, ocs2::vector_array_t,
                            ocs2::vector_array_t>());

    /* bind the actual mpc interface */
    pybind11::class_<MobileManipulatorPythonInterface>(m, "mpc_interface")
        .def(pybind11::init<const std::string&, const std::string&,
                            const TaskSettings&>(),
             "taskFile"_a, "libFolder"_a, "settings"_a)
        .def("getStateDim", &MobileManipulatorPythonInterface::getStateDim)
        .def("getInputDim", &MobileManipulatorPythonInterface::getInputDim)
        .def("setObservation",
             &MobileManipulatorPythonInterface::setObservation, "t"_a,
             "x"_a.noconvert(), "u"_a.noconvert())
        .def("setTargetTrajectories",
             &MobileManipulatorPythonInterface::setTargetTrajectories,
             "targetTrajectories"_a)
        .def("reset", &MobileManipulatorPythonInterface::reset,
             "targetTrajectories"_a)
        .def("advanceMpc", &MobileManipulatorPythonInterface::advanceMpc)
        .def("getMpcSolution",
             &MobileManipulatorPythonInterface::getMpcSolution,
             "t"_a.noconvert(), "x"_a.noconvert(), "u"_a.noconvert())
        .def("evaluateMpcSolution",
             &MobileManipulatorPythonInterface::evaluateMpcSolution,
             "current_time"_a.noconvert(), "current_state"_a.noconvert(),
             "opt_state"_a.noconvert(), "opt_input"_a.noconvert())
        .def("getLinearFeedbackGain",
             &MobileManipulatorPythonInterface::getLinearFeedbackGain,
             "t"_a.noconvert())
        .def("flowMap", &MobileManipulatorPythonInterface::flowMap, "t"_a,
             "x"_a.noconvert(), "u"_a.noconvert())
        .def("flowMapLinearApproximation",
             &MobileManipulatorPythonInterface::flowMapLinearApproximation,
             "t"_a, "x"_a.noconvert(), "u"_a.noconvert())
        .def("cost", &MobileManipulatorPythonInterface::cost, "t"_a,
             "x"_a.noconvert(), "u"_a.noconvert())
        .def("costQuadraticApproximation",
             &MobileManipulatorPythonInterface::costQuadraticApproximation,
             "t"_a, "x"_a.noconvert(), "u"_a.noconvert())
        .def("valueFunction", &MobileManipulatorPythonInterface::valueFunction,
             "t"_a, "x"_a.noconvert())
        .def("valueFunctionStateDerivative",
             &MobileManipulatorPythonInterface::valueFunctionStateDerivative,
             "t"_a, "x"_a.noconvert())
        .def("stateInputEqualityConstraint",
             &MobileManipulatorPythonInterface::stateInputEqualityConstraint,
             "t"_a, "x"_a.noconvert(), "u"_a.noconvert())
        .def("stateInputEqualityConstraintLinearApproximation",
             &MobileManipulatorPythonInterface::
                 stateInputEqualityConstraintLinearApproximation,
             "t"_a, "x"_a.noconvert(), "u"_a.noconvert())
        .def("stateInputEqualityConstraintLagrangian",
             &MobileManipulatorPythonInterface::
                 stateInputEqualityConstraintLagrangian,
             "t"_a, "x"_a.noconvert(), "u"_a.noconvert())
        .def("stateInputInequalityConstraint",
             &MobileManipulatorPythonInterface::stateInputInequalityConstraint,
             "name"_a, "t"_a, "x"_a.noconvert(), "u"_a.noconvert())
        .def("softStateInputInequalityConstraint",
             &MobileManipulatorPythonInterface::
                 softStateInputInequalityConstraint,
             "name"_a, "t"_a, "x"_a.noconvert(), "u"_a.noconvert())
        .def("stateInequalityConstraint",
             &MobileManipulatorPythonInterface::stateInequalityConstraint,
             "name"_a, "t"_a, "x"_a.noconvert())
        .def("visualizeTrajectory",
             &MobileManipulatorPythonInterface::visualizeTrajectory,
             "t"_a.noconvert(), "x"_a.noconvert(), "u"_a.noconvert(),
             "speed"_a);
}
