#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <ocs2_core/Types.h>
#include <ocs2_python_interface/PybindMacros.h>
#include <tray_balance_constraints/inequality_constraints.h>
#include <tray_balance_constraints/robust.h>
#include <tray_balance_constraints/types.h>

#include "tray_balance_ocs2/MobileManipulatorPythonInterface.h"
#include "tray_balance_ocs2/TaskSettings.h"
#include "tray_balance_ocs2/constraint/CollisionAvoidanceConstraint.h"
#include "tray_balance_ocs2/constraint/ConstraintType.h"
#include "tray_balance_ocs2/constraint/ObstacleConstraint.h"
#include "tray_balance_ocs2/constraint/tray_balance/TrayBalanceConfigurations.h"
#include "tray_balance_ocs2/constraint/tray_balance/TrayBalanceSettings.h"

using namespace ocs2;
using namespace mobile_manipulator;

/* make vector types opaque so they are not converted to python lists */
PYBIND11_MAKE_OPAQUE(ocs2::scalar_array_t)
PYBIND11_MAKE_OPAQUE(ocs2::vector_array_t)
PYBIND11_MAKE_OPAQUE(ocs2::matrix_array_t)

using CollisionSphereVector = std::vector<CollisionSphere<scalar_t>>;
using StringPairVector = std::vector<std::pair<std::string, std::string>>;

PYBIND11_MAKE_OPAQUE(CollisionSphereVector)
PYBIND11_MAKE_OPAQUE(StringPairVector)

/* create a python module */
PYBIND11_MODULE(MobileManipulatorPythonInterface, m) {
    /* bind vector types so they can be used natively in python */
    VECTOR_TYPE_BINDING(ocs2::scalar_array_t, "scalar_array")
    VECTOR_TYPE_BINDING(ocs2::vector_array_t, "vector_array")
    VECTOR_TYPE_BINDING(ocs2::matrix_array_t, "matrix_array")

    VECTOR_TYPE_BINDING(CollisionSphereVector, "CollisionSphereVector")
    VECTOR_TYPE_BINDING(StringPairVector, "StringPairVector")

    /* bind settings */
    /// Robust balancing
    pybind11::class_<Ball<scalar_t>>(m, "Ball")
        .def(pybind11::init<const Vec3<scalar_t>&, const scalar_t>(),
             "center"_a, "radius"_a)
        .def_readwrite("center", &Ball<scalar_t>::center)
        .def_readwrite("radius", &Ball<scalar_t>::radius);

    pybind11::class_<RobustParameterSet<scalar_t>>(m, "RobustParameterSet")
        .def(pybind11::init<>())
        .def_readwrite("balls", &RobustParameterSet<scalar_t>::balls)
        .def_readwrite("min_support_dist",
                       &RobustParameterSet<scalar_t>::min_support_dist)
        .def_readwrite("min_mu", &RobustParameterSet<scalar_t>::min_mu)
        .def_readwrite("min_r_tau", &RobustParameterSet<scalar_t>::min_r_tau)
        .def_readwrite("max_radius", &RobustParameterSet<scalar_t>::max_radius);

    /// Normal balancing
    pybind11::class_<RigidBody<scalar_t>>(m, "RigidBody")
        .def(pybind11::init<const scalar_t, const Mat3<scalar_t>&,
                            const Vec3<scalar_t>&>(),
             "mass"_a, "inertia"_a, "com"_a)
        .def_readwrite("mass", &RigidBody<scalar_t>::mass)
        .def_readwrite("inertia", &RigidBody<scalar_t>::inertia)
        .def_readwrite("com", &RigidBody<scalar_t>::com);

    pybind11::class_<SupportAreaBase<scalar_t>>(m, "SupportAreaBase");

    pybind11::class_<CircleSupportArea<scalar_t>, SupportAreaBase<scalar_t>>(
        m, "CircleSupportArea")
        .def(pybind11::init<const scalar_t, const Vec2<scalar_t>&,
                            const scalar_t>(),
             "radius"_a, "offset"_a, "margin"_a)
        .def_readwrite("radius", &CircleSupportArea<scalar_t>::radius)
        .def_readwrite("offset", &CircleSupportArea<scalar_t>::offset)
        .def_readwrite("margin", &CircleSupportArea<scalar_t>::margin);

    pybind11::class_<PolygonSupportArea<scalar_t>, SupportAreaBase<scalar_t>>(
        m, "PolygonSupportArea")
        .def(pybind11::init<const std::vector<Vec2<scalar_t>>&,
                            const Vec2<scalar_t>&, const scalar_t>(),
             "vertices"_a, "offset"_a, "margin"_a=0)
        .def_readwrite("vertices", &PolygonSupportArea<scalar_t>::vertices)
        .def_readwrite("offset", &PolygonSupportArea<scalar_t>::offset)
        .def_readwrite("margin", &PolygonSupportArea<scalar_t>::margin)
        .def_static("circle", &PolygonSupportArea<scalar_t>::circle, "radius"_a,
                    "margin"_a=0)
        .def_static("equilateral_triangle",
                    &PolygonSupportArea<scalar_t>::equilateral_triangle,
                    "side_length"_a, "margin"_a=0)
        .def_static("axis_aligned_rectangle",
                    &PolygonSupportArea<scalar_t>::axis_aligned_rectangle,
                    "sx"_a, "sy"_a, "margin"_a=0);

    pybind11::class_<BalancedObject<scalar_t>>(m, "BalancedObject")
        .def(pybind11::init<const RigidBody<scalar_t>&, scalar_t,
                            const SupportAreaBase<scalar_t>&, scalar_t,
                            scalar_t>(),
             "body"_a, "com_height"_a, "support_area"_a, "r_tau"_a, "mu"_a)
        .def_static("compose", &BalancedObject<scalar_t>::compose, "objects"_a);

    pybind11::class_<TrayBalanceConfiguration>(m, "TrayBalanceConfiguration")
        .def(pybind11::init<>())
        .def_readwrite("objects", &TrayBalanceConfiguration::objects)
        .def("num_constraints", &TrayBalanceConfiguration::num_constraints);

    pybind11::class_<TrayBalanceSettings>(m, "TrayBalanceSettings")
        .def(pybind11::init<>())
        .def_readwrite("enabled", &TrayBalanceSettings::enabled)
        .def_readwrite("robust", &TrayBalanceSettings::robust)
        .def_readwrite("constraint_type", &TrayBalanceSettings::constraint_type)
        .def_readwrite("mu", &TrayBalanceSettings::mu)
        .def_readwrite("delta", &TrayBalanceSettings::delta)
        .def_readwrite("config", &TrayBalanceSettings::config)
        .def_readwrite("robust_params", &TrayBalanceSettings::robust_params);

    /// Other stuff
    pybind11::enum_<ConstraintType>(m, "ConstraintType")
        .value("Soft", ConstraintType::Soft)
        .value("Hard", ConstraintType::Hard);


    pybind11::class_<CollisionSphere<scalar_t>>(m, "CollisionSphere")
        .def(pybind11::init<const std::string&, const std::string&,
                            const Eigen::Matrix<scalar_t, 3, 1>&, const scalar_t>(),
             "name"_a, "parent_frame_name"_a, "offset"_a, "radius"_a)
        .def_readwrite("name", &CollisionSphere<scalar_t>::name)
        .def_readwrite("parent_frame_name", &CollisionSphere<scalar_t>::parent_frame_name)
        .def_readwrite("offset", &CollisionSphere<scalar_t>::offset)
        .def_readwrite("radius", &CollisionSphere<scalar_t>::radius);

    pybind11::class_<CollisionAvoidanceSettings>(m,
                                                 "CollisionAvoidanceSettings")
        .def(pybind11::init<>())
        .def_readwrite("enabled", &CollisionAvoidanceSettings::enabled)
        .def_readwrite("collision_link_pairs",
                       &CollisionAvoidanceSettings::collision_link_pairs)
        .def_readwrite("minimum_distance",
                       &CollisionAvoidanceSettings::minimum_distance)
        .def_readwrite("mu", &CollisionAvoidanceSettings::mu)
        .def_readwrite("delta", &CollisionAvoidanceSettings::delta)
        .def_readwrite("extra_spheres", &CollisionAvoidanceSettings::extra_spheres);

    pybind11::class_<DynamicObstacleSettings>(m, "DynamicObstacleSettings")
        .def(pybind11::init<>())
        .def_readwrite("enabled", &DynamicObstacleSettings::enabled)
        .def_readwrite("obstacle_radius",
                       &DynamicObstacleSettings::obstacle_radius)
        .def_readwrite("mu", &DynamicObstacleSettings::mu)
        .def_readwrite("delta", &DynamicObstacleSettings::delta)
        .def_readwrite("collision_spheres", &DynamicObstacleSettings::collision_spheres);

    pybind11::class_<TaskSettings> task_settings(m, "TaskSettings");
    task_settings.def(pybind11::init<>())
        .def_readwrite("method", &TaskSettings::method)
        .def_readwrite("dynamic_obstacle_settings",
                       &TaskSettings::dynamic_obstacle_settings)
        .def_readwrite("collision_avoidance_settings",
                       &TaskSettings::collision_avoidance_settings)
        .def_readwrite("tray_balance_settings",
                       &TaskSettings::tray_balance_settings)
        .def_readwrite("initial_state", &TaskSettings::initial_state);

    pybind11::enum_<TaskSettings::Method>(task_settings, "Method")
        .value("DDP", TaskSettings::Method::DDP)
        .value("SQP", TaskSettings::Method::SQP);

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
