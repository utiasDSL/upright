#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <ocs2_core/Types.h>
#include <ocs2_python_interface/PybindMacros.h>
#include <tray_balance_constraints/dynamics.h>
#include <tray_balance_constraints/nominal.h>
#include <tray_balance_constraints/types.h>

#include "tray_balance_ocs2/MobileManipulatorPythonInterface.h"
#include "tray_balance_ocs2/TaskSettings.h"
#include "tray_balance_ocs2/constraint/CollisionAvoidanceConstraint.h"
#include "tray_balance_ocs2/constraint/ConstraintType.h"
#include "tray_balance_ocs2/constraint/ObstacleConstraint.h"
#include "tray_balance_ocs2/constraint/balancing/BalancingSettings.h"

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
    // TODO move this stuff to the tray_balance_constraints package

    /// Normal balancing
    pybind11::class_<RigidBody<scalar_t>>(m, "RigidBody")
        .def(pybind11::init<const scalar_t, const Mat3<scalar_t>&,
                            const Vec3<scalar_t>&>(),
             "mass"_a, "inertia"_a, "com"_a)
        .def_readwrite("mass", &RigidBody<scalar_t>::mass)
        .def_readwrite("inertia", &RigidBody<scalar_t>::inertia)
        .def_readwrite("com", &RigidBody<scalar_t>::com);

    pybind11::class_<BalancedObject<scalar_t>>(m, "BalancedObject")
        .def(pybind11::init<const RigidBody<scalar_t>&, scalar_t,
                            const SupportAreaBase<scalar_t>&, scalar_t,
                            scalar_t>(),
             "body"_a, "com_height"_a, "support_area"_a, "r_tau"_a, "mu"_a)
        .def_static("compose", &BalancedObject<scalar_t>::compose, "objects"_a);

    pybind11::class_<BalanceConstraintsEnabled>(m, "BalanceConstraintsEnabled")
        .def(pybind11::init<>())
        .def_readwrite("normal", &BalanceConstraintsEnabled::normal)
        .def_readwrite("friction", &BalanceConstraintsEnabled::friction)
        .def_readwrite("zmp", &BalanceConstraintsEnabled::zmp);

    pybind11::class_<TrayBalanceConfiguration<scalar_t>>(m, "TrayBalanceConfiguration")
        .def(pybind11::init<>())
        .def_readwrite("objects", &TrayBalanceConfiguration<scalar_t>::objects)
        .def_readwrite("enabled", &TrayBalanceConfiguration<scalar_t>::enabled)
        .def("num_constraints", &TrayBalanceConfiguration<scalar_t>::num_constraints);

    pybind11::class_<TrayBalanceSettings>(m, "TrayBalanceSettings")
        .def(pybind11::init<>())
        .def_readwrite("enabled", &TrayBalanceSettings::enabled)
        .def_readwrite("bounded", &TrayBalanceSettings::bounded)
        .def_readwrite("constraint_type", &TrayBalanceSettings::constraint_type)
        .def_readwrite("mu", &TrayBalanceSettings::mu)
        .def_readwrite("delta", &TrayBalanceSettings::delta)
        .def_readwrite("nominal_config", &TrayBalanceSettings::nominal_config)
        .def_readwrite("bounded_config", &TrayBalanceSettings::bounded_config);

    /// Other stuff
    pybind11::enum_<ConstraintType>(m, "ConstraintType")
        .value("Soft", ConstraintType::Soft)
        .value("Hard", ConstraintType::Hard);

    pybind11::class_<CollisionSphere<scalar_t>>(m, "CollisionSphere")
        .def(pybind11::init<const std::string&, const std::string&,
                            const Eigen::Matrix<scalar_t, 3, 1>&,
                            const scalar_t>(),
             "name"_a, "parent_frame_name"_a, "offset"_a, "radius"_a)
        .def_readwrite("name", &CollisionSphere<scalar_t>::name)
        .def_readwrite("parent_frame_name",
                       &CollisionSphere<scalar_t>::parent_frame_name)
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
        .def_readwrite("extra_spheres",
                       &CollisionAvoidanceSettings::extra_spheres);

    pybind11::class_<DynamicObstacleSettings>(m, "DynamicObstacleSettings")
        .def(pybind11::init<>())
        .def_readwrite("enabled", &DynamicObstacleSettings::enabled)
        .def_readwrite("obstacle_radius",
                       &DynamicObstacleSettings::obstacle_radius)
        .def_readwrite("mu", &DynamicObstacleSettings::mu)
        .def_readwrite("delta", &DynamicObstacleSettings::delta)
        .def_readwrite("collision_spheres",
                       &DynamicObstacleSettings::collision_spheres);

    pybind11::class_<ControllerSettings> ctrl_settings(m, "ControllerSettings");
    ctrl_settings.def(pybind11::init<>())
        .def_readwrite("method", &ControllerSettings::method)
        .def_readwrite("dynamic_obstacle_settings",
                       &ControllerSettings::dynamic_obstacle_settings)
        .def_readwrite("collision_avoidance_settings",
                       &ControllerSettings::collision_avoidance_settings)
        .def_readwrite("tray_balance_settings",
                       &ControllerSettings::tray_balance_settings)
        .def_readwrite("initial_state", &ControllerSettings::initial_state)
        .def_readwrite("input_weight", &ControllerSettings::input_weight)
        .def_readwrite("state_weight", &ControllerSettings::state_weight)
        .def_readwrite("end_effector_weight", &ControllerSettings::end_effector_weight)
        .def_readwrite("input_limit_lower", &ControllerSettings::input_limit_lower)
        .def_readwrite("input_limit_upper", &ControllerSettings::input_limit_upper)
        .def_readwrite("state_limit_lower", &ControllerSettings::state_limit_lower)
        .def_readwrite("state_limit_upper", &ControllerSettings::state_limit_upper)
        .def_readwrite("robot_urdf_path", &ControllerSettings::robot_urdf_path)
        .def_readwrite("obstacle_urdf_path", &ControllerSettings::obstacle_urdf_path)
        .def_readwrite("ocs2_config_path", &ControllerSettings::ocs2_config_path);

    pybind11::enum_<ControllerSettings::Method>(ctrl_settings, "Method")
        .value("DDP", ControllerSettings::Method::DDP)
        .value("SQP", ControllerSettings::Method::SQP);

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
                            const ControllerSettings&>(),
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
