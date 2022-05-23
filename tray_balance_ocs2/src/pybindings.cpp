#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

#include <ocs2_core/Types.h>
#include <ocs2_core/control/LinearController.h>
#include <ocs2_python_interface/PybindMacros.h>
#include <tray_balance_constraints/dynamics.h>
#include <tray_balance_constraints/nominal.h>
#include <tray_balance_constraints/types.h>

#include "tray_balance_ocs2/ControllerSettings.h"
#include "tray_balance_ocs2/MobileManipulatorPythonInterface.h"
#include "tray_balance_ocs2/constraint/BoundedBalancingConstraints.h"
#include "tray_balance_ocs2/constraint/CollisionAvoidanceConstraint.h"
#include "tray_balance_ocs2/constraint/ConstraintType.h"
#include "tray_balance_ocs2/constraint/ObstacleConstraint.h"
#include "tray_balance_ocs2/dynamics/BaseType.h"
#include "tray_balance_ocs2/dynamics/Dimensions.h"
#include "tray_balance_ocs2/dynamics/FixedBasePinocchioMapping.h"
#include "tray_balance_ocs2/dynamics/MobileManipulatorPinocchioMapping.h"

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
PYBIND11_MODULE(bindings, m) {
    /* bind vector types so they can be used natively in python */
    VECTOR_TYPE_BINDING(ocs2::scalar_array_t, "scalar_array")
    VECTOR_TYPE_BINDING(ocs2::vector_array_t, "vector_array")
    VECTOR_TYPE_BINDING(ocs2::matrix_array_t, "matrix_array")

    VECTOR_TYPE_BINDING(CollisionSphereVector, "CollisionSphereVector")
    VECTOR_TYPE_BINDING(StringPairVector, "StringPairVector")

    pybind11::class_<FixedBasePinocchioMapping<scalar_t>>(
        m, "FixedBasePinocchioMapping")
        .def(pybind11::init<const RobotDimensions &>(), "dims")
        .def("get_pinocchio_joint_position",
             &FixedBasePinocchioMapping<scalar_t>::getPinocchioJointPosition,
             "state"_a)
        .def("get_pinocchio_joint_velocity",
             &FixedBasePinocchioMapping<scalar_t>::getPinocchioJointVelocity,
             "state"_a, "input"_a)
        .def(
            "get_pinocchio_joint_acceleration",
            &FixedBasePinocchioMapping<scalar_t>::getPinocchioJointAcceleration,
            "state"_a, "input"_a);

    pybind11::class_<MobileManipulatorPinocchioMapping<scalar_t>>(
        m, "OmnidirectionalPinocchioMapping")
        .def(pybind11::init<const RobotDimensions &>(), "dims")
        .def("get_pinocchio_joint_position",
             &MobileManipulatorPinocchioMapping<
                 scalar_t>::getPinocchioJointPosition,
             "state"_a)
        .def("get_pinocchio_joint_velocity",
             &MobileManipulatorPinocchioMapping<
                 scalar_t>::getPinocchioJointVelocity,
             "state"_a, "input"_a)
        .def("get_pinocchio_joint_acceleration",
             &MobileManipulatorPinocchioMapping<
                 scalar_t>::getPinocchioJointAcceleration,
             "state"_a, "input"_a);

    pybind11::class_<TrayBalanceSettings>(m, "TrayBalanceSettings")
        .def(pybind11::init<>())
        .def_readwrite("enabled", &TrayBalanceSettings::enabled)
        .def_readwrite("constraints_enabled", &TrayBalanceSettings::constraints_enabled)
        .def_readwrite("objects", &TrayBalanceSettings::objects)
        .def_readwrite("constraint_type", &TrayBalanceSettings::constraint_type)
        .def_readwrite("mu", &TrayBalanceSettings::mu)
        .def_readwrite("delta", &TrayBalanceSettings::delta);

    /// Other stuff
    pybind11::enum_<ConstraintType>(m, "ConstraintType")
        .value("Soft", ConstraintType::Soft)
        .value("Hard", ConstraintType::Hard);

    pybind11::class_<CollisionSphere<scalar_t>>(m, "CollisionSphere")
        .def(pybind11::init<const std::string &, const std::string &,
                            const Eigen::Matrix<scalar_t, 3, 1> &,
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

    pybind11::class_<RobotDimensions>(m, "RobotDimensions")
        .def(pybind11::init<>())
        .def_readwrite("q", &RobotDimensions::q)
        .def_readwrite("v", &RobotDimensions::v)
        .def_readwrite("x", &RobotDimensions::x)
        .def_readwrite("u", &RobotDimensions::u);

    pybind11::enum_<RobotBaseType>(m, "RobotBaseType")
        .value("Fixed", RobotBaseType::Fixed)
        .value("Nonholonomic", RobotBaseType::Nonholonomic)
        .value("Omnidirectional", RobotBaseType::Omnidirectional)
        .value("Floating", RobotBaseType::Floating);

    m.def("robot_base_type_from_string", &robot_base_type_from_string);
    m.def("robot_base_type_to_string", &robot_base_type_to_string);

    pybind11::class_<InertialAlignmentSettings>(m, "InertialAlignmentSettings")
        .def(pybind11::init<>())
        .def_readwrite("enabled", &InertialAlignmentSettings::enabled)
        .def_readwrite("use_angular_acceleration",
                       &InertialAlignmentSettings::use_angular_acceleration)
        .def_readwrite("weight", &InertialAlignmentSettings::weight)
        .def_readwrite("r_oe_e", &InertialAlignmentSettings::r_oe_e);

    pybind11::class_<ControllerSettings> ctrl_settings(m, "ControllerSettings");
    ctrl_settings.def(pybind11::init<>())
        .def_readwrite("gravity", &ControllerSettings::gravity)
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
        .def_readwrite("end_effector_weight",
                       &ControllerSettings::end_effector_weight)
        .def_readwrite("input_limit_lower",
                       &ControllerSettings::input_limit_lower)
        .def_readwrite("input_limit_upper",
                       &ControllerSettings::input_limit_upper)
        .def_readwrite("input_limit_mu",
                       &ControllerSettings::input_limit_mu)
        .def_readwrite("input_limit_delta",
                       &ControllerSettings::input_limit_delta)
        .def_readwrite("state_limit_lower",
                       &ControllerSettings::state_limit_lower)
        .def_readwrite("state_limit_upper",
                       &ControllerSettings::state_limit_upper)
        .def_readwrite("state_limit_mu",
                       &ControllerSettings::state_limit_mu)
        .def_readwrite("state_limit_delta",
                       &ControllerSettings::state_limit_delta)
        .def_readwrite("robot_urdf_path", &ControllerSettings::robot_urdf_path)
        .def_readwrite("obstacle_urdf_path",
                       &ControllerSettings::obstacle_urdf_path)
        .def_readwrite("ocs2_config_path",
                       &ControllerSettings::ocs2_config_path)
        .def_readwrite("lib_folder", &ControllerSettings::lib_folder)
        .def_readwrite("robot_base_type", &ControllerSettings::robot_base_type)
        .def_readwrite("dims", &ControllerSettings::dims)
        .def_readwrite("end_effector_link_name",
                       &ControllerSettings::end_effector_link_name)
        .def_readwrite("use_operating_points",
                       &ControllerSettings::use_operating_points)
        .def_readwrite("operating_times", &ControllerSettings::operating_times)
        .def_readwrite("operating_states",
                       &ControllerSettings::operating_states)
        .def_readwrite("operating_inputs",
                       &ControllerSettings::operating_inputs)
        .def_readwrite("inertial_alignment_settings",
                       &ControllerSettings::inertial_alignment_settings);

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

    // pybind11::class_<ocs2::OperatingPoints>(m, "OperatingPoints")
    //     .def(pybind11::init<ocs2::scalar_array_t, ocs2::vector_array_t,
    //                         ocs2::vector_array_t>());

    pybind11::class_<ocs2::LinearController>(m, "LinearController")
        .def(pybind11::init<ocs2::scalar_array_t, ocs2::vector_array_t,
                            ocs2::matrix_array_t>(),
             "times"_a, "biases"_a, "gains"_a)
        .def("computeInput", &ocs2::LinearController::computeInput, "t"_a,
             "x"_a)
        .def(pybind11::pickle(
            [](const ocs2::LinearController &c) {
                /* Return a tuple that fully encodes the state of the object */
                return pybind11::make_tuple(c.timeStamp_, c.biasArray_,
                                            c.gainArray_);
            },
            [](pybind11::tuple t) {
                if (t.size() != 3) {
                    throw std::runtime_error("Invalid state!");
                }

                ocs2::LinearController c(t[0].cast<ocs2::scalar_array_t>(),
                                         t[1].cast<ocs2::vector_array_t>(),
                                         t[2].cast<ocs2::matrix_array_t>());
                return c;
            }));

    /* bind the actual mpc interface */
    pybind11::class_<MobileManipulatorPythonInterface>(m, "ControllerInterface")
        .def(pybind11::init<const ControllerSettings &>(), "settings"_a)
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
        .def("getBias", &MobileManipulatorPythonInterface::getBias,
             "t"_a.noconvert())
        .def("getLinearController",
             &MobileManipulatorPythonInterface::getLinearController)
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
