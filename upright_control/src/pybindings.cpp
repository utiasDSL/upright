#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

#include <ocs2_core/Types.h>
#include <ocs2_core/control/FeedforwardController.h>
#include <ocs2_core/control/LinearController.h>
#include <ocs2_python_interface/PybindMacros.h>

#include <upright_control/balancing_constraint_wrapper.h>
#include <upright_control/constraint/bounded_balancing_constraints.h>
#include <upright_control/constraint/constraint_type.h>
#include <upright_control/constraint/obstacle_constraint.h>
#include <upright_control/controller_python_interface.h>
#include <upright_control/controller_settings.h>
#include <upright_control/dimensions.h>
#include <upright_control/dynamics/base_type.h>
#include <upright_control/dynamics/system_pinocchio_mapping.h>
#include <upright_control/inertial_alignment.h>

using namespace upright;
using namespace ocs2;  // TODO perhaps avoid using

/* make vector types opaque so they are not converted to python lists */
PYBIND11_MAKE_OPAQUE(ocs2::scalar_array_t)
PYBIND11_MAKE_OPAQUE(ocs2::vector_array_t)
PYBIND11_MAKE_OPAQUE(ocs2::matrix_array_t)

using CollisionSphereVector = std::vector<CollisionSphere<scalar_t>>;
using StringPairVector = std::vector<std::pair<std::string, std::string>>;

using SystemMapping =
    SystemPinocchioMapping<TripleIntegratorPinocchioMapping<ocs2::scalar_t>,
                           ocs2::scalar_t>;

PYBIND11_MAKE_OPAQUE(CollisionSphereVector)
PYBIND11_MAKE_OPAQUE(StringPairVector)
PYBIND11_MAKE_OPAQUE(std::vector<DynamicObstacle>)

/* create a python module */
PYBIND11_MODULE(bindings, m) {
    /* bind vector types so they can be used natively in python */
    VECTOR_TYPE_BINDING(ocs2::scalar_array_t, "scalar_array")
    VECTOR_TYPE_BINDING(ocs2::vector_array_t, "vector_array")
    VECTOR_TYPE_BINDING(ocs2::matrix_array_t, "matrix_array")

    VECTOR_TYPE_BINDING(CollisionSphereVector, "CollisionSphereVector")
    VECTOR_TYPE_BINDING(StringPairVector, "StringPairVector")
    VECTOR_TYPE_BINDING(std::vector<DynamicObstacle>, "DynamicObstacleVector")

    pybind11::class_<SystemMapping>(m, "SystemPinocchioMapping")
        .def(pybind11::init<const OptimizationDimensions &>(), "dims")
        .def("get_pinocchio_joint_position",
             &SystemMapping::getPinocchioJointPosition, "state"_a)
        .def("get_pinocchio_joint_velocity",
             &SystemMapping::getPinocchioJointVelocity, "state"_a, "input"_a)
        .def("get_pinocchio_joint_acceleration",
             &SystemMapping::getPinocchioJointAcceleration, "state"_a,
             "input"_a);

    pybind11::class_<BalancingSettings>(m, "BalancingSettings")
        .def(pybind11::init<>())
        .def_readwrite("enabled", &BalancingSettings::enabled)
        .def_readwrite("use_force_constraints",
                       &BalancingSettings::use_force_constraints)
        .def_readwrite("constraints_enabled",
                       &BalancingSettings::constraints_enabled)
        .def_readwrite("objects", &BalancingSettings::objects)
        .def_readwrite("contacts", &BalancingSettings::contacts)
        .def_readwrite("force_weight", &BalancingSettings::force_weight)
        .def_readwrite("constraint_type", &BalancingSettings::constraint_type)
        .def_readwrite("mu", &BalancingSettings::mu)
        .def_readwrite("delta", &BalancingSettings::delta);

    /// Other stuff
    pybind11::enum_<ConstraintType>(m, "ConstraintType")
        .value("Soft", ConstraintType::Soft)
        .value("Hard", ConstraintType::Hard);
    m.def("constraint_type_from_string", &constraint_type_from_string);
    m.def("constraint_type_to_string", &constraint_type_to_string);

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

    pybind11::class_<DynamicObstacle>(m, "DynamicObstacle")
        .def(pybind11::init<>())
        .def_readwrite("name", &DynamicObstacle::name)
        .def_readwrite("radius", &DynamicObstacle::radius)
        .def_readwrite("position", &DynamicObstacle::position)
        .def_readwrite("velocity", &DynamicObstacle::velocity)
        .def_readwrite("acceleration", &DynamicObstacle::acceleration);

    pybind11::class_<ObstacleSettings>(m, "ObstacleSettings")
        .def(pybind11::init<>())
        .def_readwrite("enabled", &ObstacleSettings::enabled)
        .def_readwrite("collision_link_pairs",
                       &ObstacleSettings::collision_link_pairs)
        .def_readwrite("minimum_distance", &ObstacleSettings::minimum_distance)
        .def_readwrite("constraint_type", &ObstacleSettings::constraint_type)
        .def_readwrite("mu", &ObstacleSettings::mu)
        .def_readwrite("delta", &ObstacleSettings::delta)
        .def_readwrite("obstacle_urdf_path",
                       &ObstacleSettings::obstacle_urdf_path)
        .def_readwrite("dynamic_obstacles",
                       &ObstacleSettings::dynamic_obstacles)
        .def_readwrite("extra_spheres", &ObstacleSettings::extra_spheres);

    // pybind11::class_<DynamicObstacleSettings>(m, "DynamicObstacleSettings")
    //     .def(pybind11::init<>())
    //     .def_readwrite("enabled", &DynamicObstacleSettings::enabled)
    //     .def_readwrite("obstacle_radius",
    //                    &DynamicObstacleSettings::obstacle_radius)
    //     .def_readwrite("mu", &DynamicObstacleSettings::mu)
    //     .def_readwrite("delta", &DynamicObstacleSettings::delta)
    //     .def_readwrite("collision_spheres",
    //                    &DynamicObstacleSettings::collision_spheres);

    pybind11::class_<RobotDimensions>(m, "RobotDimensions")
        .def(pybind11::init<>())
        .def_readwrite("q", &RobotDimensions::q)
        .def_readwrite("v", &RobotDimensions::v)
        .def_readwrite("x", &RobotDimensions::x)
        .def_readwrite("u", &RobotDimensions::u);

    pybind11::class_<OptimizationDimensions>(m, "OptimizationDimensions")
        .def(pybind11::init<>())
        .def_readwrite("robot", &OptimizationDimensions::robot)
        .def_readwrite("o", &OptimizationDimensions::o)
        .def_readwrite("c", &OptimizationDimensions::c)
        .def("q", &OptimizationDimensions::q)
        .def("v", &OptimizationDimensions::v)
        .def("x", &OptimizationDimensions::x)
        .def("u", &OptimizationDimensions::u)
        .def("f", &OptimizationDimensions::f);

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
        .def_readwrite("use_constraint",
                       &InertialAlignmentSettings::use_constraint)
        .def_readwrite("use_angular_acceleration",
                       &InertialAlignmentSettings::use_angular_acceleration)
        .def_readwrite("cost_weight", &InertialAlignmentSettings::cost_weight)
        .def_readwrite("contact_plane_normal", &InertialAlignmentSettings::contact_plane_normal)
        .def_readwrite("com", &InertialAlignmentSettings::com);

    pybind11::class_<ControllerSettings> ctrl_settings(m, "ControllerSettings");
    ctrl_settings.def(pybind11::init<>())
        .def_readwrite("gravity", &ControllerSettings::gravity)
        .def_readwrite("solver_method", &ControllerSettings::solver_method)
        .def_readwrite("obstacle_settings",
                       &ControllerSettings::obstacle_settings)
        .def_readwrite("balancing_settings",
                       &ControllerSettings::balancing_settings)
        .def_readwrite("initial_state", &ControllerSettings::initial_state)
        .def_readwrite("input_weight", &ControllerSettings::input_weight)
        .def_readwrite("state_weight", &ControllerSettings::state_weight)
        .def_readwrite("end_effector_weight",
                       &ControllerSettings::end_effector_weight)
        .def_readwrite("limit_constraint_type",
                       &ControllerSettings::limit_constraint_type)
        .def_readwrite("input_limit_lower",
                       &ControllerSettings::input_limit_lower)
        .def_readwrite("input_limit_upper",
                       &ControllerSettings::input_limit_upper)
        .def_readwrite("input_limit_mu", &ControllerSettings::input_limit_mu)
        .def_readwrite("input_limit_delta",
                       &ControllerSettings::input_limit_delta)
        .def_readwrite("state_limit_lower",
                       &ControllerSettings::state_limit_lower)
        .def_readwrite("state_limit_upper",
                       &ControllerSettings::state_limit_upper)
        .def_readwrite("state_limit_mu", &ControllerSettings::state_limit_mu)
        .def_readwrite("state_limit_delta",
                       &ControllerSettings::state_limit_delta)
        .def_readwrite("end_effector_box_constraint_enabled",
                       &ControllerSettings::end_effector_box_constraint_enabled)
        .def_readwrite("xyz_lower", &ControllerSettings::xyz_lower)
        .def_readwrite("xyz_upper", &ControllerSettings::xyz_upper)
        .def_readwrite("robot_urdf_path", &ControllerSettings::robot_urdf_path)
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
                       &ControllerSettings::inertial_alignment_settings)
        .def_readwrite("Kp", &ControllerSettings::Kp)
        .def_readwrite("rate", &ControllerSettings::rate)
        .def("solver_method_from_string",
             &ControllerSettings::solver_method_from_string)
        .def("solver_method_to_string",
             &ControllerSettings::solver_method_to_string);

    pybind11::enum_<ControllerSettings::SolverMethod>(ctrl_settings,
                                                      "SolverMethod")
        .value("DDP", ControllerSettings::SolverMethod::DDP)
        .value("SQP", ControllerSettings::SolverMethod::SQP);

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

    pybind11::class_<BalancingConstraintWrapper>(m,
                                                 "BalancingConstraintWrapper")
        .def(pybind11::init<const ControllerSettings &>(), "settings"_a)
        .def("getLinearApproximation",
             &BalancingConstraintWrapper::getLinearApproximation, "t"_a, "x"_a,
             "u"_a);

    /* bind TargetTrajectories class */
    pybind11::class_<ocs2::TargetTrajectories>(m, "TargetTrajectories")
        .def(pybind11::init<ocs2::scalar_array_t, ocs2::vector_array_t,
                            ocs2::vector_array_t>())
        .def("get_desired_state", &ocs2::TargetTrajectories::getDesiredState,
             "t"_a)
        .def("get_desired_input", &ocs2::TargetTrajectories::getDesiredInput,
             "t"_a)
        .def_readonly("ts", &ocs2::TargetTrajectories::timeTrajectory)
        .def_readonly("xs", &ocs2::TargetTrajectories::stateTrajectory)
        .def_readonly("us", &ocs2::TargetTrajectories::inputTrajectory);

    // pybind11::class_<ocs2::OperatingPoints>(m, "OperatingPoints")
    //     .def(pybind11::init<ocs2::scalar_array_t, ocs2::vector_array_t,
    //                         ocs2::vector_array_t>());

    pybind11::class_<ocs2::LinearController>(m, "LinearController")
        .def(pybind11::init<ocs2::scalar_array_t, ocs2::vector_array_t,
                            ocs2::matrix_array_t>(),
             "times"_a, "biases"_a, "gains"_a)
        .def("computeInput", &ocs2::LinearController::computeInput, "t"_a,
             "x"_a)
        // We need an intermediate function to ensure the data is copied
        // correctly; otherwise the vector of pointers misbehaves.
        .def_static(
            "unflatten",
            [](const size_array_t &stateDim, const size_array_t &inputDim,
               const scalar_array_t &timeArray,
               const std::vector<std::vector<float>> &data) {
                std::vector<std::vector<float> const *> data_ptr_array(
                    data.size(), nullptr);
                for (int i = 0; i < data.size(); i++) {
                    data_ptr_array[i] = &(data[i]);
                }
                return LinearController::unFlatten(stateDim, inputDim,
                                                   timeArray, data_ptr_array);
            })
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

    pybind11::class_<ocs2::FeedforwardController>(m, "FeedforwardController")
        .def(pybind11::init<ocs2::scalar_array_t, ocs2::vector_array_t>(),
             "times"_a, "inputs"_a)
        .def("computeInput", &ocs2::FeedforwardController::computeInput, "t"_a,
             "x"_a)
        .def_static("unflatten",
                    [](const scalar_array_t &timeArray,
                       const std::vector<std::vector<float>> &data) {
                        std::vector<std::vector<float> const *> data_ptr_array(
                            data.size(), nullptr);
                        for (int i = 0; i < data.size(); i++) {
                            data_ptr_array[i] = &(data[i]);
                        }
                        return FeedforwardController::unFlatten(timeArray,
                                                                data_ptr_array);
                    });

    /* bind the actual mpc interface */
    pybind11::class_<ControllerPythonInterface>(m, "ControllerInterface")
        .def(pybind11::init<const ControllerSettings &>(), "settings"_a)
        .def("getStateDim", &ControllerPythonInterface::getStateDim)
        .def("getInputDim", &ControllerPythonInterface::getInputDim)
        .def("setObservation", &ControllerPythonInterface::setObservation,
             "t"_a, "x"_a.noconvert(), "u"_a.noconvert())
        .def("setTargetTrajectories",
             &ControllerPythonInterface::setTargetTrajectories,
             "targetTrajectories"_a)
        .def("reset", &ControllerPythonInterface::reset, "targetTrajectories"_a)
        .def("advanceMpc", &ControllerPythonInterface::advanceMpc)
        .def("getMpcSolution", &ControllerPythonInterface::getMpcSolution,
             "t"_a.noconvert(), "x"_a.noconvert(), "u"_a.noconvert())
        .def("evaluateMpcSolution",
             &ControllerPythonInterface::evaluateMpcSolution,
             "current_time"_a.noconvert(), "current_state"_a.noconvert(),
             "opt_state"_a.noconvert(), "opt_input"_a.noconvert())
        .def("getLinearFeedbackGain",
             &ControllerPythonInterface::getLinearFeedbackGain,
             "t"_a.noconvert())
        .def("getBias", &ControllerPythonInterface::getBias, "t"_a.noconvert())
        .def("getLinearController",
             &ControllerPythonInterface::getLinearController)
        .def("flowMap", &ControllerPythonInterface::flowMap, "t"_a,
             "x"_a.noconvert(), "u"_a.noconvert())
        .def("flowMapLinearApproximation",
             &ControllerPythonInterface::flowMapLinearApproximation, "t"_a,
             "x"_a.noconvert(), "u"_a.noconvert())
        .def("cost", &ControllerPythonInterface::cost, "t"_a, "x"_a.noconvert(),
             "u"_a.noconvert())
        .def("costQuadraticApproximation",
             &ControllerPythonInterface::costQuadraticApproximation, "t"_a,
             "x"_a.noconvert(), "u"_a.noconvert())
        .def("valueFunction", &ControllerPythonInterface::valueFunction, "t"_a,
             "x"_a.noconvert())
        .def("valueFunctionStateDerivative",
             &ControllerPythonInterface::valueFunctionStateDerivative, "t"_a,
             "x"_a.noconvert())
        .def("stateInputEqualityConstraint",
             &ControllerPythonInterface::stateInputEqualityConstraint, "t"_a,
             "x"_a.noconvert(), "u"_a.noconvert())
        .def("stateInputEqualityConstraintLinearApproximation",
             &ControllerPythonInterface::
                 stateInputEqualityConstraintLinearApproximation,
             "t"_a, "x"_a.noconvert(), "u"_a.noconvert())
        .def("stateInputEqualityConstraintLagrangian",
             &ControllerPythonInterface::stateInputEqualityConstraintLagrangian,
             "t"_a, "x"_a.noconvert(), "u"_a.noconvert())

        .def("getStateInputEqualityConstraintValue",
             &ControllerPythonInterface::getStateInputEqualityConstraintValue,
             "name"_a, "t"_a, "x"_a.noconvert(), "u"_a.noconvert())
        .def("getStateInputInequalityConstraintValue",
             &ControllerPythonInterface::getStateInputInequalityConstraintValue,
             "name"_a, "t"_a, "x"_a.noconvert(), "u"_a.noconvert())
        .def("getSoftStateInputInequalityConstraintValue",
             &ControllerPythonInterface::
                 getSoftStateInputInequalityConstraintValue,
             "name"_a, "t"_a, "x"_a.noconvert(), "u"_a.noconvert())
        .def("getStateInequalityConstraintValue",
             &ControllerPythonInterface::getStateInequalityConstraintValue,
             "name"_a, "t"_a, "x"_a.noconvert())
        .def("getSoftStateInequalityConstraintValue",
             &ControllerPythonInterface::getSoftStateInequalityConstraintValue,
             "name"_a, "t"_a, "x"_a.noconvert())

        .def("visualizeTrajectory",
             &ControllerPythonInterface::visualizeTrajectory, "t"_a.noconvert(),
             "x"_a.noconvert(), "u"_a.noconvert(), "speed"_a);
}
