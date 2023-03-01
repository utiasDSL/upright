#include <hpipm_catkin/HpipmInterfaceSettings.h>
#include <ocs2_core/Types.h>
#include <ocs2_core/control/FeedforwardController.h>
#include <ocs2_core/control/LinearController.h>
#include <ocs2_mpc/MPC_Settings.h>
#include <ocs2_oc/rollout/RolloutSettings.h>
#include <ocs2_python_interface/PybindMacros.h>
#include <ocs2_sqp/MultipleShootingSettings.h>
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
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
PYBIND11_MAKE_OPAQUE(std::vector<DynamicObstacleMode>)
PYBIND11_MAKE_OPAQUE(std::map<std::string, ocs2::scalar_t>)

/* create a python module */
PYBIND11_MODULE(bindings, m) {
    /* bind vector types so they can be used natively in python */
    VECTOR_TYPE_BINDING(ocs2::scalar_array_t, "scalar_array")
    VECTOR_TYPE_BINDING(ocs2::vector_array_t, "vector_array")
    VECTOR_TYPE_BINDING(ocs2::matrix_array_t, "matrix_array")

    VECTOR_TYPE_BINDING(CollisionSphereVector, "CollisionSphereVector")
    VECTOR_TYPE_BINDING(StringPairVector, "StringPairVector")
    VECTOR_TYPE_BINDING(std::vector<DynamicObstacle>, "DynamicObstacleVector")
    VECTOR_TYPE_BINDING(std::vector<DynamicObstacleMode>,
                        "DynamicObstacleModeVector")

    pybind11::bind_map<std::map<std::string, ocs2::scalar_t>>(
        m, "MapStringScalar");

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
        .def_readwrite("arrangement_name", &BalancingSettings::arrangement_name)
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

    pybind11::class_<DynamicObstacleMode>(m, "DynamicObstacleMode")
        .def(pybind11::init<>())
        .def_readwrite("time", &DynamicObstacleMode::time)
        .def_readwrite("position", &DynamicObstacleMode::position)
        .def_readwrite("velocity", &DynamicObstacleMode::velocity)
        .def_readwrite("acceleration", &DynamicObstacleMode::acceleration);

    pybind11::class_<DynamicObstacle>(m, "DynamicObstacle")
        .def(pybind11::init<>())
        .def_readwrite("name", &DynamicObstacle::name)
        .def_readwrite("radius", &DynamicObstacle::radius)
        .def_readwrite("modes", &DynamicObstacle::modes);

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
        .def_readwrite("nf", &OptimizationDimensions::nf)
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
        .def_readwrite("cost_enabled", &InertialAlignmentSettings::cost_enabled)
        .def_readwrite("constraint_enabled",
                       &InertialAlignmentSettings::constraint_enabled)
        .def_readwrite("use_angular_acceleration",
                       &InertialAlignmentSettings::use_angular_acceleration)
        .def_readwrite("align_with_fixed_vector",
                       &InertialAlignmentSettings::align_with_fixed_vector)
        .def_readwrite("cost_weight", &InertialAlignmentSettings::cost_weight)
        .def_readwrite("alpha", &InertialAlignmentSettings::alpha)
        .def_readwrite("contact_plane_normal",
                       &InertialAlignmentSettings::contact_plane_normal)
        .def_readwrite("contact_plane_span",
                       &InertialAlignmentSettings::contact_plane_span)
        .def_readwrite("com", &InertialAlignmentSettings::com);

    pybind11::class_<ocs2::mpc::Settings>(m, "MPCSettings")
        .def(pybind11::init<>())
        .def_readwrite("time_horizon", &ocs2::mpc::Settings::timeHorizon_)
        .def_readwrite("debug_print", &ocs2::mpc::Settings::debugPrint_)
        .def_readwrite("cold_start", &ocs2::mpc::Settings::coldStart_);

    pybind11::class_<ocs2::rollout::Settings>(m, "RolloutSettings")
        .def(pybind11::init<>())
        .def_readwrite("abs_tol_ode", &ocs2::rollout::Settings::absTolODE)
        .def_readwrite("rel_tol_ode", &ocs2::rollout::Settings::relTolODE)
        .def_readwrite("max_num_steps_per_second",
                       &ocs2::rollout::Settings::maxNumStepsPerSecond)
        .def_readwrite("timestep", &ocs2::rollout::Settings::timeStep)
        .def_readwrite("check_numerical_stability",
                       &ocs2::rollout::Settings::checkNumericalStability);

    pybind11::class_<ocs2::hpipm_interface::SlackSettings>(m, "SlackSettings")
        .def(pybind11::init<>())
        .def_readwrite("enabled",
                       &ocs2::hpipm_interface::SlackSettings::enabled)
        .def_readwrite("upper_L2_penalty",
                       &ocs2::hpipm_interface::SlackSettings::upper_L2_penalty)
        .def_readwrite("lower_L2_penalty",
                       &ocs2::hpipm_interface::SlackSettings::lower_L2_penalty)
        .def_readwrite("upper_L1_penalty",
                       &ocs2::hpipm_interface::SlackSettings::upper_L1_penalty)
        .def_readwrite("lower_L1_penalty",
                       &ocs2::hpipm_interface::SlackSettings::lower_L1_penalty)
        .def_readwrite("upper_low_bound",
                       &ocs2::hpipm_interface::SlackSettings::upper_low_bound)
        .def_readwrite("lower_low_bound",
                       &ocs2::hpipm_interface::SlackSettings::lower_low_bound);

    pybind11::class_<ocs2::hpipm_interface::Settings>(m, "HPIPMSettings")
        .def(pybind11::init<>())
        .def_readwrite("iter_max", &ocs2::hpipm_interface::Settings::iter_max)
        .def_readwrite("warm_start",
                       &ocs2::hpipm_interface::Settings::warm_start)
        .def_readwrite("slacks", &ocs2::hpipm_interface::Settings::slacks);

    pybind11::class_<ocs2::multiple_shooting::Settings>(m, "SQPSettings")
        .def(pybind11::init<>())
        .def_readwrite("sqp_iteration",
                       &ocs2::multiple_shooting::Settings::sqpIteration)
        .def_readwrite("init_sqp_iteration",
                       &ocs2::multiple_shooting::Settings::initSqpIteration)
        .def_readwrite("delta_tol",
                       &ocs2::multiple_shooting::Settings::deltaTol)
        .def_readwrite("cost_tol", &ocs2::multiple_shooting::Settings::costTol)
        .def_readwrite("use_feedback_policy",
                       &ocs2::multiple_shooting::Settings::useFeedbackPolicy)
        .def_readwrite("dt", &ocs2::multiple_shooting::Settings::dt)
        .def_readwrite("project_state_input_equality_constraints",
                       &ocs2::multiple_shooting::Settings::
                           projectStateInputEqualityConstraints)
        .def_readwrite("print_solver_status",
                       &ocs2::multiple_shooting::Settings::printSolverStatus)
        .def_readwrite(
            "print_solver_statistics",
            &ocs2::multiple_shooting::Settings::printSolverStatistics)
        .def_readwrite("print_line_search",
                       &ocs2::multiple_shooting::Settings::printLinesearch)
        .def_readwrite("hpipm",
                       &ocs2::multiple_shooting::Settings::hpipmSettings);

    pybind11::class_<TrackingSettings>(m, "TrackingSettings")
        .def(pybind11::init<>())
        .def_readwrite("rate", &TrackingSettings::rate)
        .def_readwrite("min_policy_update_time",
                       &TrackingSettings::min_policy_update_time)
        .def_readwrite("kp", &TrackingSettings::kp)
        .def_readwrite("kv", &TrackingSettings::kv)
        .def_readwrite("ka", &TrackingSettings::ka)
        .def_readwrite("enforce_state_limits",
                       &TrackingSettings::enforce_state_limits)
        .def_readwrite("enforce_input_limits",
                       &TrackingSettings::enforce_input_limits)
        .def_readwrite("enforce_ee_position_limits",
                       &TrackingSettings::enforce_ee_position_limits)
        .def_readwrite("use_projectile", &TrackingSettings::use_projectile)
        .def_readwrite("state_violation_margin",
                       &TrackingSettings::state_violation_margin)
        .def_readwrite("input_violation_margin",
                       &TrackingSettings::input_violation_margin)
        .def_readwrite("ee_position_violation_margin",
                       &TrackingSettings::ee_position_violation_margin);

    pybind11::class_<EstimationSettings>(m, "EstimationSettings")
        .def(pybind11::init<>())
        .def_readwrite("robot_init_variance",
                       &EstimationSettings::robot_init_variance)
        .def_readwrite("robot_process_variance",
                       &EstimationSettings::robot_process_variance)
        .def_readwrite("robot_measurement_variance",
                       &EstimationSettings::robot_measurement_variance);

    pybind11::class_<ControllerSettings> ctrl_settings(m, "ControllerSettings");
    ctrl_settings.def(pybind11::init<>())
        .def_readwrite("gravity", &ControllerSettings::gravity)
        .def_readwrite("solver_method", &ControllerSettings::solver_method)
        .def_readwrite("recompile_libraries",
                       &ControllerSettings::recompile_libraries)
        .def_readwrite("debug", &ControllerSettings::debug)
        .def_readwrite("mpc", &ControllerSettings::mpc)
        .def_readwrite("sqp", &ControllerSettings::sqp)
        .def_readwrite("rollout", &ControllerSettings::rollout)
        .def_readwrite("tracking", &ControllerSettings::tracking)
        .def_readwrite("estimation", &ControllerSettings::estimation)
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
        .def_readwrite("projectile_path_constraint_enabled",
                       &ControllerSettings::projectile_path_constraint_enabled)
        .def_readwrite("projectile_path_distances",
                       &ControllerSettings::projectile_path_distances)
        .def_readwrite("projectile_path_scale",
                       &ControllerSettings::projectile_path_scale)
        .def_readwrite("projectile_path_collision_links",
                       &ControllerSettings::projectile_path_collision_links)
        .def_readwrite("robot_urdf_path", &ControllerSettings::robot_urdf_path)
        .def_readwrite("lib_folder", &ControllerSettings::lib_folder)
        .def_readwrite("robot_base_type", &ControllerSettings::robot_base_type)
        .def_readwrite("locked_joints", &ControllerSettings::locked_joints)
        .def_readwrite("base_pose", &ControllerSettings::base_pose)
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
        .def_readwrite("xd", &ControllerSettings::xd)
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
        .def("getCostValue", &ControllerPythonInterface::getCostValue, "name"_a,
             "t"_a, "x"_a.noconvert(), "u"_a.noconvert())
        .def("visualizeTrajectory",
             &ControllerPythonInterface::visualizeTrajectory, "t"_a.noconvert(),
             "x"_a.noconvert(), "u"_a.noconvert(), "speed"_a);
}
