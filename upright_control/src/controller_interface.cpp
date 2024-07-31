// NOTE: pinocchio needs to be included before other things to prevent the
// compiler fussing
#include "upright_control/controller_interface.h"

#include <hpp/fcl/shape/geometric_shapes.h>
#include <ocs2_core/constraint/BoundConstraint.h>
#include <ocs2_core/constraint/LinearStateConstraint.h>
#include <ocs2_core/constraint/LinearStateInputConstraint.h>
#include <ocs2_core/cost/QuadraticStateCost.h>
#include <ocs2_core/initialization/DefaultInitializer.h>
#include <ocs2_core/initialization/OperatingPoints.h>
#include <ocs2_core/integration/SensitivityIntegrator.h>
#include <ocs2_core/penalties/penalties/DoubleSidedPenalty.h>
#include <ocs2_core/penalties/penalties/QuadraticPenalty.h>
#include <ocs2_core/penalties/penalties/RelaxedBarrierPenalty.h>
#include <ocs2_core/penalties/penalties/SquaredHingePenalty.h>
#include <ocs2_core/soft_constraint/StateInputSoftConstraint.h>
#include <ocs2_core/soft_constraint/StateSoftConstraint.h>
#include <ocs2_oc/rollout/TimeTriggeredRollout.h>
#include <ocs2_oc/synchronized_module/ReferenceManager.h>
#include <ocs2_pinocchio_interface/PinocchioEndEffectorKinematics.h>
#include <ocs2_pinocchio_interface/PinocchioEndEffectorKinematicsCppAd.h>
#include <ocs2_pinocchio_interface/PinocchioInterface.h>
#include <ocs2_self_collision/PinocchioGeometryInterface.h>
#include <ocs2_self_collision/SelfCollisionConstraintCppAd.h>
#include <ocs2_sqp/MultipleShootingMpc.h>
#include <upright_control/constraint/balancing_constraints.h>
#include <upright_control/constraint/end_effector_box_constraint.h>
#include <upright_control/constraint/joint_state_input_limits.h>
#include <upright_control/constraint/obstacle_constraint.h>
#include <upright_control/constraint/projectile_path_constraint.h>
#include <upright_control/constraint/projectile_plane_constraint.h>
#include <upright_control/constraint/state_to_state_input_constraint.h>
#include <upright_control/cost/end_effector_cost.h>
#include <upright_control/cost/quadratic_joint_state_input_cost.h>
#include <upright_control/dynamics/base_type.h>
#include <upright_control/dynamics/system_dynamics.h>
#include <upright_control/dynamics/system_pinocchio_mapping.h>
#include <upright_control/inertial_alignment.h>
#include <upright_control/util.h>

#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/kinematics.hpp>
#include <pinocchio/algorithm/model.hpp>
#include <pinocchio/fwd.hpp>
#include <pinocchio/multibody/geometry.hpp>
#include <pinocchio/multibody/joint/joint-composite.hpp>
#include <pinocchio/multibody/model.hpp>
#include <pinocchio/parsers/urdf.hpp>

namespace upright {

std::tuple<pinocchio::Model, pinocchio::GeometryModel>
build_dynamic_obstacle_model(const std::vector<DynamicObstacle>& obstacles) {
    pinocchio::Model model;
    model.name = "dynamic_obstacles";
    pinocchio::GeometryModel geom_model;

    for (const auto& obstacle : obstacles) {
        // free-floating joint
        std::string joint_name = obstacle.name + "_joint";
        auto joint_placement = pinocchio::SE3::Identity();
        auto joint_id = model.addJoint(0, pinocchio::JointModelTranslation(),
                                       joint_placement, joint_name);

        // body
        ocs2::scalar_t mass = 1.0;
        auto body_placement = pinocchio::SE3::Identity();
        auto inertia = pinocchio::Inertia::FromSphere(mass, obstacle.radius);
        model.appendBodyToJoint(joint_id, inertia, body_placement);

        // collision model
        pinocchio::GeometryObject::CollisionGeometryPtr shape_ptr(
            new hpp::fcl::Sphere(obstacle.radius));
        pinocchio::GeometryObject geom_obj(obstacle.name, joint_id, shape_ptr,
                                           body_placement);
        geom_model.addGeometryObject(geom_obj);
    }

    return {model, geom_model};
}

pinocchio::GeometryModel build_geometry_model(const std::string& urdf_path) {
    ocs2::PinocchioInterface::Model model;
    pinocchio::urdf::buildModel(urdf_path, model);
    pinocchio::GeometryModel geom_model;
    pinocchio::urdf::buildGeom(model, urdf_path, pinocchio::COLLISION,
                               geom_model);
    return geom_model;
}

void add_ground_plane(const ocs2::PinocchioInterface::Model& model,
                      pinocchio::GeometryModel& geom_model) {
    auto ground_placement = pinocchio::SE3::Identity();
    pinocchio::GeometryObject::CollisionGeometryPtr ground_shape_ptr(
        new hpp::fcl::Halfspace(Vec3d::UnitZ(), 0));
    pinocchio::GeometryObject ground_obj("ground", model.frames[0].parent,
                                         ground_shape_ptr, ground_placement);
    geom_model.addGeometryObject(ground_obj);
}

ControllerInterface::ControllerInterface(const ControllerSettings& settings)
    : settings_(settings) {
    std::cerr << "library folder = " << settings_.lib_folder << std::endl;

    // Pinocchio interface
    std::cerr << "Robot URDF: " << settings_.robot_urdf_path << std::endl;
    pinocchio_interface_ptr.reset(
        new ocs2::PinocchioInterface(build_pinocchio_interface(
            settings_.robot_urdf_path, settings_.robot_base_type,
            settings_.locked_joints, settings_.base_pose)));

    const bool recompile_libraries = settings_.recompile_libraries;
    settings_.sqp.integratorType = ocs2::SensitivityIntegratorType::RK4;

    // Dynamics
    // NOTE: we don't have any branches here because every system we use
    // currently is an integrator
    std::unique_ptr<ocs2::SystemDynamicsBase> dynamics_ptr(
        new SystemDynamics<TripleIntegratorDynamics<ocs2::ad_scalar_t>>(
            "system_dynamics", settings_.dims, settings_.lib_folder,
            recompile_libraries, true));

    // Rollout
    rollout_ptr_.reset(
        new ocs2::TimeTriggeredRollout(*dynamics_ptr, settings_.rollout));

    // Reference manager
    reference_manager_ptr_.reset(new ocs2::ReferenceManager);

    // Optimal control problem
    problem_.dynamicsPtr = std::move(dynamics_ptr);

    // Regularization cost
    problem_.costPtr->add("state_input_cost", get_quadratic_state_input_cost());

    // Build the end effector kinematics
    SystemPinocchioMapping<TripleIntegratorPinocchioMapping<ocs2::ad_scalar_t>,
                           ocs2::ad_scalar_t>
        mapping(settings_.dims);

    /* Constraints */
    const bool frictionless = (settings_.dims.nf == 1);
    // if (frictionless) {
        problem_.boundConstraintPtr->setZero(settings_.dims.x(),
                                             settings_.dims.u());
    // } else {
    //     problem_.boundConstraintPtr->setZero(settings_.dims.x(),
    //                                          settings_.dims.robot.u);
    // }
    problem_.boundConstraintPtr->state_lb_.head(settings_.dims.robot.x) =
        settings_.state_limit_lower;
    problem_.boundConstraintPtr->state_ub_.head(settings_.dims.robot.x) =
        settings_.state_limit_upper;
    problem_.boundConstraintPtr->setStateIndices(0, settings_.dims.robot.x);

    problem_.boundConstraintPtr->input_lb_.head(settings_.dims.robot.u) =
        settings_.input_limit_lower;
    problem_.boundConstraintPtr->input_ub_.head(settings_.dims.robot.u) =
        settings_.input_limit_upper;
    problem_.boundConstraintPtr->setInputIndices(0, settings_.dims.robot.u);

    // Collision avoidance
    if (settings_.obstacle_settings.enabled) {
        ocs2::PinocchioGeometryInterface geom_interface(
            *pinocchio_interface_ptr);

        // Add obstacle collision objects to the geometry model, so we can check
        // them against the robot.
        std::string obs_urdf_path =
            settings_.obstacle_settings.obstacle_urdf_path;
        if (obs_urdf_path.size() > 0) {
            std::cout << "Obstacle URDF: " << obs_urdf_path << std::endl;
            pinocchio::GeometryModel obs_geom_model =
                build_geometry_model(obs_urdf_path);
            geom_interface.addGeometryObjects(obs_geom_model);
        }

        const auto& model = pinocchio_interface_ptr->getModel();
        auto& geom_model = geom_interface.getGeometryModel();
        add_ground_plane(model, geom_model);

        // Add dynamic obstacles.
        if (settings_.obstacle_settings.dynamic_obstacles.size() > 0) {
            ocs2::PinocchioInterface::Model dyn_obs_model, new_model;
            pinocchio::GeometryModel dyn_obs_geom_model, new_geom_model;

            std::tie(dyn_obs_model, dyn_obs_geom_model) =
                build_dynamic_obstacle_model(
                    settings_.obstacle_settings.dynamic_obstacles);

            // Update models
            pinocchio::appendModel(
                model, dyn_obs_model, geom_model, dyn_obs_geom_model, 0,
                pinocchio::SE3::Identity(), new_model, new_geom_model);

            pinocchio_interface_ptr.reset(
                new ocs2::PinocchioInterface(new_model));
            geom_interface = ocs2::PinocchioGeometryInterface(new_geom_model);
        }

        std::cout << *pinocchio_interface_ptr << std::endl;

        // Get the usual state constraint
        std::unique_ptr<ocs2::StateConstraint> obstacle_constraint =
            get_obstacle_constraint(*pinocchio_interface_ptr, geom_interface,
                                    settings_.obstacle_settings,
                                    settings_.lib_folder, recompile_libraries);

        // Map it to a state-input constraint so it works with the current
        // implementation of the hard inequality constraints
        problem_.inequalityConstraintPtr->add(
            "obstacle_avoidance",
            std::unique_ptr<ocs2::StateInputConstraint>(
                new StateToStateInputConstraint(*obstacle_constraint)));
        std::cerr << "Hard obstacle avoidance constraints are enabled."
                  << std::endl;
    } else {
        std::cerr << "Obstacle avoidance is disabled." << std::endl;
    }

    ocs2::PinocchioEndEffectorKinematicsCppAd end_effector_kinematics(
        *pinocchio_interface_ptr, mapping, {settings_.end_effector_link_name},
        settings_.dims.x(), settings_.dims.u(), "end_effector_kinematics",
        settings_.lib_folder, recompile_libraries, false);

    // Store for possible use by other callers.
    end_effector_kinematics_ptr_.reset(end_effector_kinematics.clone());

    // End effector pose cost
    std::unique_ptr<ocs2::StateCost> end_effector_cost(new EndEffectorCost(
        settings_.end_effector_weight, end_effector_kinematics));
    problem_.stateCostPtr->add("end_effector_cost",
                               std::move(end_effector_cost));

    // Alternative auto-diff version with full Hessian
    // std::unique_ptr<ocs2::StateCost> end_effector_cost(new
    // EndEffectorCostCppAd(
    //     settings_.end_effector_weight, end_effector_kinematics,
    //     settings_.dims, recompile_libraries));
    // problem_.stateCostPtr->add("end_effector_cost",
    //                            std::move(end_effector_cost));

    // End effector position box constraint
    if (settings_.end_effector_box_constraint_enabled) {
        std::cout << "End effector box constraint is enabled." << std::endl;
        std::unique_ptr<ocs2::StateConstraint> end_effector_box_constraint(
            new EndEffectorBoxConstraint(
                settings_.xyz_lower, settings_.xyz_upper,
                end_effector_kinematics, *reference_manager_ptr_));
        problem_.inequalityConstraintPtr->add(
            "end_effector_box_constraint",
            std::unique_ptr<ocs2::StateInputConstraint>(
                new StateToStateInputConstraint(*end_effector_box_constraint)));
    } else {
        std::cout << "End effector box constraint is disabled." << std::endl;
    }

    // Experimental: projectile avoidance constraint
    if (settings_.projectile_path_constraint_enabled) {
        // TODO: hardcoded link name
        ocs2::PinocchioEndEffectorKinematicsCppAd
            end_effector_collision_kinematics(
                *pinocchio_interface_ptr, mapping,
                settings_.projectile_path_collision_links, settings_.dims.x(),
                settings_.dims.u(), "end_effector_collision_kinematics",
                settings_.lib_folder, recompile_libraries, false);

        std::unique_ptr<ocs2::StateConstraint> projectile_constraint(
            new ProjectilePathConstraint(end_effector_collision_kinematics,
                                         *reference_manager_ptr_,
                                         settings_.projectile_path_distances,
                                         settings_.projectile_path_scale));
        // new ProjectilePlaneConstraint(end_effector_collision_kinematics,
        //                              *reference_manager_ptr_,
        //                              settings_.projectile_path_distance));
        problem_.inequalityConstraintPtr->add(
            "projectile_constraint",
            std::unique_ptr<ocs2::StateInputConstraint>(
                new StateToStateInputConstraint(*projectile_constraint)));
    }

    // Inertial alignment
    if (settings_.inertial_alignment_settings.cost_enabled) {
        std::unique_ptr<ocs2::StateInputCost> inertial_alignment_cost(
            new InertialAlignmentCostGaussNewton(
                end_effector_kinematics, settings_.inertial_alignment_settings,
                settings_.gravity, settings_.dims, true));
        problem_.costPtr->add("inertial_alignment_cost",
                              std::move(inertial_alignment_cost));
        std::cout << "Inertial alignment cost enabled." << std::endl;
    }
    if (settings_.inertial_alignment_settings.constraint_enabled) {
        std::unique_ptr<ocs2::StateInputConstraint>
            inertial_alignment_constraint(new InertialAlignmentConstraint(
                end_effector_kinematics, settings_.inertial_alignment_settings,
                settings_.gravity, settings_.dims, recompile_libraries));
        problem_.inequalityConstraintPtr->add(
            "inertial_alignment_constraint",
            std::move(inertial_alignment_constraint));
        std::cout << "Inertial alignment constraint enabled." << std::endl;
    }

    if (settings_.balancing_settings.enabled) {
        problem_.equalityConstraintPtr->add(
            "object_dynamics",
            get_object_dynamics_constraint(end_effector_kinematics,
                                           recompile_libraries));

        // Inequalities for the friction cones
        if (frictionless) {
            // lower bounds are already zero, make the upper ones
            // arbitrary high values
            problem_.boundConstraintPtr->input_ub_.tail(settings_.dims.f())
                .setConstant(1e2);
            // indicate that all inputs are now box constrained (real
            // inputs and contact forces)
            problem_.boundConstraintPtr->setInputIndices(0, settings_.dims.u());

            std::cout << "input lb = "
                      << problem_.boundConstraintPtr->input_lb_.transpose()
                      << std::endl;
            std::cout << "input ub = "
                      << problem_.boundConstraintPtr->input_ub_.transpose()
                      << std::endl;
        } else {
            problem_.inequalityConstraintPtr->add(
                "contact_forces",
                get_contact_force_constraint(end_effector_kinematics,
                                             recompile_libraries));

            // TODO try to keep the bounds but make them large
            problem_.boundConstraintPtr->setInputIndices(0, settings_.dims.u());
            problem_.boundConstraintPtr->input_ub_.tail(settings_.dims.f())
                .setConstant(1e2);
            problem_.boundConstraintPtr->input_lb_.tail(settings_.dims.f())
                .setConstant(-1e2);
        }

        std::cout << "input bound idx = ";
        for (auto i : problem_.boundConstraintPtr->input_idx_) {
            std::cout << i << " ";
        }
        std::cout << std::endl;
        std::cout << "input bound ub = " << problem_.boundConstraintPtr->input_ub_.transpose() << std::endl;
        std::cout << "input bound lb = " << problem_.boundConstraintPtr->input_lb_.transpose() << std::endl;

    } else {
        std::cerr << "Balancing constraints disabled." << std::endl;
    }

    // TODO: disable bound constraints
    problem_.boundConstraintPtr->state_idx_.resize(0);
    problem_.boundConstraintPtr->input_idx_.resize(0);

    // Initialization state
    if (settings_.use_operating_points) {
        initializer_ptr_.reset(new ocs2::OperatingPoints(
            settings_.operating_times, settings_.operating_states,
            settings_.operating_inputs));
    } else {
        initializer_ptr_.reset(
            new ocs2::DefaultInitializer(settings_.dims.u()));
    }

    // reference_manager_ptr_->setTargetTrajectories(settings_.target_trajectory);

    initial_state_ = settings_.initial_state;
    std::cerr << "Initial State:   " << initial_state_.transpose() << std::endl;
}

std::unique_ptr<ocs2::MPC_BASE> ControllerInterface::get_mpc() {
    return std::unique_ptr<ocs2::MPC_BASE>(new ocs2::MultipleShootingMpc(
        settings_.mpc, settings_.sqp, problem_, *initializer_ptr_));
}

std::unique_ptr<ocs2::StateInputCost>
ControllerInterface::get_quadratic_state_input_cost() {
    // augment R with cost on the contact forces
    MatXd input_weight =
        settings_.balancing_settings.force_weight *
        MatXd::Identity(settings_.dims.u(), settings_.dims.u());
    input_weight.topLeftCorner(settings_.dims.robot.u, settings_.dims.robot.u) =
        settings_.input_weight;

    // TODO do I need weight on obstacle dynamics?
    MatXd state_weight = MatXd::Zero(settings_.dims.x(), settings_.dims.x());
    state_weight.topLeftCorner(settings_.dims.robot.x, settings_.dims.robot.x) =
        settings_.state_weight;

    std::cout << "Q: " << state_weight << std::endl;
    std::cout << "R: " << input_weight << std::endl;

    return std::unique_ptr<ocs2::StateInputCost>(
        new QuadraticJointStateInputCost(state_weight, input_weight,
                                         settings_.xd));
}

std::unique_ptr<ocs2::StateInputConstraint>
ControllerInterface::get_object_dynamics_constraint(
    const ocs2::PinocchioEndEffectorKinematicsCppAd& end_effector_kinematics,
    bool recompile_libraries) {
    return std::unique_ptr<ocs2::StateInputConstraint>(
        new ObjectDynamicsConstraints(
            end_effector_kinematics, settings_.balancing_settings,
            settings_.gravity, settings_.dims, recompile_libraries));
}

std::unique_ptr<ocs2::StateInputConstraint>
ControllerInterface::get_contact_force_constraint(
    const ocs2::PinocchioEndEffectorKinematicsCppAd& end_effector_kinematics,
    bool recompile_libraries) {
    return std::unique_ptr<ocs2::StateInputConstraint>(
        new ContactForceBalancingConstraints(
            end_effector_kinematics, settings_.balancing_settings,
            settings_.gravity, settings_.dims, recompile_libraries));
}

std::unique_ptr<ocs2::StateConstraint>
ControllerInterface::get_obstacle_constraint(
    ocs2::PinocchioInterface& pinocchio_interface,
    ocs2::PinocchioGeometryInterface& geom_interface,
    const ObstacleSettings& settings, const std::string& library_folder,
    bool recompile_libraries) {
    const auto& geom_model = geom_interface.getGeometryModel();
    for (int i = 0; i < geom_model.ngeoms; ++i) {
        std::cout << geom_model.geometryObjects[i].name << std::endl;
    }

    geom_interface.addCollisionPairsByName(settings.collision_link_pairs);

    const size_t nc = geom_interface.getNumCollisionPairs();
    std::cerr << "Testing for " << nc << " collision pairs." << std::endl;

    std::vector<hpp::fcl::DistanceResult> distances =
        geom_interface.computeDistances(pinocchio_interface);
    for (int i = 0; i < distances.size(); ++i) {
        std::cout << "dist = " << distances[i].min_distance << std::endl;
    }

    SystemPinocchioMapping<TripleIntegratorPinocchioMapping<ocs2::scalar_t>,
                           ocs2::scalar_t>
        mapping(settings_.dims);

    return std::unique_ptr<ocs2::StateConstraint>(
        new ocs2::SelfCollisionConstraintCppAd(
            pinocchio_interface, mapping, geom_interface,
            settings.minimum_distance, "obstacle_avoidance", library_folder,
            recompile_libraries, false));
}

std::unique_ptr<ocs2::StateInputConstraint>
ControllerInterface::get_joint_state_input_limit_constraint() {
    VecXd state_limit_lower = settings_.state_limit_lower;
    VecXd state_limit_upper = settings_.state_limit_upper;

    VecXd input_limit_lower = settings_.input_limit_lower;
    VecXd input_limit_upper = settings_.input_limit_upper;

    std::cout << "state limit lower: " << state_limit_lower.transpose()
              << std::endl;
    std::cout << "state limit upper: " << state_limit_upper.transpose()
              << std::endl;
    std::cout << "input limit lower: " << input_limit_lower.transpose()
              << std::endl;
    std::cout << "input limit upper: " << input_limit_upper.transpose()
              << std::endl;

    return std::unique_ptr<ocs2::StateInputConstraint>(
        new JointStateInputLimits(settings_.dims));
}

}  // namespace upright
