/******************************************************************************
Copyright (c) 2020, Farbod Farshidian. All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

 * Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

 * Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

 * Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
******************************************************************************/

#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/kinematics.hpp>
#include <pinocchio/fwd.hpp>
#include <pinocchio/multibody/joint/joint-composite.hpp>
#include <pinocchio/multibody/model.hpp>

#include <ocs2_core/constraint/LinearStateConstraint.h>
#include <ocs2_core/constraint/LinearStateInputConstraint.h>
#include <ocs2_core/cost/QuadraticStateCost.h>
#include <ocs2_core/initialization/DefaultInitializer.h>
#include <ocs2_core/initialization/OperatingPoints.h>
#include <ocs2_core/misc/LoadData.h>
#include <ocs2_core/penalties/penalties/DoubleSidedPenalty.h>
#include <ocs2_core/penalties/penalties/QuadraticPenalty.h>
#include <ocs2_core/penalties/penalties/RelaxedBarrierPenalty.h>
#include <ocs2_core/penalties/penalties/SquaredHingePenalty.h>
#include <ocs2_core/soft_constraint/StateInputSoftConstraint.h>
#include <ocs2_core/soft_constraint/StateSoftConstraint.h>
#include <ocs2_ddp/DDP_Settings.h>
#include <ocs2_ddp/GaussNewtonDDP_MPC.h>
#include <ocs2_oc/synchronized_module/ReferenceManager.h>
#include <ocs2_pinocchio_interface/PinocchioEndEffectorKinematics.h>
#include <ocs2_pinocchio_interface/PinocchioEndEffectorKinematicsCppAd.h>
#include <ocs2_pinocchio_interface/urdf.h>
#include <ocs2_self_collision/SelfCollisionConstraint.h>
#include <ocs2_self_collision/SelfCollisionConstraintCppAd.h>
#include <ocs2_sqp/MultipleShootingMpc.h>
#include <ocs2_sqp/MultipleShootingSettings.h>

#include <upright_control/constraint/bounded_balancing_constraints.h>
#include <upright_control/constraint/joint_state_input_limits.h>
#include <upright_control/constraint/obstacle_constraint.h>
#include <upright_control/constraint/state_to_state_input_constraint.h>
#include <upright_control/cost/end_effector_cost.h>
#include <upright_control/cost/inertial_alignment_cost.h>
#include <upright_control/cost/quadratic_joint_state_input_cost.h>
#include <upright_control/dynamics/base_type.h>
#include <upright_control/dynamics/fixed_base_dynamics.h>
#include <upright_control/dynamics/fixed_base_pinocchio_mapping.h>
#include <upright_control/dynamics/omnidirectional_dynamics.h>
#include <upright_control/dynamics/omnidirectional_pinocchio_mapping.h>

#include "upright_control/controller_interface.h"

namespace upright {

ControllerInterface::ControllerInterface(const ControllerSettings& settings)
    : settings_(settings) {
    // load setting from config file
    loadSettings();
}

ocs2::PinocchioInterface ControllerInterface::build_pinocchio_interface(
    const std::string& urdf_path) {
    if (settings_.robot_base_type == RobotBaseType::Omnidirectional) {
        // add 3 DOF for wheelbase
        pinocchio::JointModelComposite root_joint(3);
        root_joint.addJoint(pinocchio::JointModelPX());
        root_joint.addJoint(pinocchio::JointModelPY());
        root_joint.addJoint(pinocchio::JointModelRZ());
        // root_joint.addJoint(pinocchio::JointModelRUBZ());

        return ocs2::getPinocchioInterfaceFromUrdfFile(urdf_path, root_joint);
    }
    // Fixed base
    return ocs2::getPinocchioInterfaceFromUrdfFile(urdf_path);
}

pinocchio::GeometryModel ControllerInterface::build_geometry_model(
    const std::string& urdf_path) {
    ocs2::PinocchioInterface::Model model;
    pinocchio::urdf::buildModel(urdf_path, model);
    pinocchio::GeometryModel geom_model;
    pinocchio::urdf::buildGeom(model, urdf_path, pinocchio::COLLISION,
                               geom_model);
    return geom_model;
}

void ControllerInterface::loadSettings() {
    std::string taskFile = settings_.ocs2_config_path;
    std::string libraryFolder = settings_.lib_folder;

    std::cerr << "taskFile = " << taskFile << std::endl;
    std::cerr << "libraryFolder = " << libraryFolder << std::endl;

    boost::property_tree::ptree pt;
    boost::property_tree::read_info(taskFile, pt);

    /*
     * URDF files
     */
    std::cerr << "Robot URDF: " << settings_.robot_urdf_path << std::endl;
    std::cerr << "Obstacle URDF: " << settings_.obstacle_urdf_path << std::endl;

    pinocchioInterfacePtr_.reset(new ocs2::PinocchioInterface(
        build_pinocchio_interface(settings_.robot_urdf_path)));
    std::cerr << *pinocchioInterfacePtr_;

    /*
     * Model settings
     */
    bool usePreComputation = false;
    bool recompileLibraries = true;
    std::cerr << "\n #### Model Settings:";
    std::cerr << "\n #### "
                 "============================================================="
                 "================\n";
    ocs2::loadData::loadPtreeValue(pt, usePreComputation,
                                   "model_settings.usePreComputation", true);
    ocs2::loadData::loadPtreeValue(pt, recompileLibraries,
                                   "model_settings.recompileLibraries", true);
    std::cerr << " #### "
                 "============================================================="
                 "================"
              << std::endl;

    // DDP-MPC settings
    ddpSettings_ = ocs2::ddp::loadSettings(taskFile, "ddp");
    mpcSettings_ = ocs2::mpc::loadSettings(taskFile, "mpc");
    sqpSettings_ =
        ocs2::multiple_shooting::loadSettings(taskFile, "multiple_shooting");

    // sqpSettings_.hpipmSettings.warm_start = true;
    sqpSettings_.hpipmSettings.use_slack = true;

    // Dynamics
    std::vector<std::unique_ptr<Dynamics<ocs2::ad_scalar_t>>> dynamics;

    // Robot dynamics
    // NOTE: we don't have any checks here because every system we use
    // currently is an integrator
    robot_dynamics_ptr = std::unique_ptr<Dynamics<ocs2::ad_scalar_t>>(
        new IntegratorDynamics<ocs2::ad_scalar_t>(settings_.dims.robot));
    dynamics.push_back(robot_dynamics_ptr);

    // Dynamic obstacles (optional)
    for (const auto& obstacles : settings_.dims.o) {
        obstacle_dynamics_ptr = std::unique_ptr<Dynamics<ocs2::ad_scalar_t>>(
            new ObstacleDynamics<ocs2::ad_scalar_t>());
        dynamics.push_back(obstacle_dynamics_ptr);
    }
    // if (settings_.robot_base_type == RobotBaseType::Omnidirectional) {
    //     dynamicsPtr.reset(new OmnidirectionalDynamics(
    //         "robot_dynamics", settings_.dims, libraryFolder,
    //         recompileLibraries, true));
    // } else {
    //     dynamicsPtr.reset(new FixedBaseDynamics("robot_dynamics",
    //                                             settings_.dims,
    //                                             libraryFolder,
    //                                             recompileLibraries, true));
    // }
    std::unique_ptr<ocs2::SystemDynamicsBase> dynamicsPtr(
        new SystemDynamics("robot_dynamics", dynamics, settings_.dims,
                           libraryFolder, recompileLibraries, true));

    // Rollout
    const auto rolloutSettings =
        ocs2::rollout::loadSettings(taskFile, "rollout");
    rolloutPtr_.reset(
        new ocs2::TimeTriggeredRollout(*dynamicsPtr, rolloutSettings));

    // Reference manager
    referenceManagerPtr_.reset(new ocs2::ReferenceManager);

    // Optimal control problem
    problem_.dynamicsPtr = std::move(dynamicsPtr);

    // Cost
    problem_.costPtr->add("state_input_cost",
                          getQuadraticStateInputCost(taskFile));

    // Build the end effector kinematics
    std::unique_ptr<ocs2::PinocchioStateInputMapping<ocs2::ad_scalar_t>>
        pinocchio_mapping_ptr;
    if (settings_.robot_base_type == RobotBaseType::Omnidirectional) {
        pinocchio_mapping_ptr.reset(
            new OmnidirectionalPinocchioMapping<ocs2::ad_scalar_t>(
                settings_.dims.robot(0)));
    } else {
        pinocchio_mapping_ptr.reset(
            new FixedBasePinocchioMapping<ocs2::ad_scalar_t>(
                settings_.dims.robot(0)));
    }

    ocs2::PinocchioEndEffectorKinematicsCppAd end_effector_kinematics(
        *pinocchioInterfacePtr_, *pinocchio_mapping_ptr,
        {settings_.end_effector_link_name}, settings_.dims.x(),
        settings_.dims.u(), "end_effector_kinematics", libraryFolder,
        recompileLibraries, false);

    problem_.stateCostPtr->add("end_effector_cost",
                               getEndEffectorCost(end_effector_kinematics));

    if (settings_.inertial_alignment_settings.enabled) {
        std::unique_ptr<ocs2::StateInputCost> inertial_alignment_cost(
            new InertialAlignmentCost(end_effector_kinematics,
                                      settings_.inertial_alignment_settings,
                                      settings_.gravity, settings_.dims, true));
        problem_.costPtr->add("inertial_alignment_cost",
                              std::move(inertial_alignment_cost));
    }

    /* Constraints */
    if (settings_.limit_constraint_type == ConstraintType::Soft) {
        problem_.softConstraintPtr->add(
            "joint_state_input_limits",
            get_soft_joint_state_input_limit_constraint());
        std::cerr << "Soft state and input limits are enabled." << std::endl;
    } else {
        // TODO this should be a box constraint, but this is not implemented
        // yet
        std::unique_ptr<ocs2::StateInputConstraint>
            joint_state_input_constraint(new JointStateInputConstraint(
                settings_.dims, settings_.state_limit_lower,
                settings_.state_limit_upper, settings_.input_limit_lower,
                settings_.input_limit_upper));
        problem_.inequalityConstraintPtr->add(
            "joint_state_input_limits",
            std::move(joint_state_input_constraint));
        std::cerr << "Hard state and input limits are enabled." << std::endl;
    }

    // Self-collision avoidance and collision avoidance with static obstacles
    if (settings_.static_obstacle_settings.enabled) {
        if (settings_.static_obstacle_settings.constraint_type ==
            ConstraintType::Soft) {
            problem_.stateSoftConstraintPtr->add(
                "static_obstacle_avoidance",
                get_soft_static_obstacle_constraint(
                    *pinocchioInterfacePtr_, settings_.static_obstacle_settings,
                    settings_.obstacle_urdf_path, usePreComputation,
                    libraryFolder, recompileLibraries));
            std::cerr
                << "Soft static obstacle avoidance constraints are enabled."
                << std::endl;
        } else {
            // Get the usual state constraint
            std::unique_ptr<ocs2::StateConstraint> static_obstacle_constraint =
                get_static_obstacle_constraint(
                    *pinocchioInterfacePtr_, settings_.static_obstacle_settings,
                    settings_.obstacle_urdf_path, usePreComputation,
                    libraryFolder, recompileLibraries);

            // Map it to a state-input constraint so it works with the current
            // implementation of the hard inequality constraints
            problem_.inequalityConstraintPtr->add(
                "static_obstacle_avoidance",
                std::unique_ptr<ocs2::StateInputConstraint>(
                    new StateToStateInputConstraint(
                        *static_obstacle_constraint)));
            std::cerr
                << "Hard static obstacle avoidance constraints are enabled."
                << std::endl;
        }
    } else {
        std::cerr << "Static obstacle avoidance is disabled." << std::endl;
    }

    // Collision avoidance with dynamic obstacles specified via the reference
    // trajectory
    if (settings_.dynamic_obstacle_settings.enabled) {
        std::cerr << "Dynamic obstacle avoidance is enabled." << std::endl;
        problem_.stateSoftConstraintPtr->add(
            "dynamic_obstacle_avoidance",
            getDynamicObstacleConstraint(
                *pinocchioInterfacePtr_, settings_.dynamic_obstacle_settings,
                usePreComputation, libraryFolder, recompileLibraries));
    } else {
        std::cerr << "Dynamic obstacle avoidance is disabled." << std::endl;
    }

    // TODO we're getting too nested here
    if (settings_.balancing_settings.enabled) {
        if (settings_.balancing_settings.use_force_constraints) {
            // Currently we always use exact constraints for the object
            // dynamics.
            problem_.equalityConstraintPtr->add(
                "object_dynamics",
                get_object_dynamics_constraint(end_effector_kinematics,
                                               recompileLibraries));

            // Inequalities for the friction cones
            // NOTE: the hard inequality constraints appear to work much better
            // (avoid phantom gradients and such)
            if (settings_.balancing_settings.constraint_type ==
                ConstraintType::Soft) {
                std::cerr
                    << "Soft contact force-based balancing constraints enabled."
                    << std::endl;
                problem_.softConstraintPtr->add(
                    "contact_forces",
                    get_soft_contact_force_constraint(end_effector_kinematics,
                                                      recompileLibraries));
            } else {
                std::cerr
                    << "Hard contact force-based balancing constraints enabled."
                    << std::endl;
                problem_.inequalityConstraintPtr->add(
                    "contact_forces",
                    get_contact_force_constraint(end_effector_kinematics,
                                                 recompileLibraries));
            }
        } else {
            if (settings_.balancing_settings.constraint_type ==
                ConstraintType::Soft) {
                std::cerr << "Soft ZMP/limit surface-based balancing "
                             "constraints enabled."
                          << std::endl;

                problem_.softConstraintPtr->add(
                    "balancing",
                    get_soft_balancing_constraint(end_effector_kinematics,
                                                  recompileLibraries));

            } else {
                std::cerr << "Hard ZMP/limit surface-based balancing "
                             "constraints enabled."
                          << std::endl;
                problem_.inequalityConstraintPtr->add(
                    "balancing",
                    get_balancing_constraint(end_effector_kinematics,
                                             recompileLibraries));
            }
        }
    } else {
        std::cerr << "Balancing constraints disabled." << std::endl;
    }

    // Initialization state
    if (settings_.use_operating_points) {
        initializerPtr_.reset(new ocs2::OperatingPoints(
            settings_.operating_times, settings_.operating_states,
            settings_.operating_inputs));
    } else {
        initializerPtr_.reset(new ocs2::DefaultInitializer(settings_.dims.u()));
    }

    // referenceManagerPtr_->setTargetTrajectories(settings_.target_trajectory);

    initialState_ = settings_.initial_state;
    std::cerr << "Initial State:   " << initialState_.transpose() << std::endl;
}

std::unique_ptr<ocs2::MPC_BASE> ControllerInterface::getMpc() {
    if (settings_.solver_method == ControllerSettings::SolverMethod::DDP) {
        return std::unique_ptr<ocs2::MPC_BASE>(new ocs2::GaussNewtonDDP_MPC(
            mpcSettings_, ddpSettings_, *rolloutPtr_, problem_,
            *initializerPtr_));
    } else {
        return std::unique_ptr<ocs2::MPC_BASE>(new ocs2::MultipleShootingMpc(
            mpcSettings_, sqpSettings_, problem_, *initializerPtr_));
    }
}

std::unique_ptr<ocs2::StateInputCost>
ControllerInterface::getQuadraticStateInputCost(const std::string& taskFile) {
    MatXd Q = settings_.state_weight;
    MatXd R = settings_.input_weight;

    // augment R with cost on the contact forces
    MatXd Rf = settings_.balancing_settings.force_weight *
               MatXd::Identity(settings_.dims.u(), settings_.dims.u());
    Rf.topLeftCorner(R.rows(), R.cols()) = R;

    std::cout << "Q: " << Q << std::endl;
    std::cout << "Rf: " << Rf << std::endl;

    return std::unique_ptr<ocs2::StateInputCost>(
        new QuadraticJointStateInputCost(Q, Rf));
}

std::unique_ptr<ocs2::StateCost>
ControllerInterface::getDynamicObstacleConstraint(
    ocs2::PinocchioInterface pinocchioInterface,
    const DynamicObstacleSettings& settings, bool usePreComputation,
    const std::string& libraryFolder, bool recompileLibraries) {
    // Add collision spheres to the Pinocchio model
    ocs2::PinocchioInterface::Model model = pinocchioInterface.getModel();
    Mat3d R = Mat3d::Identity();
    for (const auto& sphere : settings.collision_spheres) {
        pinocchio::FrameIndex parent_frame_id =
            model.getFrameId(sphere.parent_frame_name);
        pinocchio::Frame parent_frame = model.frames[parent_frame_id];
        pinocchio::JointIndex parent_joint_id = parent_frame.parent;

        pinocchio::SE3 T_jf = parent_frame.placement;
        pinocchio::SE3 T_fs(R, sphere.offset);
        pinocchio::SE3 T_js = T_jf * T_fs;  // sphere relative to joint

        std::cerr << sphere.name << std::endl;
        std::cerr << sphere.parent_frame_name << std::endl;
        std::cerr << "parent frame id = " << parent_frame_id << std::endl;

        std::cerr << "T_jf = " << T_jf << std::endl;
        std::cerr << "T_fs = " << T_fs << std::endl;
        std::cerr << "T_js = " << T_js << std::endl;

        model.addBodyFrame(sphere.name, parent_joint_id, T_js);
    }

    // Re-initialize interface with the updated model.
    pinocchioInterface = ocs2::PinocchioInterface(model);

    std::unique_ptr<ocs2::PinocchioStateInputMapping<ocs2::ad_scalar_t>>
        pinocchio_mapping_ptr;
    if (settings_.robot_base_type == RobotBaseType::Omnidirectional) {
        pinocchio_mapping_ptr.reset(
            new OmnidirectionalPinocchioMapping<ocs2::ad_scalar_t>(
                settings_.dims.robot(0)));
    } else {
        pinocchio_mapping_ptr.reset(
            new FixedBasePinocchioMapping<ocs2::ad_scalar_t>(
                settings_.dims.robot(0)));
    }

    ocs2::PinocchioEndEffectorKinematicsCppAd object_ee_kinematics(
        pinocchioInterface, *pinocchio_mapping_ptr,
        settings.get_collision_frame_names(), settings_.dims.x(),
        settings_.dims.u(), "obstacle_ee_kinematics", libraryFolder,
        recompileLibraries, false);

    std::unique_ptr<ocs2::StateConstraint> constraint(
        new DynamicObstacleConstraint(object_ee_kinematics,
                                      *referenceManagerPtr_, settings));

    std::unique_ptr<ocs2::PenaltyBase> penalty(
        new ocs2::RelaxedBarrierPenalty({settings.mu, settings.delta}));

    return std::unique_ptr<ocs2::StateCost>(new ocs2::StateSoftConstraint(
        std::move(constraint), std::move(penalty)));
}

std::unique_ptr<ocs2::StateCost> ControllerInterface::getEndEffectorCost(
    const ocs2::PinocchioEndEffectorKinematicsCppAd& end_effector_kinematics) {
    MatXd W = settings_.end_effector_weight;
    std::cout << "W: " << W << std::endl;

    return std::unique_ptr<ocs2::StateCost>(new EndEffectorCost(
        std::move(W), end_effector_kinematics, *referenceManagerPtr_));
}

std::unique_ptr<ocs2::StateInputConstraint>
ControllerInterface::get_balancing_constraint(
    const ocs2::PinocchioEndEffectorKinematicsCppAd& end_effector_kinematics,
    bool recompileLibraries) {
    return std::unique_ptr<ocs2::StateInputConstraint>(
        new BoundedBalancingConstraints(
            end_effector_kinematics, settings_.balancing_settings,
            settings_.gravity, settings_.dims, recompileLibraries));
}

std::unique_ptr<ocs2::StateInputCost>
ControllerInterface::get_soft_balancing_constraint(
    const ocs2::PinocchioEndEffectorKinematicsCppAd& end_effector_kinematics,
    bool recompileLibraries) {
    // compute the hard constraint
    std::unique_ptr<ocs2::StateInputConstraint> constraint =
        get_balancing_constraint(end_effector_kinematics, recompileLibraries);

    // make it soft via penalty function
    std::vector<std::unique_ptr<ocs2::PenaltyBase>> penaltyArray(
        constraint->getNumConstraints(0));
    for (int i = 0; i < constraint->getNumConstraints(0); i++) {
        penaltyArray[i].reset(new ocs2::RelaxedBarrierPenalty(
            {settings_.balancing_settings.mu,
             settings_.balancing_settings.delta}));
    }

    return std::unique_ptr<ocs2::StateInputCost>(
        new ocs2::StateInputSoftConstraint(std::move(constraint),
                                           std::move(penaltyArray)));
}

std::unique_ptr<ocs2::StateInputConstraint>
ControllerInterface::get_object_dynamics_constraint(
    const ocs2::PinocchioEndEffectorKinematicsCppAd& end_effector_kinematics,
    bool recompileLibraries) {
    return std::unique_ptr<ocs2::StateInputConstraint>(
        new ObjectDynamicsConstraints(
            end_effector_kinematics, settings_.balancing_settings,
            settings_.gravity, settings_.dims, recompileLibraries));
}

std::unique_ptr<ocs2::StateInputCost>
ControllerInterface::get_soft_object_dynamics_constraint(
    const ocs2::PinocchioEndEffectorKinematicsCppAd& end_effector_kinematics,
    bool recompileLibraries) {
    // compute the hard constraint
    std::unique_ptr<ocs2::StateInputConstraint> constraint =
        get_object_dynamics_constraint(end_effector_kinematics,
                                       recompileLibraries);

    // make it soft via penalty function: since this is an equality constraint,
    // we use a quadratic penalty
    // TODO may need to increase the scaling
    std::vector<std::unique_ptr<ocs2::PenaltyBase>> penaltyArray(
        constraint->getNumConstraints(0));
    for (int i = 0; i < constraint->getNumConstraints(0); i++) {
        penaltyArray[i].reset(
            new ocs2::QuadraticPenalty(settings_.balancing_settings.mu));
    }

    return std::unique_ptr<ocs2::StateInputCost>(
        new ocs2::StateInputSoftConstraint(std::move(constraint),
                                           std::move(penaltyArray)));
}

std::unique_ptr<ocs2::StateInputConstraint>
ControllerInterface::get_contact_force_constraint(
    const ocs2::PinocchioEndEffectorKinematicsCppAd& end_effector_kinematics,
    bool recompileLibraries) {
    return std::unique_ptr<ocs2::StateInputConstraint>(
        new ContactForceBalancingConstraints(
            end_effector_kinematics, settings_.balancing_settings,
            settings_.gravity, settings_.dims, recompileLibraries));
}

std::unique_ptr<ocs2::StateInputCost>
ControllerInterface::get_soft_contact_force_constraint(
    const ocs2::PinocchioEndEffectorKinematicsCppAd& end_effector_kinematics,
    bool recompileLibraries) {
    // compute the hard constraint
    std::unique_ptr<ocs2::StateInputConstraint> constraint =
        get_contact_force_constraint(end_effector_kinematics,
                                     recompileLibraries);

    // make it soft via penalty function
    std::vector<std::unique_ptr<ocs2::PenaltyBase>> penaltyArray(
        constraint->getNumConstraints(0));
    for (int i = 0; i < constraint->getNumConstraints(0); i++) {
        penaltyArray[i].reset(new ocs2::SquaredHingePenalty(
            {1, settings_.balancing_settings.delta}));  // TODO
    }

    return std::unique_ptr<ocs2::StateInputCost>(
        new ocs2::StateInputSoftConstraint(std::move(constraint),
                                           std::move(penaltyArray)));
}

std::unique_ptr<ocs2::StateConstraint>
ControllerInterface::get_static_obstacle_constraint(
    ocs2::PinocchioInterface& pinocchioInterface,
    const StaticObstacleSettings& settings,
    const std::string& obstacle_urdf_path, bool usePreComputation,
    const std::string& libraryFolder, bool recompileLibraries) {
    // only specifying link pairs (i.e. by name)
    ocs2::PinocchioGeometryInterface geometryInterface(pinocchioInterface);

    // Add obstacle collision objects to the geometry model, so we can check
    // them against the robot.
    pinocchio::GeometryModel obs_geom_model =
        build_geometry_model(obstacle_urdf_path);
    geometryInterface.addGeometryObjects(obs_geom_model);

    const auto& model = pinocchioInterface.getModel();
    auto& data = pinocchioInterface.getData();
    // const auto q = pinocchioMapping_.getPinocchioJointPosition(x);
    // pinocchio::forwardKinematics(model, data, q);
    // pinocchio::updateFramePlacements(model, data);

    Mat3d R = Mat3d::Identity();

    std::cerr << "Number of extra collision spheres = "
              << settings.extra_spheres.size() << std::endl;

    std::vector<pinocchio::GeometryObject> extra_spheres;
    for (const auto& sphere : settings.extra_spheres) {
        // The collision sphere is specified relative to a link, but the
        // geometry interface works relative to joints. Thus we need to find
        // the parent joint and the sphere's transform w.r.t. it.
        // TODO if possible, it would be better to specify w.r.t. to a joint
        pinocchio::FrameIndex parent_frame_id =
            model.getFrameId(sphere.parent_frame_name);
        pinocchio::Frame parent_frame = model.frames[parent_frame_id];

        pinocchio::JointIndex parent_joint_id = parent_frame.parent;
        pinocchio::SE3 T_jf = parent_frame.placement;
        pinocchio::SE3 T_fs(R, sphere.offset);
        pinocchio::SE3 T_js = T_jf * T_fs;  // sphere relative to joint

        pinocchio::GeometryObject::CollisionGeometryPtr geom_ptr(
            new hpp::fcl::Sphere(sphere.radius));
        extra_spheres.push_back(pinocchio::GeometryObject(
            sphere.name, parent_joint_id, geom_ptr, T_js));
    }
    geometryInterface.addGeometryObjects(extra_spheres);

    pinocchio::GeometryModel& geometry_model =
        geometryInterface.getGeometryModel();
    for (int i = 0; i < geometry_model.ngeoms; ++i) {
        std::cout << geometry_model.geometryObjects[i].name << std::endl;
    }

    geometryInterface.addCollisionPairsByName(settings.collision_link_pairs);

    const size_t numCollisionPairs = geometryInterface.getNumCollisionPairs();
    std::cerr << "SelfCollision: Testing for " << numCollisionPairs
              << " collision pairs\n";

    std::vector<hpp::fcl::DistanceResult> distances =
        geometryInterface.computeDistances(pinocchioInterface);
    for (int i = 0; i < distances.size(); ++i) {
        std::cout << "dist = " << distances[i].min_distance << std::endl;
    }

    std::unique_ptr<ocs2::PinocchioStateInputMapping<ocs2::scalar_t>>
        pinocchio_mapping_ptr;
    if (settings_.robot_base_type == RobotBaseType::Omnidirectional) {
        pinocchio_mapping_ptr.reset(
            new OmnidirectionalPinocchioMapping<ocs2::scalar_t>(
                settings_.dims.robot(0)));
    } else {
        pinocchio_mapping_ptr.reset(
            new FixedBasePinocchioMapping<ocs2::scalar_t>(
                settings_.dims.robot(0)));
    }

    return std::unique_ptr<ocs2::StateConstraint>(
        new ocs2::SelfCollisionConstraintCppAd(
            pinocchioInterface, *pinocchio_mapping_ptr,
            std::move(geometryInterface), settings.minimum_distance,
            "static_obstacle_avoidance", libraryFolder, recompileLibraries,
            false));
    // TODO need precomp for this apparently
    // return std::unique_ptr<ocs2::StateConstraint>(
    //     new ocs2::SelfCollisionConstraint(*pinocchio_mapping_ptr,
    //                                            std::move(geometryInterface),
    //                                            settings.minimum_distance));
}

std::unique_ptr<ocs2::StateCost>
ControllerInterface::get_soft_static_obstacle_constraint(
    ocs2::PinocchioInterface pinocchioInterface,
    const StaticObstacleSettings& settings,
    const std::string& obstacle_urdf_path, bool usePreComputation,
    const std::string& libraryFolder, bool recompileLibraries) {
    std::unique_ptr<ocs2::StateConstraint> constraint =
        get_static_obstacle_constraint(pinocchioInterface, settings,
                                       obstacle_urdf_path, usePreComputation,
                                       libraryFolder, recompileLibraries);

    std::unique_ptr<ocs2::PenaltyBase> penalty(
        new ocs2::RelaxedBarrierPenalty({settings.mu, settings.delta}));

    return std::unique_ptr<ocs2::StateCost>(new ocs2::StateSoftConstraint(
        std::move(constraint), std::move(penalty)));
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

std::unique_ptr<ocs2::StateInputCost>
ControllerInterface::get_soft_joint_state_input_limit_constraint() {
    std::unique_ptr<ocs2::StateInputConstraint> constraint =
        get_joint_state_input_limit_constraint();

    VecXd state_limit_lower = settings_.state_limit_lower;
    VecXd state_limit_upper = settings_.state_limit_upper;
    ocs2::scalar_t state_limit_mu = settings_.state_limit_mu;
    ocs2::scalar_t state_limit_delta = settings_.state_limit_delta;

    VecXd input_limit_lower = settings_.input_limit_lower;
    VecXd input_limit_upper = settings_.input_limit_upper;
    ocs2::scalar_t input_limit_mu = settings_.input_limit_mu;
    ocs2::scalar_t input_limit_delta = settings_.input_limit_delta;

    auto num_constraints = constraint->getNumConstraints(0);
    std::unique_ptr<ocs2::PenaltyBase> barrierFunction;
    std::vector<std::unique_ptr<ocs2::PenaltyBase>> penaltyArray(
        num_constraints);

    // State penalty
    for (int i = 0; i < settings_.dims.robot(0).x; i++) {
        // barrierFunction.reset(new ocs2::RelaxedBarrierPenalty(
        //     {state_limit_mu, state_limit_delta}));
        barrierFunction.reset(
            new ocs2::SquaredHingePenalty({1, state_limit_delta}));
        penaltyArray[i].reset(new ocs2::DoubleSidedPenalty(
            state_limit_lower(i), state_limit_upper(i),
            std::move(barrierFunction)));
    }

    // Input penalty
    for (int i = 0; i < settings_.dims.robot(0).u; i++) {
        // barrierFunction.reset(new ocs2::RelaxedBarrierPenalty(
        //     {input_limit_mu, input_limit_delta}));
        barrierFunction.reset(
            new ocs2::SquaredHingePenalty({1, input_limit_delta}));
        penaltyArray[settings_.dims.robot(0).x + i].reset(
            new ocs2::DoubleSidedPenalty(input_limit_lower(i),
                                         input_limit_upper(i),
                                         std::move(barrierFunction)));
    }

    return std::unique_ptr<ocs2::StateInputCost>(
        new ocs2::StateInputSoftConstraint(std::move(constraint),
                                           std::move(penaltyArray)));
}

}  // namespace upright
