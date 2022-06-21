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
#include <ocs2_core/soft_constraint/StateInputSoftConstraint.h>
#include <ocs2_core/soft_constraint/StateSoftConstraint.h>
#include <ocs2_core/soft_constraint/penalties/DoubleSidedPenalty.h>
#include <ocs2_core/soft_constraint/penalties/QuadraticPenalty.h>
#include <ocs2_core/soft_constraint/penalties/RelaxedBarrierPenalty.h>
#include <ocs2_core/soft_constraint/penalties/SquaredHingePenalty.h>
#include <ocs2_oc/synchronized_module/ReferenceManager.h>
#include <ocs2_pinocchio_interface/PinocchioEndEffectorKinematics.h>
#include <ocs2_pinocchio_interface/PinocchioEndEffectorKinematicsCppAd.h>
#include <ocs2_pinocchio_interface/urdf.h>
#include <ocs2_self_collision/SelfCollisionConstraint.h>
#include <ocs2_self_collision/SelfCollisionConstraintCppAd.h>
#include <ocs2_self_collision/loadStdVectorOfPair.h>
#include <ocs2_sqp/MultipleShootingMpc.h>
#include <ocs2_sqp/MultipleShootingSettings.h>

#include <upright_control/controller_interface.h>
#include <upright_control/constraint/JointStateInputLimits.h>
#include <upright_control/constraint/ObstacleConstraint.h>
#include <upright_control/cost/EndEffectorCost.h>
#include <upright_control/cost/InertialAlignmentCost.h>
#include <upright_control/cost/QuadraticJointStateInputCost.h>
#include <upright_control/dynamics/BaseType.h>
#include <upright_control/dynamics/FixedBaseDynamics.h>
#include <upright_control/dynamics/FixedBasePinocchioMapping.h>
#include <upright_control/dynamics/MobileManipulatorDynamics.h>
#include <upright_control/dynamics/MobileManipulatorPinocchioMapping.h>
#include <upright_control/util.h>

#include <upright_control/constraint/BoundedBalancingConstraints.h>

#include <ros/package.h>

namespace upright {

ControllerInterface::ControllerInterface(
    const ControllerSettings& settings)
    : settings_(settings) {
    // load setting from config file
    loadSettings();
}

ocs2::PinocchioInterface ControllerInterface::buildPinocchioInterface(
    const std::string& urdfPath, const std::string& obstacle_urdfPath) {
    if (settings_.robot_base_type == RobotBaseType::Omnidirectional) {
        // add 3 DOF for wheelbase
        pinocchio::JointModelComposite rootJoint(3);
        rootJoint.addJoint(pinocchio::JointModelPX());
        rootJoint.addJoint(pinocchio::JointModelPY());
        rootJoint.addJoint(pinocchio::JointModelRZ());

        return ocs2::getPinocchioInterfaceFromUrdfFile(urdfPath, rootJoint);
    }
    // Fixed base
    return ocs2::getPinocchioInterfaceFromUrdfFile(urdfPath);
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

    pinocchioInterfacePtr_.reset(
        new ocs2::PinocchioInterface(buildPinocchioInterface(
            settings_.robot_urdf_path, settings_.obstacle_urdf_path)));
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

    // Dynamics
    std::unique_ptr<ocs2::SystemDynamicsBase> dynamicsPtr;
    if (settings_.robot_base_type == RobotBaseType::Omnidirectional) {
        dynamicsPtr.reset(new MobileManipulatorDynamics(
            "robot_dynamics", settings_.dims, libraryFolder, recompileLibraries,
            true));
    } else {
        dynamicsPtr.reset(new FixedBaseDynamics("robot_dynamics",
                                                settings_.dims, libraryFolder,
                                                recompileLibraries, true));
    }

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
    problem_.costPtr->add("stateInputCost",
                          getQuadraticStateInputCost(taskFile));

    // Build the end effector kinematics
    std::unique_ptr<ocs2::PinocchioStateInputMapping<ocs2::ad_scalar_t>>
        pinocchio_mapping_ptr;
    if (settings_.robot_base_type == RobotBaseType::Omnidirectional) {
        pinocchio_mapping_ptr.reset(
            new MobileManipulatorPinocchioMapping<ocs2::ad_scalar_t>(
                settings_.dims));
    } else {
        pinocchio_mapping_ptr.reset(
            new FixedBasePinocchioMapping<ocs2::ad_scalar_t>(settings_.dims));
    }

    ocs2::PinocchioEndEffectorKinematicsCppAd end_effector_kinematics(
        *pinocchioInterfacePtr_, *pinocchio_mapping_ptr,
        {settings_.end_effector_link_name}, settings_.dims.x, settings_.dims.u,
        "end_effector_kinematics", libraryFolder, recompileLibraries, false);

    problem_.stateCostPtr->add("endEffector",
                               getEndEffectorCost(end_effector_kinematics));

    if (settings_.inertial_alignment_settings.enabled) {
        std::unique_ptr<ocs2::StateInputCost> inertial_alignment_cost(
            new InertialAlignmentCost(end_effector_kinematics,
                                      settings_.inertial_alignment_settings,
                                      settings_.gravity, settings_.dims, true));
        problem_.costPtr->add("inertial_alignment_cost",
                              std::move(inertial_alignment_cost));
    }

    // std::unique_ptr<StateConstraint> inertial_alignment_constraint(
    //     new InertialAlignmentConstraint(end_effector_kinematics,
    //     settings_.dims,
    //                                     true));
    // problem_.stateEqualityConstraintPtr->add(
    //     "inertial_alignment_constraint",
    //     std::move(inertial_alignment_constraint));

    /* Constraints */
    problem_.softConstraintPtr->add(
        "jointStateInputLimits", getJointStateInputLimitConstraint(taskFile));

    // Self-collision avoidance and collision avoidance with static obstacles
    if (settings_.collision_avoidance_settings.enabled) {
        std::cerr << "Collision avoidance is enabled." << std::endl;
        problem_.stateSoftConstraintPtr->add(
            "collisionAvoidance",
            getCollisionAvoidanceConstraint(
                *pinocchioInterfacePtr_, settings_.collision_avoidance_settings,
                settings_.obstacle_urdf_path, usePreComputation, libraryFolder,
                recompileLibraries));
    } else {
        std::cerr << "Collision avoidance is disabled." << std::endl;
    }

    // Collision avoidance with dynamic obstacles specified via the reference
    // trajectory
    if (settings_.dynamic_obstacle_settings.enabled) {
        std::cerr << "Dynamic obstacle avoidance is enabled." << std::endl;
        problem_.stateSoftConstraintPtr->add(
            "dynamicObstacleAvoidance",
            getDynamicObstacleConstraint(
                *pinocchioInterfacePtr_, settings_.dynamic_obstacle_settings,
                usePreComputation, libraryFolder, recompileLibraries));
    } else {
        std::cerr << "Dynamic obstacle avoidance is disabled." << std::endl;
    }

    if (settings_.tray_balance_settings.enabled) {
        if (settings_.tray_balance_settings.constraint_type ==
            ConstraintType::Soft) {
            std::cerr << "Soft tray balance constraints enabled." << std::endl;
            problem_.softConstraintPtr->add(
                "trayBalance",
                getTrayBalanceSoftConstraint(end_effector_kinematics,
                                             recompileLibraries));

        } else {
            // TODO: hard inequality constraints not currently implemented by
            // OCS2
            std::cerr << "Hard tray balance constraints enabled." << std::endl;
            problem_.inequalityConstraintPtr->add(
                "trayBalance", getTrayBalanceConstraint(end_effector_kinematics,
                                                        recompileLibraries));
        }
    } else {
        std::cerr << "Tray balance constraints disabled." << std::endl;
    }

    // Initialization state
    if (settings_.use_operating_points) {
        initializerPtr_.reset(new ocs2::OperatingPoints(
            settings_.operating_times, settings_.operating_states,
            settings_.operating_inputs));
    } else {
        initializerPtr_.reset(new ocs2::DefaultInitializer(settings_.dims.u));
    }

    // referenceManagerPtr_->setTargetTrajectories(settings_.target_trajectory);

    initialState_ = settings_.initial_state;
    std::cerr << "Initial State:   " << initialState_.transpose() << std::endl;
}

std::unique_ptr<ocs2::MPC_BASE> ControllerInterface::getMpc() {
    if (settings_.solver_method == ControllerSettings::SolverMethod::DDP) {
        return std::unique_ptr<ocs2::MPC_BASE>(
            new ocs2::MPC_DDP(mpcSettings_, ddpSettings_, *rolloutPtr_,
                              problem_, *initializerPtr_));
    } else {
        return std::unique_ptr<ocs2::MPC_BASE>(new ocs2::MultipleShootingMpc(
            mpcSettings_, sqpSettings_, problem_, *initializerPtr_));
    }
}

std::unique_ptr<ocs2::StateInputCost>
ControllerInterface::getQuadraticStateInputCost(
    const std::string& taskFile) {
    MatXd Q = settings_.state_weight;
    MatXd R = settings_.input_weight;

    std::cout << "Q: " << Q << std::endl;
    std::cout << "R: " << R << std::endl;

    return std::unique_ptr<ocs2::StateInputCost>(
        new QuadraticJointStateInputCost(std::move(Q), std::move(R)));
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
            new MobileManipulatorPinocchioMapping<ocs2::ad_scalar_t>(
                settings_.dims));
    } else {
        pinocchio_mapping_ptr.reset(
            new FixedBasePinocchioMapping<ocs2::ad_scalar_t>(settings_.dims));
    }

    ocs2::PinocchioEndEffectorKinematicsCppAd eeKinematics(
        pinocchioInterface, *pinocchio_mapping_ptr,
        settings.get_collision_frame_names(), settings_.dims.x,
        settings_.dims.u, "obstacle_ee_kinematics", libraryFolder,
        recompileLibraries, false);

    std::unique_ptr<ocs2::StateConstraint> constraint(
        new DynamicObstacleConstraint(eeKinematics, *referenceManagerPtr_,
                                      settings));

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
ControllerInterface::getTrayBalanceConstraint(
    const ocs2::PinocchioEndEffectorKinematicsCppAd& end_effector_kinematics,
    bool recompileLibraries) {
    return std::unique_ptr<ocs2::StateInputConstraint>(
        new BoundedBalancingConstraints(
            end_effector_kinematics, settings_.tray_balance_settings,
            settings_.gravity, settings_.dims, recompileLibraries));
}

std::unique_ptr<ocs2::StateInputCost>
ControllerInterface::getTrayBalanceSoftConstraint(
    const ocs2::PinocchioEndEffectorKinematicsCppAd& end_effector_kinematics,
    bool recompileLibraries) {
    // compute the hard constraint
    std::unique_ptr<ocs2::StateInputConstraint> constraint =
        getTrayBalanceConstraint(end_effector_kinematics, recompileLibraries);

    // make it soft via penalty function
    std::vector<std::unique_ptr<ocs2::PenaltyBase>> penaltyArray(
        constraint->getNumConstraints(0));
    for (int i = 0; i < constraint->getNumConstraints(0); i++) {
        penaltyArray[i].reset(
            // new SquaredHingePenalty({settings.mu, settings.delta}));
            new ocs2::RelaxedBarrierPenalty(
                {settings_.tray_balance_settings.mu,
                 settings_.tray_balance_settings.delta}));
    }

    return std::unique_ptr<ocs2::StateInputCost>(
        new ocs2::StateInputSoftConstraint(std::move(constraint),
                                           std::move(penaltyArray)));
}

std::unique_ptr<ocs2::StateCost>
ControllerInterface::getCollisionAvoidanceConstraint(
    ocs2::PinocchioInterface pinocchioInterface,
    const CollisionAvoidanceSettings& settings,
    const std::string& obstacle_urdf_path, bool usePreComputation,
    const std::string& libraryFolder, bool recompileLibraries) {
    // only specifying link pairs (i.e. by name)
    ocs2::PinocchioGeometryInterface geometryInterface(pinocchioInterface);

    // Add obstacle collision objects to the geometry model, so we can check
    // them against the robot.
    pinocchio::GeometryModel obs_geom_model =
        build_geometry_model(obstacle_urdf_path);
    geometryInterface.addGeometryObjects(obs_geom_model);

    ocs2::PinocchioInterface::Model model = pinocchioInterface.getModel();
    Mat3d R = Mat3d::Identity();

    std::cerr << "Number of extra collision spheres = "
              << settings.extra_spheres.size() << std::endl;

    std::vector<pinocchio::GeometryObject> extra_spheres;
    for (const auto& sphere : settings.extra_spheres) {
        // The collision sphere is specified relative to a link, but the
        // geometry interface works relative to joints. Thus we need to find
        // the parent joint and the sphere's transform w.r.t. it.
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

    // std::vector<hpp::fcl::DistanceResult> distances =
    //     geometryInterface.computeDistances(pinocchioInterface);
    // for (int i = 0; i < distances.size(); ++i) {
    //     std::cout << "dist = " << distances[i].min_distance << std::endl;
    // }

    std::unique_ptr<ocs2::PinocchioStateInputMapping<ocs2::scalar_t>>
        pinocchio_mapping_ptr;
    if (settings_.robot_base_type == RobotBaseType::Omnidirectional) {
        pinocchio_mapping_ptr.reset(
            new MobileManipulatorPinocchioMapping<ocs2::scalar_t>(
                settings_.dims));
    } else {
        pinocchio_mapping_ptr.reset(
            new FixedBasePinocchioMapping<ocs2::scalar_t>(settings_.dims));
    }

    std::unique_ptr<ocs2::StateConstraint> constraint(
        new ocs2::SelfCollisionConstraintCppAd(
            pinocchioInterface, *pinocchio_mapping_ptr,
            std::move(geometryInterface), settings.minimum_distance,
            "self_collision", libraryFolder, recompileLibraries, false));

    std::unique_ptr<ocs2::PenaltyBase> penalty(
        new ocs2::RelaxedBarrierPenalty({settings.mu, settings.delta}));

    return std::unique_ptr<ocs2::StateCost>(new ocs2::StateSoftConstraint(
        std::move(constraint), std::move(penalty)));
}

std::unique_ptr<ocs2::StateInputCost>
ControllerInterface::getJointStateInputLimitConstraint(
    const std::string& taskFile) {
    VecXd state_limit_lower = settings_.state_limit_lower;
    VecXd state_limit_upper = settings_.state_limit_upper;
    ocs2::scalar_t state_limit_mu = settings_.state_limit_mu;
    ocs2::scalar_t state_limit_delta = settings_.state_limit_delta;

    VecXd input_limit_lower = settings_.input_limit_lower;
    VecXd input_limit_upper = settings_.input_limit_upper;
    ocs2::scalar_t input_limit_mu = settings_.input_limit_mu;
    ocs2::scalar_t input_limit_delta = settings_.input_limit_delta;

    std::cout << "state limit lower: " << state_limit_lower.transpose()
              << std::endl;
    std::cout << "state limit upper: " << state_limit_upper.transpose()
              << std::endl;
    std::cout << "input limit lower: " << input_limit_lower.transpose()
              << std::endl;
    std::cout << "input limit upper: " << input_limit_upper.transpose()
              << std::endl;

    std::unique_ptr<ocs2::StateInputConstraint> constraint(
        new JointStateInputLimits(settings_.dims));
    auto num_constraints = constraint->getNumConstraints(0);
    std::unique_ptr<ocs2::PenaltyBase> barrierFunction;
    std::vector<std::unique_ptr<ocs2::PenaltyBase>> penaltyArray(
        num_constraints);

    // State penalty
    for (int i = 0; i < settings_.dims.x; i++) {
        barrierFunction.reset(new ocs2::RelaxedBarrierPenalty(
            {state_limit_mu, state_limit_delta}));
        penaltyArray[i].reset(new ocs2::DoubleSidedPenalty(
            state_limit_lower(i), state_limit_upper(i),
            std::move(barrierFunction)));
    }

    // Input penalty
    for (int i = 0; i < settings_.dims.u; i++) {
        barrierFunction.reset(new ocs2::RelaxedBarrierPenalty(
            {input_limit_mu, input_limit_delta}));
        penaltyArray[settings_.dims.x + i].reset(new ocs2::DoubleSidedPenalty(
            input_limit_lower(i), input_limit_upper(i),
            std::move(barrierFunction)));
    }

    return std::unique_ptr<ocs2::StateInputCost>(
        new ocs2::StateInputSoftConstraint(std::move(constraint),
                                           std::move(penaltyArray)));
}

}  // namespace upright
