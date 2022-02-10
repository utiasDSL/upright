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

#include <pinocchio/fwd.hpp>
#include <pinocchio/multibody/joint/joint-composite.hpp>
#include <pinocchio/multibody/model.hpp>
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/kinematics.hpp>

#include <ocs2_core/cost/QuadraticStateCost.h>
#include <ocs2_core/initialization/DefaultInitializer.h>
#include <ocs2_core/misc/LoadData.h>
#include <ocs2_core/soft_constraint/StateInputSoftConstraint.h>
#include <ocs2_core/soft_constraint/StateSoftConstraint.h>
#include <ocs2_core/soft_constraint/penalties/DoubleSidedPenalty.h>
#include <ocs2_core/soft_constraint/penalties/QuadraticPenalty.h>
#include <ocs2_core/soft_constraint/penalties/RelaxedBarrierPenalty.h>
#include <ocs2_oc/synchronized_module/ReferenceManager.h>
#include <ocs2_pinocchio_interface/PinocchioEndEffectorKinematics.h>
#include <ocs2_pinocchio_interface/PinocchioEndEffectorKinematicsCppAd.h>
#include <ocs2_pinocchio_interface/urdf.h>
#include <ocs2_self_collision/SelfCollisionConstraint.h>
#include <ocs2_self_collision/SelfCollisionConstraintCppAd.h>
#include <ocs2_self_collision/loadStdVectorOfPair.h>
#include <ocs2_sqp/MultipleShootingMpc.h>
#include <ocs2_sqp/MultipleShootingSettings.h>

#include <tray_balance_ocs2/MobileManipulatorDynamics.h>
#include <tray_balance_ocs2/MobileManipulatorInterface.h>
#include <tray_balance_ocs2/MobileManipulatorPreComputation.h>
#include <tray_balance_ocs2/constraint/CollisionAvoidanceConstraint.h>
#include <tray_balance_ocs2/constraint/EndEffectorConstraint.h>
#include <tray_balance_ocs2/constraint/JointStateInputLimits.h>
#include <tray_balance_ocs2/constraint/ObstacleConstraint.h>
#include <tray_balance_ocs2/cost/EndEffectorCost.h>
#include <tray_balance_ocs2/cost/QuadraticJointStateInputCost.h>
#include <tray_balance_ocs2/cost/ZMPCost.h>
#include <tray_balance_ocs2/definitions.h>
#include <tray_balance_ocs2/util.h>

#include <tray_balance_ocs2/constraint/tray_balance/RobustTrayBalanceConstraints.h>
#include <tray_balance_ocs2/constraint/tray_balance/TrayBalanceConstraints.h>
#include <tray_balance_ocs2/constraint/tray_balance/TrayBalanceSettings.h>

#include <ros/package.h>

namespace ocs2 {
namespace mobile_manipulator {

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
MobileManipulatorInterface::MobileManipulatorInterface(
    const std::string& taskFile, const std::string& libraryFolder,
    const TaskSettings& settings)
    : settings_(settings) {
    std::cerr << "Loading task file: " << taskFile << std::endl;

    // load setting from config file
    loadSettings(taskFile, libraryFolder);
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
PinocchioInterface MobileManipulatorInterface::buildPinocchioInterface(
    const std::string& urdfPath, const std::string& obstacle_urdfPath) {
    // add 3 DOF for wheelbase
    pinocchio::JointModelComposite rootJoint(3);
    rootJoint.addJoint(pinocchio::JointModelPX());
    rootJoint.addJoint(pinocchio::JointModelPY());
    rootJoint.addJoint(pinocchio::JointModelRZ());

    return getPinocchioInterfaceFromUrdfFile(urdfPath, rootJoint);
}

pinocchio::GeometryModel MobileManipulatorInterface::build_geometry_model(
    const std::string& urdf_path) {
    PinocchioInterface::Model model;
    pinocchio::urdf::buildModel(urdf_path, model);
    pinocchio::GeometryModel geom_model;
    pinocchio::urdf::buildGeom(model, urdf_path, pinocchio::COLLISION,
                               geom_model);
    return geom_model;
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
void MobileManipulatorInterface::loadSettings(
    const std::string& taskFile, const std::string& libraryFolder) {
    std::cerr << "taskFile = " << taskFile << std::endl;
    boost::property_tree::ptree pt;
    boost::property_tree::read_info(taskFile, pt);

    /*
     * URDF files
     */
    std::string robot_urdf_path, obstacle_urdf_path;
    std::tie(robot_urdf_path, obstacle_urdf_path) = load_urdf_paths(taskFile);
    std::cerr << "Robot URDF: " << robot_urdf_path << std::endl;
    std::cerr << "Obstacle URDF: " << obstacle_urdf_path << std::endl;

    pinocchioInterfacePtr_.reset(new PinocchioInterface(
        buildPinocchioInterface(robot_urdf_path, obstacle_urdf_path)));
    std::cerr << *pinocchioInterfacePtr_;

    /*
     * Model settings
     */
    bool usePreComputation = true;
    bool recompileLibraries = true;
    std::cerr << "\n #### Model Settings:";
    std::cerr << "\n #### "
                 "============================================================="
                 "================\n";
    loadData::loadPtreeValue(pt, usePreComputation,
                             "model_settings.usePreComputation", true);
    loadData::loadPtreeValue(pt, recompileLibraries,
                             "model_settings.recompileLibraries", true);
    std::cerr << " #### "
                 "============================================================="
                 "================"
              << std::endl;

    /*
     * DDP-MPC settings
     */
    ddpSettings_ = ddp::loadSettings(taskFile, "ddp");
    mpcSettings_ = mpc::loadSettings(taskFile, "mpc");
    sqpSettings_ =
        multiple_shooting::loadSettings(taskFile, "multiple_shooting");

    /*
     * Dynamics
     */
    std::unique_ptr<MobileManipulatorDynamics> dynamicsPtr(
        new MobileManipulatorDynamics("mobile_manipulator_dynamics",
                                      libraryFolder, recompileLibraries, true));

    /*
     * Rollout
     */
    const auto rolloutSettings = rollout::loadSettings(taskFile, "rollout");
    rolloutPtr_.reset(new TimeTriggeredRollout(*dynamicsPtr, rolloutSettings));

    /*
     * Reference manager
     */
    referenceManagerPtr_.reset(new ReferenceManager);

    /*
     * Optimal control problem
     */
    problem_.dynamicsPtr = std::move(dynamicsPtr);

    /* Cost */
    problem_.costPtr->add("stateInputCost",
                          getQuadraticStateInputCost(taskFile));

    // ZMP cost
    // problem_.costPtr->add(
    //     "zmpCost",
    //     get_zmp_cost(*pinocchioInterfacePtr_, taskFile, "zmpCost",
    //                  usePreComputation, libraryFolder, recompileLibraries));

    // TODO do we need a final cost on state/input?
    // matrix_t Qf = matrix_t::Zero(STATE_DIM, STATE_DIM);
    // Qf.block<INPUT_DIM, INPUT_DIM>(INPUT_DIM, INPUT_DIM) =
    //     0.1 * matrix_t::Identity(INPUT_DIM, INPUT_DIM);
    // matrix_t Rf = 0.1 * matrix_t::Identity(INPUT_DIM, INPUT_DIM);
    // loadData::loadEigenMatrix(taskFile, "stateCost.Q", Qf);
    // problem_.finalCostPtr->add(
    //     "finalCost", std::unique_ptr<StateCost>(new QuadraticStateCost(Qf)));

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
                obstacle_urdf_path, usePreComputation, libraryFolder,
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
                getTrayBalanceSoftConstraint(
                    *pinocchioInterfacePtr_, settings_.tray_balance_settings,
                    usePreComputation, libraryFolder, recompileLibraries));

        } else {
            // TODO: hard inequality constraints do not appear to be
            // implemented by OCS2
            std::cerr << "Hard tray balance constraints enabled." << std::endl;
            problem_.inequalityConstraintPtr->add(
                "trayBalance",
                getTrayBalanceConstraint(
                    *pinocchioInterfacePtr_, settings_.tray_balance_settings,
                    usePreComputation, libraryFolder, recompileLibraries));
        }
    } else {
        std::cerr << "Tray balance constraints disabled." << std::endl;
    }

    // Alternative EE pose matching formulated as a (soft) constraint
    // problem_.stateSoftConstraintPtr->add(
    //     "endEffector",
    //     getEndEffectorConstraint(*pinocchioInterfacePtr_, taskFile,
    //                              "endEffector", usePreComputation,
    //                              libraryFolder, recompileLibraries));
    // problem_.finalSoftConstraintPtr->add(
    //     "finalEndEffector",
    //     getEndEffectorConstraint(*pinocchioInterfacePtr_, taskFile,
    //                              "finalEndEffector", usePreComputation,
    //                              libraryFolder, recompileLibraries));

    problem_.stateCostPtr->add(
        "endEffector", getEndEffectorCost(*pinocchioInterfacePtr_, taskFile,
                                          "endEffector", usePreComputation,
                                          libraryFolder, recompileLibraries));
    // problem_.finalCostPtr->add(
    //     "finalEndEffector",
    //     getEndEffectorCost(*pinocchioInterfacePtr_, taskFile,
    //                        "finalEndEffector", usePreComputation,
    //                        libraryFolder, recompileLibraries));

    /*
     * Use pre-computation
     */
    if (usePreComputation) {
        problem_.preComputationPtr.reset(
            new MobileManipulatorPreComputation(*pinocchioInterfacePtr_));
    }

    /*
     * Initialization state
     */
    initializerPtr_.reset(new DefaultInitializer(INPUT_DIM));

    loadData::loadEigenMatrix(taskFile, "initialState", initialState_);
    std::cerr << "Initial State:   " << initialState_.transpose() << std::endl;
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
std::unique_ptr<MPC_BASE> MobileManipulatorInterface::getMpc() {
    if (settings_.method == TaskSettings::Method::DDP) {
        return std::unique_ptr<MPC_BASE>(new MPC_DDP(mpcSettings_, ddpSettings_,
                                                     *rolloutPtr_, problem_,
                                                     *initializerPtr_));
    } else {
        return std::unique_ptr<MPC_BASE>(new MultipleShootingMpc(
            mpcSettings_, sqpSettings_, problem_, *initializerPtr_));
    }
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
std::unique_ptr<StateInputCost>
MobileManipulatorInterface::getQuadraticStateInputCost(
    const std::string& taskFile) {
    matrix_t Q(STATE_DIM, STATE_DIM);
    matrix_t R(INPUT_DIM, INPUT_DIM);

    std::cerr << "\n #### Input Cost Settings: ";
    std::cerr << "\n #### "
                 "============================================================="
                 "================\n";
    loadData::loadEigenMatrix(taskFile, "inputCost.R", R);
    loadData::loadEigenMatrix(taskFile, "stateCost.Q", Q);
    std::cerr << "stateCost.Q:  \n" << Q << '\n';
    std::cerr << "inputCost.R:  \n" << R << '\n';
    std::cerr << " #### "
                 "============================================================="
                 "================"
              << std::endl;

    // Q.setZero();

    return std::unique_ptr<StateInputCost>(
        new QuadraticJointStateInputCost(std::move(Q), std::move(R)));
    // return std::unique_ptr<StateInputCost>(
    //     new QuadraticInputCost(std::move(R)));
}

std::unique_ptr<StateCost> MobileManipulatorInterface::getQuadraticStateCost(
    const std::string& taskFile) {
    matrix_t Q(STATE_DIM, STATE_DIM);

    std::cerr << "\n #### State Cost Settings: ";
    std::cerr << "\n #### "
                 "============================================================="
                 "================\n";
    loadData::loadEigenMatrix(taskFile, "stateCost.Q", Q);
    std::cerr << "stateCost.Q:  \n" << Q << '\n';
    std::cerr << " #### "
                 "============================================================="
                 "================"
              << std::endl;

    return std::unique_ptr<StateCost>(new QuadraticStateCost(std::move(Q)));
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
std::unique_ptr<StateCost> MobileManipulatorInterface::getEndEffectorConstraint(
    PinocchioInterface pinocchioInterface, const std::string& taskFile,
    const std::string& prefix, bool usePreComputation,
    const std::string& libraryFolder, bool recompileLibraries) {
    scalar_t muPosition = 1.0;
    scalar_t muOrientation = 1.0;
    std::string name = "thing_tool";

    boost::property_tree::ptree pt;
    boost::property_tree::read_info(taskFile, pt);
    std::cerr << "\n #### " << prefix << " Settings: ";
    std::cerr << "\n #### "
                 "============================================================="
                 "================\n";
    loadData::loadPtreeValue(pt, muPosition, prefix + ".muPosition", true);
    loadData::loadPtreeValue(pt, muOrientation, prefix + ".muOrientation",
                             true);
    std::cerr << " #### "
                 "============================================================="
                 "================"
              << std::endl;

    std::unique_ptr<StateConstraint> constraint;
    if (usePreComputation) {
        MobileManipulatorPinocchioMapping<scalar_t> pinocchioMapping;
        PinocchioEndEffectorKinematics eeKinematics(pinocchioInterface,
                                                    pinocchioMapping, {name});
        constraint.reset(
            new EndEffectorConstraint(eeKinematics, *referenceManagerPtr_));
    } else {
        MobileManipulatorPinocchioMapping<ad_scalar_t> pinocchioMappingCppAd;
        PinocchioEndEffectorKinematicsCppAd eeKinematics(
            pinocchioInterface, pinocchioMappingCppAd, {name}, STATE_DIM,
            INPUT_DIM, "end_effector_kinematics", libraryFolder,
            recompileLibraries, false);
        constraint.reset(
            new EndEffectorConstraint(eeKinematics, *referenceManagerPtr_));
    }

    std::vector<std::unique_ptr<PenaltyBase>> penaltyArray(6);
    std::generate_n(penaltyArray.begin(), 3, [&] {
        return std::unique_ptr<PenaltyBase>(new QuadraticPenalty(muPosition));
    });
    std::generate_n(penaltyArray.begin() + 3, 3, [&] {
        return std::unique_ptr<PenaltyBase>(
            new QuadraticPenalty(muOrientation));
    });

    return std::unique_ptr<StateCost>(new StateSoftConstraint(
        std::move(constraint), std::move(penaltyArray)));
}

std::unique_ptr<StateCost>
MobileManipulatorInterface::getDynamicObstacleConstraint(
    PinocchioInterface pinocchioInterface,
    const DynamicObstacleSettings& settings, bool usePreComputation,
    const std::string& libraryFolder, bool recompileLibraries) {

    // Add collision spheres to the Pinocchio model
    PinocchioInterface::Model model = pinocchioInterface.getModel();
    Eigen::Matrix<scalar_t, 3, 3> R = Eigen::Matrix<scalar_t, 3, 3>::Identity();
    for (const auto& sphere : settings.collision_spheres) {

        pinocchio::FrameIndex parent_frame_id =
            model.getFrameId(sphere.parent_frame_name);
        pinocchio::Frame parent_frame = model.frames[parent_frame_id];
        pinocchio::JointIndex parent_joint_id = parent_frame.parent;

        pinocchio::SE3 T_jf = parent_frame.placement;
        pinocchio::SE3 T_fs(R, sphere.offset);
        pinocchio::SE3 T_js = T_jf * T_fs;  // sphere relative to joint

        model.addBodyFrame(sphere.name, parent_joint_id, T_js);
    }

    // Re-initialize interface with the updated model.
    pinocchioInterface = PinocchioInterface(model);

    MobileManipulatorPinocchioMapping<ad_scalar_t> pinocchioMappingCppAd;
    PinocchioEndEffectorKinematicsCppAd eeKinematics(
        pinocchioInterface, pinocchioMappingCppAd,
        settings.get_collision_frame_names(), STATE_DIM, INPUT_DIM,
        "obstacle_ee_kinematics", libraryFolder, recompileLibraries, false);

    std::unique_ptr<StateConstraint> constraint(new DynamicObstacleConstraint(
        eeKinematics, *referenceManagerPtr_, settings));

    std::unique_ptr<PenaltyBase> penalty(
        new RelaxedBarrierPenalty({settings.mu, settings.delta}));

    return std::unique_ptr<StateCost>(
        new StateSoftConstraint(std::move(constraint), std::move(penalty)));
}

std::unique_ptr<StateCost> MobileManipulatorInterface::getEndEffectorCost(
    PinocchioInterface pinocchioInterface, const std::string& taskFile,
    const std::string& prefix, bool usePreComputation,
    const std::string& libraryFolder, bool recompileLibraries) {
    std::string name = "thing_tool";
    matrix_t W(6, 6);

    std::cerr << "\n #### End Effector Cost Settings: ";
    std::cerr << "\n #### "
                 "============================================================="
                 "================\n";
    loadData::loadEigenMatrix(taskFile, "endEffectorCost.W", W);
    std::cerr << "endEffectorCost.W:  \n" << W << '\n';
    std::cerr << " #### "
                 "============================================================="
                 "================"
              << std::endl;

    std::unique_ptr<StateCost> ee_cost;
    if (usePreComputation) {
        MobileManipulatorPinocchioMapping<scalar_t> pinocchioMapping;
        PinocchioEndEffectorKinematics eeKinematics(pinocchioInterface,
                                                    pinocchioMapping, {name});
        ee_cost.reset(new EndEffectorCost(std::move(W), eeKinematics,
                                          *referenceManagerPtr_));
    } else {
        MobileManipulatorPinocchioMapping<ad_scalar_t> pinocchioMappingCppAd;
        PinocchioEndEffectorKinematicsCppAd eeKinematics(
            pinocchioInterface, pinocchioMappingCppAd, {name}, STATE_DIM,
            INPUT_DIM, "end_effector_kinematics", libraryFolder,
            recompileLibraries, false);
        ee_cost.reset(new EndEffectorCost(std::move(W), eeKinematics,
                                          *referenceManagerPtr_));
    }

    return ee_cost;
}

std::unique_ptr<StateInputCost> MobileManipulatorInterface::get_zmp_cost(
    PinocchioInterface pinocchioInterface, const std::string& taskFile,
    const std::string& prefix, bool usePreComputation,
    const std::string& libraryFolder, bool recompileLibraries) {
    std::string name = "thing_tool";

    MobileManipulatorPinocchioMapping<ad_scalar_t> pinocchioMappingCppAd;
    PinocchioEndEffectorKinematicsCppAd eeKinematics(
        pinocchioInterface, pinocchioMappingCppAd, {name}, STATE_DIM, INPUT_DIM,
        "zmp_ee_kinematics", libraryFolder, recompileLibraries, false);

    return std::unique_ptr<StateInputCost>(new ZMPCost(eeKinematics));
}

std::unique_ptr<StateInputConstraint>
MobileManipulatorInterface::getTrayBalanceConstraint(
    PinocchioInterface pinocchioInterface, const TrayBalanceSettings& settings,
    bool usePreComputation, const std::string& libraryFolder,
    bool recompileLibraries) {
    // TODO precomputation is not implemented
    const std::string name = "thing_tool";
    MobileManipulatorPinocchioMapping<ad_scalar_t> pinocchioMappingCppAd;
    PinocchioEndEffectorKinematicsCppAd pinocchioEEKinematics(
        pinocchioInterface, pinocchioMappingCppAd, {name}, STATE_DIM, INPUT_DIM,
        "tray_balance_ee_kinematics", libraryFolder, recompileLibraries, false);

    if (settings.robust) {
        return std::unique_ptr<StateInputConstraint>(
            new RobustTrayBalanceConstraints(pinocchioEEKinematics,
                                             settings.robust_params,
                                             recompileLibraries));
    } else {
        return std::unique_ptr<StateInputConstraint>(new TrayBalanceConstraints(
            pinocchioEEKinematics, settings.config, recompileLibraries));
    }
}

std::unique_ptr<StateInputCost>
MobileManipulatorInterface::getTrayBalanceSoftConstraint(
    PinocchioInterface pinocchioInterface, const TrayBalanceSettings& settings,
    bool usePreComputation, const std::string& libraryFolder,
    bool recompileLibraries) {
    // compute the hard constraint
    std::unique_ptr<StateInputConstraint> constraint = getTrayBalanceConstraint(
        pinocchioInterface, settings, usePreComputation, libraryFolder,
        recompileLibraries);

    // make it soft via penalty function
    std::vector<std::unique_ptr<PenaltyBase>> penaltyArray(
        constraint->getNumConstraints(0));
    for (int i = 0; i < constraint->getNumConstraints(0); i++) {
        penaltyArray[i].reset(
            new RelaxedBarrierPenalty({settings.mu, settings.delta}));
    }

    return std::unique_ptr<StateInputCost>(new StateInputSoftConstraint(
        std::move(constraint), std::move(penaltyArray)));
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
std::unique_ptr<StateCost>
MobileManipulatorInterface::getCollisionAvoidanceConstraint(
    PinocchioInterface pinocchioInterface,
    const CollisionAvoidanceSettings& settings,
    const std::string& obstacle_urdf_path, bool usePreComputation,
    const std::string& libraryFolder, bool recompileLibraries) {
    // only specifying link pairs (i.e. by name)
    PinocchioGeometryInterface geometryInterface(pinocchioInterface);

    // Add obstacle collision objects to the geometry model, so we can check
    // them against the robot.
    pinocchio::GeometryModel obs_geom_model =
        build_geometry_model(obstacle_urdf_path);
    geometryInterface.addGeometryObjects(obs_geom_model);

    PinocchioInterface::Model model = pinocchioInterface.getModel();
    Eigen::Matrix<scalar_t, 3, 3> R = Eigen::Matrix<scalar_t, 3, 3>::Identity();
    std::vector<pinocchio::GeometryObject> extra_spheres;
    for (auto& sphere : settings.extra_spheres) {
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

    // pinocchio::GeometryModel& geometry_model =
    //     geometryInterface.getGeometryModel();
    // for (int i = 0; i < geometry_model.ngeoms; ++i) {
    //     std::cout << geometry_model.geometryObjects[i].name << std::endl;
    // }
    geometryInterface.addCollisionPairsByName(settings.collision_link_pairs);

    const size_t numCollisionPairs = geometryInterface.getNumCollisionPairs();
    std::cerr << "SelfCollision: Testing for " << numCollisionPairs
              << " collision pairs\n";

    // std::vector<hpp::fcl::DistanceResult> distances =
    //     geometryInterface.computeDistances(pinocchioInterface);
    // for (int i = 0; i < distances.size(); ++i) {
    //     std::cout << "dist = " << distances[i].min_distance << std::endl;
    // }

    std::unique_ptr<StateConstraint> constraint;
    if (usePreComputation) {
        constraint =
            std::unique_ptr<StateConstraint>(new CollisionAvoidanceConstraint(
                MobileManipulatorPinocchioMapping<scalar_t>(),
                std::move(geometryInterface), settings.minimum_distance));
    } else {
        constraint =
            std::unique_ptr<StateConstraint>(new SelfCollisionConstraintCppAd(
                pinocchioInterface,
                MobileManipulatorPinocchioMapping<scalar_t>(),
                std::move(geometryInterface), settings.minimum_distance,
                "self_collision", libraryFolder, recompileLibraries, false));
    }

    std::unique_ptr<PenaltyBase> penalty(
        new RelaxedBarrierPenalty({settings.mu, settings.delta}));

    return std::unique_ptr<StateCost>(
        new StateSoftConstraint(std::move(constraint), std::move(penalty)));
}

std::unique_ptr<StateInputCost>
MobileManipulatorInterface::getJointStateInputLimitConstraint(
    const std::string& taskFile) {
    vector_t state_lower_bound(STATE_DIM);
    vector_t state_upper_bound(STATE_DIM);
    scalar_t state_mu = 1e-2;
    scalar_t state_delta = 1e-3;

    vector_t input_lower_bound(INPUT_DIM);
    vector_t input_upper_bound(INPUT_DIM);
    scalar_t input_mu = 1e-2;
    scalar_t input_delta = 1e-3;

    boost::property_tree::ptree pt;
    boost::property_tree::read_info(taskFile, pt);
    std::cerr << "\n #### Joint limit settings: ";
    std::cerr << "\n #### "
                 "============================================================="
                 "================\n";

    // State limits
    std::string prefix = "jointLimits.state";
    loadData::loadEigenMatrix(taskFile, prefix + ".lowerBound",
                              state_lower_bound);
    std::cerr << " #### 'state lower bound':  " << state_lower_bound.transpose()
              << std::endl;
    loadData::loadEigenMatrix(taskFile, prefix + ".upperBound",
                              state_upper_bound);
    std::cerr << " #### 'state upper bound':  " << state_upper_bound.transpose()
              << std::endl;
    loadData::loadPtreeValue(pt, state_mu, prefix + ".mu", true);
    loadData::loadPtreeValue(pt, state_delta, prefix + ".delta", true);

    // Input limits
    prefix = "jointLimits.input";
    loadData::loadEigenMatrix(taskFile, prefix + ".lowerBound",
                              input_lower_bound);
    std::cerr << " #### 'input lower bound':  " << input_lower_bound.transpose()
              << std::endl;
    loadData::loadEigenMatrix(taskFile, prefix + ".upperBound",
                              input_upper_bound);
    std::cerr << " #### 'input upper bound':  " << input_upper_bound.transpose()
              << std::endl;
    loadData::loadPtreeValue(pt, input_mu, prefix + ".mu", true);
    loadData::loadPtreeValue(pt, input_delta, prefix + ".delta", true);

    std::cerr << " #### "
                 "============================================================="
                 "================"
              << std::endl;

    std::unique_ptr<StateInputConstraint> constraint(new JointStateInputLimits);

    auto num_constraints = constraint->getNumConstraints(0);
    std::unique_ptr<PenaltyBase> barrierFunction;
    std::vector<std::unique_ptr<PenaltyBase>> penaltyArray(num_constraints);

    // State penalty
    for (int i = 0; i < STATE_DIM; i++) {
        barrierFunction.reset(
            new RelaxedBarrierPenalty({state_mu, state_delta}));
        penaltyArray[i].reset(
            new DoubleSidedPenalty(state_lower_bound(i), state_upper_bound(i),
                                   std::move(barrierFunction)));
    }

    // Input penalty
    for (int i = 0; i < INPUT_DIM; i++) {
        barrierFunction.reset(
            new RelaxedBarrierPenalty({input_mu, input_delta}));
        penaltyArray[STATE_DIM + i].reset(
            new DoubleSidedPenalty(input_lower_bound(i), input_upper_bound(i),
                                   std::move(barrierFunction)));
    }

    return std::unique_ptr<StateInputCost>(new StateInputSoftConstraint(
        std::move(constraint), std::move(penaltyArray)));
}

}  // namespace mobile_manipulator
}  // namespace ocs2
