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

#include <ocs2_mobile_manipulator_modified/MobileManipulatorDynamics.h>
#include <ocs2_mobile_manipulator_modified/MobileManipulatorInterface.h>
#include <ocs2_mobile_manipulator_modified/MobileManipulatorPreComputation.h>
#include <ocs2_mobile_manipulator_modified/constraint/EndEffectorConstraint.h>
#include <ocs2_mobile_manipulator_modified/constraint/JointAccelerationLimits.h>
#include <ocs2_mobile_manipulator_modified/constraint/JointStateInputLimits.h>
#include <ocs2_mobile_manipulator_modified/constraint/MobileManipulatorSelfCollisionConstraint.h>
#include <ocs2_mobile_manipulator_modified/constraint/ObstacleConstraint.h>
#include <ocs2_mobile_manipulator_modified/constraint/TrayBalanceConstraints.h>
#include <ocs2_mobile_manipulator_modified/cost/EndEffectorCost.h>
#include <ocs2_mobile_manipulator_modified/cost/QuadraticJointStateInputCost.h>
#include <ocs2_mobile_manipulator_modified/definitions.h>

#include <ros/package.h>

namespace ocs2 {
namespace mobile_manipulator {

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
MobileManipulatorInterface::MobileManipulatorInterface(
    const std::string& taskFileFolderName) {
    const std::string taskFile =
        ros::package::getPath("ocs2_mobile_manipulator_modified") + "/config/" +
        taskFileFolderName + "/task.info";
    std::cerr << "Loading task file: " << taskFile << std::endl;

    const std::string libraryFolder =
        "/tmp/ocs2/ocs2_mobile_manipulator_modified";
    std::cerr << "Generated library path: " << libraryFolder << std::endl;

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

    // TODO try loading another URDF with the obstacles, and adding it to the
    // existing model with given root joint
    // PinocchioInterface pinocchio_interface =
    // getPinocchioInterfaceFromUrdfFile(urdfPath, rootJoint);
    // PinocchioInterface::Model pinocchio_model =
    // pinocchio_interface.getModel();
    //
    // pinocchio::JointModelComposite even_rootier_joint;

    // TODO this doesn't make sense, because the root joint has these DOFs
    // added

    // return pinocchio_interface;

    return getPinocchioInterfaceFromUrdfFile(urdfPath, rootJoint);
}

pinocchio::GeometryModel MobileManipulatorInterface::build_geometry_model(const std::string& urdf_path) {
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
    const std::string urdfPath =
        ros::package::getPath("ocs2_mobile_manipulator_modified") +
        "/urdf/mm.urdf";
    const std::string obstacle_urdfPath =
        ros::package::getPath("ocs2_mobile_manipulator_modified") +
        "/urdf/obstacles.urdf";
    std::cerr << "Load Pinocchio model from " << urdfPath << '\n';

    pinocchioInterfacePtr_.reset(new PinocchioInterface(
        buildPinocchioInterface(urdfPath, obstacle_urdfPath)));
    std::cerr << *pinocchioInterfacePtr_;

    bool usePreComputation = true;
    bool recompileLibraries = true;
    boost::property_tree::ptree pt;
    boost::property_tree::read_info(taskFile, pt);
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

    // TODO do we need a final cost on state/input?
    // matrix_t Qf = matrix_t::Zero(STATE_DIM, STATE_DIM);
    // Qf.block<INPUT_DIM, INPUT_DIM>(INPUT_DIM, INPUT_DIM) =
    //     0.1 * matrix_t::Identity(INPUT_DIM, INPUT_DIM);
    // matrix_t Rf = 0.1 * matrix_t::Identity(INPUT_DIM, INPUT_DIM);
    // loadData::loadEigenMatrix(taskFile, "stateCost.Q", Qf);
    // problem_.finalCostPtr->add(
    //     "finalCost", std::unique_ptr<StateCost>(new QuadraticStateCost(Qf)));

    /* Constraints */
    // problem_.softConstraintPtr->add(
    //     "jointAccelerationLimit",
    //     getJointAccelerationLimitConstraint(taskFile));

    problem_.softConstraintPtr->add(
        "jointStateInputLimits", getJointStateInputLimitConstraint(taskFile));

    problem_.stateSoftConstraintPtr->add(
        "selfCollision",
        getSelfCollisionConstraint(*pinocchioInterfacePtr_, taskFile, urdfPath,
                                   usePreComputation, libraryFolder,
                                   recompileLibraries));

    problem_.stateSoftConstraintPtr->add(
        "obstacleAvoidance",
        getObstacleConstraint(*pinocchioInterfacePtr_, taskFile,
                              "obstacleAvoidance", usePreComputation,
                              libraryFolder, recompileLibraries));

    problem_.softConstraintPtr->add(
        "trayBalance",
        getTrayBalanceConstraint(*pinocchioInterfacePtr_, taskFile,
                                 "trayBalanceConstraints", usePreComputation,
                                 libraryFolder, recompileLibraries));

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
std::unique_ptr<MPC_DDP> MobileManipulatorInterface::getMpc() {
    std::unique_ptr<MPC_DDP> mpc(new MPC_DDP(
        mpcSettings_, ddpSettings_, *rolloutPtr_, problem_, *initializerPtr_));
    return mpc;
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

std::unique_ptr<StateCost> MobileManipulatorInterface::getObstacleConstraint(
    PinocchioInterface pinocchioInterface, const std::string& taskFile,
    const std::string& prefix, bool usePreComputation,
    const std::string& libraryFolder, bool recompileLibraries) {
    scalar_t mu = 1e-3;
    scalar_t delta = 1e-3;
    std::string name = "thing_tool";

    // boost::property_tree::ptree pt;
    // boost::property_tree::read_info(taskFile, pt);
    // std::cerr << "\n #### " << prefix << " Settings: ";
    // std::cerr << "\n #### "
    //              "============================================================="
    //              "================\n";
    // loadData::loadPtreeValue(pt, muPosition, prefix + ".muPosition", true);
    // loadData::loadPtreeValue(pt, muOrientation, prefix + ".muOrientation",
    //                          true);
    // std::cerr << " #### "
    //              "============================================================="
    //              "================"
    //           << std::endl;

    // TODO there are a bunch of these now; should reuse
    MobileManipulatorPinocchioMapping<ad_scalar_t> pinocchioMappingCppAd;
    PinocchioEndEffectorKinematicsCppAd eeKinematics(
        pinocchioInterface, pinocchioMappingCppAd, {name}, STATE_DIM, INPUT_DIM,
        "end_effector_kinematics", libraryFolder, recompileLibraries, false);
    std::unique_ptr<StateConstraint> constraint(
        new ObstacleConstraint(eeKinematics, *referenceManagerPtr_));

    // std::unique_ptr<PenaltyBase> penalty(
    //     new RelaxedBarrierPenalty({mu, delta}));
    //
    // return std::unique_ptr<StateCost>(
    //     new StateSoftConstraint(std::move(constraint), std::move(penalty)));

    std::vector<std::unique_ptr<PenaltyBase>> penaltyArray(1);
    for (int i = 0; i < penaltyArray.size(); i++) {
        penaltyArray[i].reset(new RelaxedBarrierPenalty({mu, delta}));
    }

    return std::unique_ptr<StateCost>(new StateSoftConstraint(
        std::move(constraint), std::move(penaltyArray)));
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

    MobileManipulatorPinocchioMapping<ad_scalar_t> pinocchioMappingCppAd;
    PinocchioEndEffectorKinematicsCppAd eeKinematics(
        pinocchioInterface, pinocchioMappingCppAd, {name}, STATE_DIM, INPUT_DIM,
        "end_effector_kinematics", libraryFolder, recompileLibraries, false);

    return std::unique_ptr<StateCost>(
        new EndEffectorCost(std::move(W), eeKinematics, *referenceManagerPtr_));
}

std::unique_ptr<StateInputCost>
MobileManipulatorInterface::getTrayBalanceConstraint(
    PinocchioInterface pinocchioInterface, const std::string& taskFile,
    const std::string& prefix, bool usePreComputation,
    const std::string& libraryFolder, bool recompileLibraries) {
    // Default parameters
    scalar_t mu = 1e-2;
    scalar_t delta = 1e-3;

    std::string name = "thing_tool";

    boost::property_tree::ptree pt;
    boost::property_tree::read_info(taskFile, pt);
    std::cerr << "\n #### " << prefix << " Settings: ";
    std::cerr << "\n #### "
                 "============================================================="
                 "================\n";
    loadData::loadPtreeValue(pt, mu, prefix + ".mu", true);
    loadData::loadPtreeValue(pt, delta, prefix + ".delta", true);
    std::cerr << " #### "
                 "============================================================="
                 "================"
              << std::endl;

    // TODO precompuation is not implemented

    MobileManipulatorPinocchioMapping<ad_scalar_t> pinocchioMappingCppAd;
    PinocchioEndEffectorKinematicsCppAd pinocchioEEKinematics(
        pinocchioInterface, pinocchioMappingCppAd, {name}, STATE_DIM, INPUT_DIM,
        "tray_balance_ee_kinematics", libraryFolder, recompileLibraries, false);

    std::unique_ptr<StateInputConstraint> constraint(
        new TrayBalanceConstraints(pinocchioEEKinematics));

    std::vector<std::unique_ptr<PenaltyBase>> penaltyArray(
        NUM_TRAY_BALANCE_CONSTRAINTS);
    for (int i = 0; i < NUM_TRAY_BALANCE_CONSTRAINTS; i++) {
        penaltyArray[i].reset(new RelaxedBarrierPenalty({mu, delta}));
    }

    return std::unique_ptr<StateInputCost>(new StateInputSoftConstraint(
        std::move(constraint), std::move(penaltyArray)));
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
std::unique_ptr<StateCost>
MobileManipulatorInterface::getSelfCollisionConstraint(
    PinocchioInterface pinocchioInterface, const std::string& taskFile,
    const std::string& urdfPath, bool usePreComputation,
    const std::string& libraryFolder, bool recompileLibraries) {
    std::vector<std::pair<size_t, size_t>> collisionObjectPairs;
    std::vector<std::pair<std::string, std::string>> collisionLinkPairs;
    scalar_t mu = 1e-2;
    scalar_t delta = 1e-3;
    scalar_t minimumDistance = 0.0;

    boost::property_tree::ptree pt;
    boost::property_tree::read_info(taskFile, pt);
    const std::string prefix = "selfCollision.";
    std::cerr << "\n #### SelfCollision Settings: ";
    std::cerr << "\n #### "
                 "============================================================="
                 "================\n";
    loadData::loadPtreeValue(pt, mu, prefix + "mu", true);
    loadData::loadPtreeValue(pt, delta, prefix + "delta", true);
    loadData::loadPtreeValue(pt, minimumDistance, prefix + "minimumDistance",
                             true);

    // NOTE: object vs. link is confusing: the real distinction is that
    // "object" means "id" (i.e., an index, of type size_t) and "link" means
    // "name" (i.e. string)
    // loadData::loadStdVectorOfPair(taskFile, prefix + "collisionObjectPairs",
    //                               collisionObjectPairs, true);
    loadData::loadStdVectorOfPair(taskFile, prefix + "collisionLinkPairs",
                                  collisionLinkPairs, true);
    std::cerr << " #### "
                 "============================================================="
                 "================"
              << std::endl;

    // only specifying link pairs (i.e. by name)
    // PinocchioGeometryInterface geometryInterface(pinocchioInterface,
    //                                              collisionLinkPairs, {});
    PinocchioGeometryInterface geometryInterface(pinocchioInterface);

    // NOTE: need to get a reference to avoid copy-assignment
    // pinocchio::GeometryModel& geometry_model =
    //     geometryInterface.getGeometryModel();

    // Add obstacle collision objects to the geometry model, so we can check
    // them against the robot.
    const std::string obstacle_urdf_path =
        ros::package::getPath("ocs2_mobile_manipulator_modified") +
        "/urdf/obstacles.urdf";
    pinocchio::GeometryModel obs_geom_model =
        build_geometry_model(obstacle_urdf_path);
    geometryInterface.addGeometryObjects(obs_geom_model);

    // for (int i = 0; i < obs_geom_model.ngeoms; ++i) {
    //     geometry_model.addGeometryObject(obs_geom_model.geometryObjects[i]);
    // }

    // geometryInterface.addCollisionLinkPairs(pinocchioInterface,
    //                                         collisionLinkPairs);
    // for (int i = 0; i < geometry_model.ngeoms; ++i) {
    //     std::cout << geometry_model.geometryObjects[i].name << std::endl;
    // }
    //
    // auto obs1_id = geometry_model.getGeometryId("obstacle1_link_0");
    // auto chassis_id = geometry_model.getGeometryId("chassis_link_0");
    //
    // std::cout << "obstacle1_link id = " << obs1_id << std::endl;
    // std::cout << "chassis_link id = " << chassis_id << std::endl;
    //
    // std::cout << "obstacle1_link placement = "
    //           << geometry_model.geometryObjects[obs1_id].placement << std::endl;
    // std::cout << "chassis_link parent joint id = "
    //           << geometry_model.geometryObjects[chassis_id].parentJoint
    //           << std::endl;
    //
    // geometry_model.addCollisionPair(
    //     pinocchio::CollisionPair{chassis_id, obs1_id});
    //
    // std::cout << geometry_model.collisionPairs.size() << std::endl;
    geometryInterface.addCollisionPairsByName(collisionLinkPairs);

    const size_t numCollisionPairs = geometryInterface.getNumCollisionPairs();
    std::cerr << "SelfCollision: Testing for " << numCollisionPairs
              << " collision pairs\n";

    std::vector<hpp::fcl::DistanceResult> distances =
        geometryInterface.computeDistances(pinocchioInterface);
    for (int i = 0; i < distances.size(); ++i) {
        std::cout << "dist = " << distances[i].min_distance << std::endl;
    }

    std::unique_ptr<StateConstraint> constraint;
    if (usePreComputation) {
        constraint = std::unique_ptr<StateConstraint>(
            new MobileManipulatorSelfCollisionConstraint(
                MobileManipulatorPinocchioMapping<scalar_t>(),
                std::move(geometryInterface), minimumDistance));
    } else {
        constraint =
            std::unique_ptr<StateConstraint>(new SelfCollisionConstraintCppAd(
                pinocchioInterface,
                MobileManipulatorPinocchioMapping<scalar_t>(),
                std::move(geometryInterface), minimumDistance, "self_collision",
                libraryFolder, recompileLibraries, false));
    }

    std::unique_ptr<PenaltyBase> penalty(
        new RelaxedBarrierPenalty({mu, delta}));

    return std::unique_ptr<StateCost>(
        new StateSoftConstraint(std::move(constraint), std::move(penalty)));
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
std::unique_ptr<StateInputCost>
MobileManipulatorInterface::getJointAccelerationLimitConstraint(
    const std::string& taskFile) {
    vector_t lowerBound(INPUT_DIM);
    vector_t upperBound(INPUT_DIM);
    scalar_t mu = 1e-2;
    scalar_t delta = 1e-3;

    boost::property_tree::ptree pt;
    boost::property_tree::read_info(taskFile, pt);
    const std::string prefix = "jointAccelerationLimits";
    std::cerr << "\n #### JointAccelerationLimits Settings: ";
    std::cerr << "\n #### "
                 "============================================================="
                 "================\n";
    loadData::loadEigenMatrix(taskFile, prefix + ".lowerBound", lowerBound);
    std::cerr << " #### 'lowerBound':  " << lowerBound.transpose() << std::endl;
    loadData::loadEigenMatrix(taskFile, prefix + ".upperBound", upperBound);
    std::cerr << " #### 'upperBound':  " << upperBound.transpose() << std::endl;
    loadData::loadPtreeValue(pt, mu, prefix + ".mu", true);
    loadData::loadPtreeValue(pt, delta, prefix + ".delta", true);
    std::cerr << " #### "
                 "============================================================="
                 "================"
              << std::endl;

    std::unique_ptr<StateInputConstraint> constraint(
        new JointAccelerationLimits);

    std::unique_ptr<PenaltyBase> barrierFunction;
    std::vector<std::unique_ptr<PenaltyBase>> penaltyArray(INPUT_DIM);
    for (int i = 0; i < INPUT_DIM; i++) {
        barrierFunction.reset(new RelaxedBarrierPenalty({mu, delta}));
        penaltyArray[i].reset(new DoubleSidedPenalty(
            lowerBound(i), upperBound(i), std::move(barrierFunction)));
    }

    return std::unique_ptr<StateInputCost>(new StateInputSoftConstraint(
        std::move(constraint), std::move(penaltyArray)));
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
