#pragma once

#include <ocs2_core/ComputationRequest.h>
#include <ocs2_core/soft_constraint/StateInputSoftConstraint.h>
#include <ocs2_core/soft_constraint/StateSoftConstraint.h>
#include <ocs2_oc/oc_problem/OptimalControlProblem.h>
#include <ocs2_python_interface/PythonInterface.h>

#include <upright_control/controller_interface.h>
#include <upright_control/controller_settings.h>

namespace upright {

class ControllerPythonInterface final : public ocs2::PythonInterface {
   public:
    explicit ControllerPythonInterface(const ControllerSettings& settings) {
        ControllerInterface control_interface(settings);

        problem_ = control_interface.getOptimalControlProblem();
        // ocs2::ReferenceManager* ref_ptr = control_interface.getReferenceManagerPtr().get();
        reference_manager_ptr_.reset(control_interface.getReferenceManagerPtr().get());

        // Set the reference manager -- otherwise there are problems with the
        // EndEffectorCost
        std::unique_ptr<ocs2::MPC_BASE> mpcPtr = control_interface.getMpc();
        mpcPtr->getSolverPtr()->setReferenceManager(
            control_interface.getReferenceManagerPtr());
        ocs2::PythonInterface::init(control_interface, std::move(mpcPtr));
    }

    // Get the value of the constraint underlying a soft state-input inequality
    // constraint by name
    VecXd getSoftStateInputInequalityConstraintValue(
        const std::string& name, ocs2::scalar_t t, Eigen::Ref<const VecXd> x,
        Eigen::Ref<const VecXd> u) {
        problem_.preComputationPtr->request(ocs2::Request::Constraint, t, x, u);
        return dynamic_cast<ocs2::StateInputSoftConstraint*>(
                   &problem_.softConstraintPtr->get(name))
            ->get()
            .getValue(t, x, u, *problem_.preComputationPtr);
    }

    // Hard state-input inequality constraints
    VecXd getStateInputInequalityConstraintValue(const std::string& name,
                                                 ocs2::scalar_t t,
                                                 Eigen::Ref<const VecXd> x,
                                                 Eigen::Ref<const VecXd> u) {
        problem_.preComputationPtr->request(ocs2::Request::Constraint, t, x, u);
        return problem_.inequalityConstraintPtr->get(name).getValue(
            t, x, u, *problem_.preComputationPtr);
    }

    // Hard state-input equality constraints
    VecXd getStateInputEqualityConstraintValue(const std::string& name,
                                               ocs2::scalar_t t,
                                               Eigen::Ref<const VecXd> x,
                                               Eigen::Ref<const VecXd> u) {
        problem_.preComputationPtr->request(ocs2::Request::Constraint, t, x, u);
        return problem_.equalityConstraintPtr->get(name).getValue(
            t, x, u, *problem_.preComputationPtr);
    }

    // Hard state-only inequality constraints
    VecXd getStateInequalityConstraintValue(const std::string& name,
                                            ocs2::scalar_t t,
                                            Eigen::Ref<const VecXd> x) {
        return problem_.stateEqualityConstraintPtr->get(name).getValue(
            t, x, *problem_.preComputationPtr);
    }

    // Soft state-only inequality constraints
    VecXd getSoftStateInequalityConstraintValue(const std::string& name,
                                                ocs2::scalar_t t,
                                                Eigen::Ref<const VecXd> x) {
        return dynamic_cast<ocs2::StateSoftConstraint*>(
                   &problem_.stateSoftConstraintPtr->get(name))
            ->get()
            .getValue(t, x, *problem_.preComputationPtr);
    }

    ocs2::scalar_t getCostValue(const std::string& name, ocs2::scalar_t t,
                                Eigen::Ref<const VecXd> x,
                                Eigen::Ref<const VecXd> u) {
        const auto& target = reference_manager_ptr_->getTargetTrajectories();
        return problem_.costPtr->get(name).getValue(
            t, x, u, target, *problem_.preComputationPtr);
    }

   private:
    ocs2::OptimalControlProblem problem_;
    std::shared_ptr<ocs2::ReferenceManagerInterface> reference_manager_ptr_;
};

}  // namespace upright
