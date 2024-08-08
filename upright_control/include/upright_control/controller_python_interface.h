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
    explicit ControllerPythonInterface(const ControllerSettings& settings)
        : control_interface_(settings) {
        problem_ = control_interface_.getOptimalControlProblem();

        // Set the reference manager -- otherwise there are problems with the
        // EndEffectorCost
        std::unique_ptr<ocs2::MPC_BASE> mpc_ptr = control_interface_.get_mpc();
        mpc_ptr->getSolverPtr()->setReferenceManager(
            control_interface_.getReferenceManagerPtr());
        ocs2::PythonInterface::init(control_interface_, std::move(mpc_ptr));
    }

    ocs2::scalar_t getLastSolveTime() const {
        return mpcMrtInterface_->getLastSolveTime();
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
        const auto& target = control_interface_.getReferenceManagerPtr()
                                 ->getTargetTrajectories();
        return problem_.costPtr->get(name).getValue(
            t, x, u, target, *problem_.preComputationPtr);
    }

   private:
    ControllerInterface control_interface_;
    ocs2::OptimalControlProblem problem_;
};

}  // namespace upright
