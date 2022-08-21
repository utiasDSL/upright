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

        // Set the reference manager -- otherwise there are problems with the
        // EndEffectorCost
        std::unique_ptr<ocs2::MPC_BASE> mpcPtr = control_interface.getMpc();
        mpcPtr->getSolverPtr()->setReferenceManager(
            control_interface.getReferenceManagerPtr());
        ocs2::PythonInterface::init(control_interface, std::move(mpcPtr));
    }

    // Get the value of the constraint underlying a soft state-input inequality
    // constraint by name
    // TODO revise these APIs
    VecXd softStateInputInequalityConstraint(const std::string& name,
                                             ocs2::scalar_t t,
                                             Eigen::Ref<const VecXd> x,
                                             Eigen::Ref<const VecXd> u) {
        problem_.preComputationPtr->request(ocs2::Request::Constraint, t, x, u);
        return dynamic_cast<ocs2::StateInputSoftConstraint*>(
                   &problem_.softConstraintPtr->get(name))
            ->get()
            .getValue(t, x, u, *problem_.preComputationPtr);
    }

    // Get the value of a hard state-input inequality constraint by name
    VecXd stateInputInequalityConstraint(const std::string& name,
                                         ocs2::scalar_t t,
                                         Eigen::Ref<const VecXd> x,
                                         Eigen::Ref<const VecXd> u) {
        problem_.preComputationPtr->request(ocs2::Request::Constraint, t, x, u);
        return problem_.inequalityConstraintPtr->get(name).getValue(
            t, x, u, *problem_.preComputationPtr);
    }

    VecXd getStateInputEqualityConstraintValue(const std::string& name,
                                       ocs2::scalar_t t,
                                       Eigen::Ref<const VecXd> x,
                                       Eigen::Ref<const VecXd> u) {
        problem_.preComputationPtr->request(ocs2::Request::Constraint, t, x, u);
        return problem_.equalityConstraintPtr->get(name).getValue(
            t, x, u, *problem_.preComputationPtr);
    }

    // TODO this is soft
    VecXd stateInequalityConstraint(const std::string& name, ocs2::scalar_t t,
                                    Eigen::Ref<const VecXd> x) {
        return dynamic_cast<ocs2::StateSoftConstraint*>(
                   &problem_.stateSoftConstraintPtr->get(name))
            ->get()
            .getValue(t, x, *problem_.preComputationPtr);
    }

   private:
    ocs2::OptimalControlProblem problem_;
};

}  // namespace upright
