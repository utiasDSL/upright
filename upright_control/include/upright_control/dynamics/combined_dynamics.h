#pragma once

#include <ocs2_core/dynamics/SystemDynamicsBaseAD.h>

#include <upright_control/dimensions.h>
#include <upright_control/types.h>

namespace upright {

// TODO might we use a templated approach (which assumes one robot) as we
// do in the pinocchio mapping?

template <typename Scalar>
class Dynamics {
   public:
    Dynamics(const RobotDimensions& dims) : dims_(dims) {}

    virtual VecX<Scalar> flowmap(Scalar t, const VecX<Scalar>& x,
                                 const VecX<Scalar>& u,
                                 const VecX<Scalar>& p) const = 0;

   protected:
    RobotDimensions dims_;
};

template <typename Scalar>
class IntegratorDynamics : Dynamics<Scalar> {
   public:
    IntegratorDynamics(const RobotDimensions& dims) : Dynamics(dims) {}

    VecX<Scalar> flowmap(Scalar t, const VecX<Scalar>& x, const VecX<Scalar>& u,
                         const VecX<Scalar>& p) const override {
        VecX<Scalar> dxdt(dims_.x);
        dxdt.head(dims.q) = x.segment(dims.q, dims.v);
        dxdt.segment(dims.q, dims.v) = x.segment(dims.q + dims.v, dims.v);
        dxdt.segment(dims.q + dims.v, dims.v) = u;
        return dxdt;
    }
};

class SystemDynamics final : public ocs2::SystemDynamicsBaseAD {
   public:
    explicit SystemDynamics(
        const std::string& modelName,
        const std::vector<Dynamics<ocs2::ad_scalar_t>*>& dynamics,
        const OptimizationDimensions& dims;
        const std::string& modelFolder = "/tmp/ocs2",
        bool recompileLibraries = true, bool verbose = true)
        : dims_(dims), dynamics_(dynamics), ocs2::SystemDynamicsBaseAD() {
        initialize(dims.x(), dims.u(), modelName, modelFolder,
                   recompileLibraries, verbose);
    }

    ~SystemDynamics() override = default;

    SystemDynamics* clone() const override { return new SystemDynamics(*this); }

    VecXad systemFlowMap(ocs2::ad_scalar_t time, const VecXad& state,
                         const VecXad& input,
                         const VecXad& parameters) const override {
        VecXad dxdt(state_dim_);
        size_t ix = 0;
        size_t iu = 0;

        // Concatenate the dynamics for each of the constituent robots
        for (int i = 0; i < dynamics_.size(); ++i) {
            const RobotDimensions& robot_dims = dims_.robot(i);
            VecXad x = state.segment(ix, robot_dims.x);
            VecXad u = input.segment(iu, robot_dims.u);

            dxdt.segment(ix, robot_dims.x) =
                dynamics_[i].flowmap(time, x, u, parameters);

            ix += robot_dims[i].x;
            iu += robot_dims[i].u;
        }
        return dxdt;
    }

   private:
    SystemDynamics(const SystemDynamics& rhs) = default;

    // Dimensions of the problem (incl. each constituent robot)
    OptimizationDimensions dims_;

    // Subsystem dynamics
    std::vector<Dynamics<ad_scalar_t>*> dynamics_;
};

}  // namespace upright
