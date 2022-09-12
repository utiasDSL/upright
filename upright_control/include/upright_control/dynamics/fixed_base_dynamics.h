#pragma once

#include <ocs2_core/dynamics/SystemDynamicsBaseAD.h>

#include <upright_control/dynamics/dimensions.h>
#include <upright_control/types.h>

namespace upright {

class FixedBaseDynamics final : public ocs2::SystemDynamicsBaseAD {
   public:
    explicit FixedBaseDynamics(const std::string& modelName,
                               const RobotDimensions& dims,
                               const std::string& modelFolder = "/tmp/ocs2",
                               bool recompileLibraries = true,
                               bool verbose = true);

    ~FixedBaseDynamics() override = default;

    FixedBaseDynamics* clone() const override {
        return new FixedBaseDynamics(*this);
    }

    VecXad systemFlowMap(ocs2::ad_scalar_t time, const VecXad& state,
                         const VecXad& input,
                         const VecXad& parameters) const override;

   private:
    FixedBaseDynamics(const FixedBaseDynamics& rhs) = default;

    RobotDimensions dims_;
};

class ObstacleDynamics final : public ocs2::SystemDynamicsBaseAD {
   public:
    explicit ObstacleDynamics(const std::string& modelName,
                              const std::string& modelFolder = "/tmp/ocs2",
                              bool recompileLibraries = true,
                              bool verbose = true)
        : ocs2::SystemDynamicsBaseAD() {
        initialize(9, 0, modelName, modelFolder, recompileLibraries, verbose);
    }

    ~ObstacleDynamics() override = default;

    ObstacleDynamics* clone() const override {
        return new ObstacleDynamics(*this);
    }

    VecXad systemFlowMap(ocs2::ad_scalar_t time, const VecXad& state,
                         const VecXad& input,
                         const VecXad& parameters) const override {
        VecXad dqdt = state.segment(3, 3);
        VecXad dvdt = state.tail(3);
        VecXad dadt = VecXad::Zero(3);  // constant acceleration

        VecXad dxdt(9);
        dxdt << dqdt, dvdt, dadt;
        return dxdt;
    }

   private:
    ObstacleDynamics(const ObstacleDynamics& rhs) = default;
};

class CombinedDynamics final : public ocs2::SystemDynamicsBaseAD {
   public:
    explicit CombinedDynamics(
        const std::string& modelName,
        const std::vector<ocs2::SystemDynamicsBaseAD>& dynamics,
        const OptimizationDimensions& dims;
        const std::string& modelFolder = "/tmp/ocs2",
        bool recompileLibraries = true, bool verbose = true)
        : dims_(dims), dynamics_(dynamics), ocs2::SystemDynamicsBaseAD() {
        // TODO is this correct? using u modified by c?
        initialize(dims.x, dims.u, modelName, modelFolder, recompileLibraries,
                   verbose);
    }

    ~CombinedDynamics() override = default;

    CombinedDynamics* clone() const override {
        return new CombinedDynamics(*this);
    }

    VecXad systemFlowMap(ocs2::ad_scalar_t time, const VecXad& state,
                         const VecXad& input,
                         const VecXad& parameters) const override {
        const std::vector<RobotDimensions> robot_dims = dims_.robots();

        VecXad dxdt(state_dim_);
        size_t ix = 0;
        size_t iu = 0;

        // Concatenate the dynamics for each of the constituent robots
        for (int i = 0; i < dynamics_.size(); ++i) {
            VecXad x = state.segment(ix, robot_dims[i].x);
            VecXad u = input.segment(iu, robot_dims[i].u);

            dxdt.segment(ix, robot_dims[i].x) =
                dynamics_[i].systemFlowMap(time, x, u, parameters);

            ix += robot_dims[i].x;
            iu += robot_dims[i].u;
        }
        return dxdt;
    }

   private:
    CombinedDynamics(const CombinedDynamics& rhs) = default;

    // Dimensions of the problem (incl. each constituent robot)
    OptimizationDimensions dims_;

    // Subsystem dynamics
    std::vector<ocs2::SystemDynamicsBaseAD> dynamics_;
};

}  // namespace upright
