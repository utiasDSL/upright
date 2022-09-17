#pragma once

#include <ocs2_core/dynamics/SystemDynamicsBaseAD.h>

#include <upright_control/dimensions.h>
#include <upright_control/types.h>

namespace upright {

template <typename Scalar>
class IntegratorDynamics {
   public:
    IntegratorDynamics(const RobotDimensions& dims) : dims_(dims) {}

    VecX<Scalar> flowmap(Scalar t, const VecX<Scalar>& x, const VecX<Scalar>& u,
                         const VecX<Scalar>& p) const {
        VecX<Scalar> dxdt(dims_.x);
        dxdt.head(dims_.q) = x.segment(dims_.q, dims_.v);
        dxdt.segment(dims_.q, dims_.v) = x.segment(dims_.q + dims_.v, dims_.v);
        dxdt.segment(dims_.q + dims_.v, dims_.v) = u;
        return dxdt;
    }

   private:
    RobotDimensions dims_;
};

template <typename Scalar>
class ObstacleDynamics {
   public:
    ObstacleDynamics() {}

    VecX<Scalar> flowmap(Scalar t, const VecX<Scalar>& x,
                         const VecX<Scalar>& p) const {
        VecX<Scalar> dxdt(9);
        dxdt << x.tail(6), VecX<Scalar>::Zero(3);
        return dxdt;
    }
};

template <typename Dynamics>
class SystemDynamics final : public ocs2::SystemDynamicsBaseAD {
   public:
    explicit SystemDynamics(const std::string& model_name,
                            const OptimizationDimensions& dims,
                            const std::string& model_folder = "/tmp/ocs2",
                            bool recompile_libraries = true,
                            bool verbose = true)
        : dims_(dims),
          robot_dynamics_(dims.robot),
          ocs2::SystemDynamicsBaseAD() {
        initialize(dims.x(), dims.u(), model_name, model_folder,
                   recompile_libraries, verbose);
    }

    ~SystemDynamics() override = default;

    SystemDynamics<Dynamics>* clone() const override {
        return new SystemDynamics<Dynamics>(*this);
    }

    VecXad systemFlowMap(ocs2::ad_scalar_t time, const VecXad& state,
                         const VecXad& input,
                         const VecXad& parameters) const override {
        VecXad dxdt(dims_.x());

        // Robot dynamics
        VecXad x_robot = state.head(dims_.robot.x);
        VecXad u_robot = input.head(dims_.robot.u);
        dxdt.head(dims_.robot.x) =
            robot_dynamics_.flowmap(time, x_robot, u_robot, parameters);

        // Obstacle dynamics
        for (int i = 0; i < dims_.o; ++i) {
            VecXad x_obs = state.segment(dims_.robot.x + i * 9, 9);
            dxdt.segment(dims_.robot.x + i * 9, 9) =
                obstacle_dynamics_.flowmap(time, x_obs, parameters);
        }
        return dxdt;
    }

   private:
    SystemDynamics(const SystemDynamics& rhs) = default;

    // Dimensions of the problem (incl. each constituent robot)
    OptimizationDimensions dims_;

    // Robot dynamics
    Dynamics robot_dynamics_;

    ObstacleDynamics<ocs2::ad_scalar_t> obstacle_dynamics_;
};

}  // namespace upright
