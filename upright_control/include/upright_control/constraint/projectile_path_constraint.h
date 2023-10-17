#pragma once

#include <ocs2_core/constraint/StateConstraint.h>
#include <ocs2_oc/synchronized_module/ReferenceManager.h>
#include <ocs2_robotic_tools/end_effector/EndEffectorKinematics.h>
#include <upright_control/reference_trajectory.h>
#include <upright_control/types.h>

namespace upright {

// Use Newton's method to find roots of cubic equation
ocs2::scalar_t cubic_newtons(ocs2::scalar_t a, ocs2::scalar_t b,
                             ocs2::scalar_t c, ocs2::scalar_t d,
                             ocs2::scalar_t x0, ocs2::scalar_t tol) {
    int max_iter = 10;

    ocs2::scalar_t x = x0;

    for (int i = 0; i < max_iter; ++i) {
        ocs2::scalar_t f = a * x * x * x + b * x * x + c * x + d;
        ocs2::scalar_t dfdx = 3 * a * x * x + 2 * b * x + c;
        ocs2::scalar_t update = f / dfdx;
        x = x - update;
        if (std::abs(update) < tol) {
            return x;
        }
    }
    std::cout << "Max iterations reached." << std::endl;
    return x;
}

// Find time at which a projectile will be closest to a point r.
ocs2::scalar_t projectile_closest_time(const Vec3d& r, const Vec3d& r0,
                                       const Vec3d& v0, const Vec3d& g,
                                       ocs2::scalar_t t_guess) {
    Vec3d dr = r - r0;
    ocs2::scalar_t a = g.dot(g);
    ocs2::scalar_t b = 3 * v0.dot(g);
    ocs2::scalar_t c = 2 * (v0.dot(v0) - dr.dot(g));
    ocs2::scalar_t d = -2 * dr.dot(v0);

    ocs2::scalar_t t = cubic_newtons(a, b, c, d, t_guess, 1e-4);
    return t;
}

class ProjectilePathConstraint final : public ocs2::StateConstraint {
   public:
    ProjectilePathConstraint(
        const ocs2::EndEffectorKinematics<ocs2::scalar_t>& kinematics,
        const ocs2::ReferenceManager& reference_manager, const VecXd& distances,
        ocs2::scalar_t scale)
        : ocs2::StateConstraint(ocs2::ConstraintOrder::Linear),
          kinematics_ptr_(kinematics.clone()),
          reference_manager_ptr_(&reference_manager),
          distances_(distances),
          scale_(scale) {
        if (kinematics.getIds().size() != distances.size()) {
            std::cout << "num ids = " << kinematics.getIds().size() << std::endl;
            std::cout << "num dists = " << distances.size() << std::endl;
            throw std::runtime_error(
                "Number of distances and EEs must be equal!");
        }
    }

    ~ProjectilePathConstraint() override = default;

    ProjectilePathConstraint* clone() const override {
        return new ProjectilePathConstraint(
            *kinematics_ptr_, *reference_manager_ptr_, distances_, scale_);
    }

    size_t getNumConstraints(ocs2::scalar_t time) const override {
        return kinematics_ptr_->getIds().size();
    }

    VecXd getValue(ocs2::scalar_t time, const VecXd& state,
                   const ocs2::PreComputation&) const override {
        const auto& target = reference_manager_ptr_->getTargetTrajectories();
        VecXd xd = target.stateTrajectory[0];
        ocs2::scalar_t s = xd(7);

        VecXd x_obs = state.tail(9);
        Vec3d r_obs = x_obs.head(3);
        Vec3d v_obs = x_obs.segment(3, 3);
        Vec3d a_obs = x_obs.tail(3);

        const size_t nc = getNumConstraints(time);
        VecXd constraint(nc);
        for (int i = 0; i < nc; ++i) {
            Vec3d r_ew_w = kinematics_ptr_->getPosition(state)[i];

            ocs2::scalar_t dt = 0.0;
            if (s > 0.5) {
                dt = projectile_closest_time(r_ew_w, r_obs, v_obs, a_obs, 0.0);
                dt = std::max(0.0, dt);
            }
            Vec3d r_closest = r_obs + dt * v_obs + 0.5 * dt * dt * a_obs;

            Vec3d delta = r_ew_w - r_closest;
            Vec3d normal = delta.normalized();
            ocs2::scalar_t w = scale_ / distances_(i);
            constraint(i) = w * s * (normal.dot(delta) - distances_(i));
        }
        return constraint;
    }

    ocs2::VectorFunctionLinearApproximation getLinearApproximation(
        ocs2::scalar_t time, const VecXd& state,
        const ocs2::PreComputation& pre_comp) const override {
        const auto& target = reference_manager_ptr_->getTargetTrajectories();
        VecXd xd = target.stateTrajectory[0];
        ocs2::scalar_t s = xd(7);

        VecXd x_obs = state.tail(9);
        Vec3d r_obs = x_obs.head(3);
        Vec3d v_obs = x_obs.segment(3, 3);
        Vec3d a_obs = x_obs.tail(3);

        const size_t nc = getNumConstraints(time);
        auto approximation =
            ocs2::VectorFunctionLinearApproximation(nc, state.rows(), 0);
        approximation.setZero(nc, state.rows(), 0);

        for (int i = 0; i < nc; ++i) {
            const auto position_approx =
                kinematics_ptr_->getPositionLinearApproximation(state)[i];
            Vec3d r_ew_w = position_approx.f;

            ocs2::scalar_t dt = 0.0;
            if (s > 0.5) {
                dt = projectile_closest_time(r_ew_w, r_obs, v_obs, a_obs, 0.0);
                dt = std::max(0.0, dt);  // don't care about the past
            }
            MatXd A_obs(3, 9);
            A_obs << Mat3d::Identity(), dt * Mat3d::Identity(),
                0.5 * dt * dt * Mat3d::Identity();

            // closest point on the obstacle's predicted future path
            Vec3d r_closest = A_obs * x_obs;

            Vec3d delta = r_ew_w - r_closest;
            Vec3d normal =
                delta.normalized();  // TODO: 0 when in contact---bad Jacobian?

            // We scale the whole constraint by scale_ / min_distance, which is
            // somewhat arbitrary. In the case of soft L2 penalties, this is
            // equivalent to using a slack weight of (scale_ / min_distance)**2.
            ocs2::scalar_t w = scale_ / distances_(i);
            approximation.f(i) = w * s * (normal.dot(delta) - distances_(i));

            MatXd dobs_dx = MatXd::Zero(3, state.size());
            dobs_dx.rightCols(9) = A_obs;
            approximation.dfdx.row(i) << w * s * normal.transpose() *
                                             (position_approx.dfdx - dobs_dx);
        }
        return approximation;
    }

   private:
    ProjectilePathConstraint(const ProjectilePathConstraint& other) = default;

    ocs2::scalar_t scale_;
    VecXd distances_;
    std::unique_ptr<ocs2::EndEffectorKinematics<ocs2::scalar_t>>
        kinematics_ptr_;
    const ocs2::ReferenceManager* reference_manager_ptr_;
};

}  // namespace upright
