#pragma once

#include <Eigen/Eigen>

#include "tray_balance_constraints/types.h"
#include "tray_balance_constraints/util.h"

// 3D ball
template <typename Scalar>
struct Ball {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    Ball(const Vec3<Scalar>& center, const Scalar radius)
        : center(center), radius(radius) {}

    Scalar max_z() { return center(2) + radius; }

    Vec3<Scalar> center;
    Scalar radius;
};

template <typename Scalar>
struct ParameterSet {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    ParameterSet(const std::vector<Ball<Scalar>>& balls,
                 const Scalar min_support_dist, const Scalar min_mu,
                 const Scalar min_r_tau)
        : balls(balls),
          min_support_dist(min_support_dist),
          min_mu(min_mu),
          min_r_tau(min_r_tau),
          max_radius(0) {
        // Compute maximum radius through convex hull of any of the
        // spheres
        // TODO: this is still not a fully general implementation, as the
        // spheres containing center of mass need not be the same as the full
        // covering of the body
        for (int i = 0; i < balls.size(); ++i) {
            for (int j = i + 1; j < balls.size(); ++j) {
                Scalar d = 0.5 * ((balls[i].center + balls[j].center).norm() +
                                  balls[i].radius + balls[j].radius);
                if (d > max_radius) {
                    max_radius = d;
                }
            }
        }
    }

    std::vector<Ball<Scalar>> balls;
    Scalar min_support_dist;
    Scalar min_mu;
    Scalar min_r_tau;
    Scalar max_radius;
};

template <typename Scalar>
Vector<Scalar> robust_balancing_constraints(
    const Mat3<Scalar>& orientation, const Vec3<Scalar>& angular_vel,
    const Vec3<Scalar>& linear_acc, const Vec3<Scalar>& angular_acc,
    const ParameterSet<Scalar>& param_set) {
    Mat3<Scalar> C_we = orientation;
    Mat3<Scalar> C_ew = C_we.transpose();

    Mat3<Scalar> S_angular_vel = skew3<Scalar>(angular_vel);
    Mat3<Scalar> S_angular_acc = skew3<Scalar>(angular_acc);
    Mat3<Scalar> ddC_we =
        (S_angular_acc + S_angular_vel * S_angular_vel) * C_we;

    Vec3<Scalar> g;
    g << Scalar(0), Scalar(0), Scalar(-9.81);
    Scalar eps(0.01);

    Eigen::Matrix<Scalar, 2, 3> S_xy;
    S_xy << Scalar(1), Scalar(0), Scalar(0), Scalar(0), Scalar(1), Scalar(0);
    Vec3<Scalar> z;
    z << Scalar(0), Scalar(0), Scalar(1);

    Scalar beta_max =
        squared(param_set.max_radius) *
        (angular_vel.dot(angular_vel) + epsilon_norm<Scalar>(angular_acc, eps));

    Vector<Scalar> constraints(3 * param_set.balls.size());
    size_t index = 0;
    for (auto ball : param_set.balls) {
        // TODO .norm() computes Frobenius norm for matrices, which is not
        // actually what we want
        Scalar alpha_max =
            epsilon_norm<Scalar>(linear_acc + ddC_we * ball.center - g, eps) +
            ball.radius * epsilon_norm<Scalar>(ddC_we, eps);

        Scalar alpha_xy_max =
            epsilon_norm<Scalar>(
                S_xy * C_ew * (linear_acc + ddC_we * ball.center - g), Scalar(0)) +
            ball.radius * epsilon_norm<Scalar>(S_xy * C_ew * ddC_we, eps);

        Vec3<Scalar> r_min = ddC_we.transpose() * C_we * z;
        r_min = -ball.radius * r_min / r_min.norm();
        Scalar alpha_z_min =
            (C_ew * (linear_acc + ddC_we * (ball.center + r_min) - g))(2);

        // friction
        // Scalar h1 = (Scalar(1) + squared(param_set.min_mu)) *
        // squared(alpha_z_min) -
        //             squared(alpha_xy_max) - squared(beta_max /
        //             param_set.min_r_tau);
        Scalar h1 = sqrt(Scalar(1) + squared(param_set.min_mu)) * alpha_z_min -
                    sqrt(squared(alpha_max) +
                         squared(beta_max / param_set.min_r_tau) + eps);

        // contact
        Scalar h2 = alpha_z_min;

        // zmp
        // TODO: numerical issues with the alpha_xy_max term in particular
        // - this probably has to do with the non-differentiability of the norm
        // - the C_ew term getting gravity g in there causes problems
        Scalar h3 = alpha_z_min * param_set.min_support_dist -
                    ball.max_z() * alpha_xy_max - beta_max;

        constraints.segment(index, 3) << h1, h2, h3;
        index += 3;
    }

    return constraints;
}
