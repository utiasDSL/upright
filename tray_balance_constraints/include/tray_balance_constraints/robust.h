#pragma once

#include <Eigen/Eigen>

#include "tray_balance_constraints/types.h"
#include "tray_balance_constraints/util.h"

template <typename Scalar>
using Mat39 = Eigen::Matrix<Scalar, 3, 9>;

template <typename Scalar>
Mat39<Scalar> lift_vector(const Vec3<Scalar>& v) {
    Mat39<Scalar> V;
    Mat3<Scalar> I = Mat3<Scalar>::Identity();
    V << v(0) * I, v(1) * I, v(2) * I;
    return V;
}

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
struct RobustParameterSet {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    RobustParameterSet()
        : min_support_dist(0), min_mu(0), min_r_tau(0), max_radius(0) {}

    RobustParameterSet(const std::vector<Ball<Scalar>>& balls,
                       const Scalar min_support_dist, const Scalar min_mu,
                       const Scalar min_r_tau, const Scalar max_radius)
        : balls(balls),
          min_support_dist(min_support_dist),
          min_mu(min_mu),
          min_r_tau(min_r_tau),
          max_radius(max_radius) {}

    // Cast to another underlying scalar type
    template <typename T>
    RobustParameterSet<T> cast() const {
        RobustParameterSet<T> other({}, T(min_support_dist), T(min_mu),
                                    T(min_r_tau), T(max_radius));

        for (auto& ball : balls) {
            Vec3<T> centerT = ball.center.template cast<T>();
            Ball<T> ballT(centerT, T(ball.radius));
            other.balls.push_back(ballT);
        }
        return other;
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
    const RobustParameterSet<Scalar>& param_set) {
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

    // These alternative betas are based on conversions between Frobenius and
    // matrix 2-norms, but they result in worse performance.
    // Scalar beta_xy_max = sqrt(Scalar(3.)) * squared(param_set.max_radius) *
    //                      (S_xy * C_ew *
    //                       (S_angular_vel * lift_vector<Scalar>(angular_vel) +
    //                        lift_vector<Scalar>(angular_acc)))
    //                          .norm();

    // Scalar beta_z_max = sqrt(Scalar(3.)) * squared(param_set.max_radius) *
    //                      (z.transpose() * C_ew *
    //                       (S_angular_vel * lift_vector<Scalar>(angular_vel) +
    //                        lift_vector<Scalar>(angular_acc)))
    //                          .norm();

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
                S_xy * C_ew * (linear_acc + ddC_we * ball.center - g), eps) +
            ball.radius * epsilon_norm<Scalar>(S_xy * C_ew * ddC_we, eps);

        Scalar alpha_z_min =
            (C_ew * (linear_acc + ddC_we * ball.center - g))(2) -
            ball.radius * (ddC_we.transpose() * C_we * z).norm();

        // friction
        // Scalar h1 = param_set.min_mu * alpha_z_min -
        //             sqrt(squared(alpha_xy_max) +
        //                  squared(beta_max / param_set.min_r_tau) + eps);
        Scalar h1 = sqrt(Scalar(1) + squared(param_set.min_mu)) * alpha_z_min -
                    sqrt(squared(alpha_max) +
                         squared(beta_max / param_set.min_r_tau) + eps);
        // Scalar h1 = (Scalar(1) + squared(param_set.min_mu)) *
        // squared(alpha_z_min) -
        //             squared(alpha_max) - squared(beta_max /
        //             param_set.min_r_tau);

        // contact
        Scalar h2 = alpha_z_min;

        // zmp
        // squaring seems to improve numerical stability somewhat
        Scalar h3 = squared(alpha_z_min * param_set.min_support_dist) -
                    squared(ball.max_z() * alpha_xy_max + beta_max);

        constraints.segment(index, 3) << h1, h2, h3;
        index += 3;
    }

    return constraints;
}
