#pragma once

#include <Eigen/Eigen>

#include "tray_balance_constraints/dynamics.h"
#include "tray_balance_constraints/support_area.h"
#include "tray_balance_constraints/types.h"
#include "tray_balance_constraints/util.h"

template <typename Scalar>
size_t BalancedObject<Scalar>::num_constraints(
    const BalanceConstraintsEnabled& enabled) const {
    const size_t num_normal = 1 * enabled.normal;
    const size_t num_fric = 1 * enabled.friction;
    const size_t num_zmp = support_area_ptr->num_constraints() * enabled.zmp;
    return num_normal + num_fric + num_zmp;
}

template <typename Scalar>
size_t BalancedObject<Scalar>::num_parameters() const {
    return 3 + body.num_parameters() + support_area_ptr->num_parameters();
}

template <typename Scalar>
Vector<Scalar> BalancedObject<Scalar>::get_parameters() const {
    Vector<Scalar> p(num_parameters());
    p << com_height, r_tau, mu, body.get_parameters(),
        support_area_ptr->get_parameters();
    return p;
}

template <typename Scalar>
BalancedObject<Scalar> BalancedObject<Scalar>::from_parameters(
    const Vector<Scalar>& p) {
    Scalar com_height = p(0);
    Scalar r_tau = p(1);
    Scalar mu = p(2);

    size_t start = 3;
    auto body = RigidBody<Scalar>::from_parameters(p, start);

    start += body.num_parameters();
    size_t num_params_remaining = p.size() - start;

    if (num_params_remaining == 4) {
        auto support = CircleSupportArea<Scalar>::from_parameters(p, start);
        return BalancedObject<Scalar>(body, com_height, support, r_tau, mu);
    } else {
        auto support = PolygonSupportArea<Scalar>::from_parameters(p, start);
        return BalancedObject<Scalar>(body, com_height, support, r_tau, mu);
    }

    // return BalancedObject<Scalar>(body, com_height, *support_ptr,
    //                               r_tau, mu);
}

template <typename Scalar>
template <typename T>
BalancedObject<T> BalancedObject<Scalar>::cast() const {
    Vector<Scalar> p = get_parameters();
    return BalancedObject<T>::from_parameters(p.template cast<T>());
}

template <typename Scalar>
BalancedObject<Scalar> BalancedObject<Scalar>::compose(
    const std::vector<BalancedObject<Scalar>>& objects) {
    std::vector<RigidBody<Scalar>> bodies;
    for (int i = 0; i < objects.size(); ++i) {
        bodies.push_back(objects[i].body);
    }

    // Relative to objects[0] since it is the one on the bottom (and so its
    // com_height is relative to the composite object's support plane)
    RigidBody<Scalar> body = RigidBody<Scalar>::compose(bodies);
    Vec3<Scalar> delta = objects[0].body.com - body.com;
    Scalar com_height = objects[0].com_height - delta(2);

    // Using a smart pointer here ensures that the cloned object is cleaned
    // up after we pass in the reference (which is itself cloned)
    std::unique_ptr<SupportAreaBase<Scalar>> support_area_ptr(
        objects[0].support_area_ptr->clone());
    support_area_ptr->offset = delta.head(2);

    BalancedObject<Scalar> composite(body, com_height, *support_area_ptr,
                                     objects[0].r_tau, objects[0].mu);
    return composite;
}

template <typename Scalar>
Vec2<Scalar> compute_zmp(const Mat3<Scalar>& orientation,
                         const Vec3<Scalar>& angular_vel,
                         const Vec3<Scalar>& linear_acc,
                         const Vec3<Scalar>& angular_acc,
                         const BalancedObject<Scalar>& object) {
    // Tray inertia (in the tray's own frame)
    Mat3<Scalar> It = object.body.inertia;

    Mat3<Scalar> C_we = orientation;
    Mat3<Scalar> C_ew = C_we.transpose();

    Mat3<Scalar> S_angular_vel = skew3<Scalar>(angular_vel);
    Mat3<Scalar> S_angular_acc = skew3<Scalar>(angular_acc);
    Mat3<Scalar> ddC_we =
        (S_angular_acc + S_angular_vel * S_angular_vel) * C_we;

    Vec3<Scalar> g;
    g << Scalar(0), Scalar(0), Scalar(-9.81);

    Vec3<Scalar> alpha =
        object.body.mass * C_ew * (linear_acc + ddC_we * object.body.com - g);

    Mat3<Scalar> Iw = C_we * It * C_ew;
    Vec3<Scalar> beta =
        C_ew * S_angular_vel * Iw * angular_vel + It * C_ew * angular_acc;

    // tipping constraint
    Eigen::Matrix<Scalar, 2, 2> S;
    S << Scalar(0), Scalar(1), Scalar(-1), Scalar(0);
    Vec2<Scalar> zmp =
        (-object.com_height * alpha.head(2) - S * beta.head(2)) / alpha(2);
    // Vector<Scalar> h3 = object.support_area_ptr->zmp_constraints(zmp);
    return zmp;
}

template <typename Scalar>
Vector<Scalar> friction_constraint_ellipsoidal(
    const BalancedObject<Scalar>& object, const Vec3<Scalar>& alpha,
    const Vec3<Scalar>& beta, const Scalar eps) {
    /*** Ellipsoidal approximation ***/
    Vector<Scalar> friction_constraint(1);
    friction_constraint << object.mu * alpha(2) -
                               sqrt(squared(alpha(0)) + squared(alpha(1)) +
                                    squared(beta(2) / object.r_tau) + eps);
    // Scalar h1 =
    //     squared(object.mu * alpha(2)) - squared(alpha(0)) + squared(alpha(1))
    //     -
    //                                 squared(beta(2) / object.r_tau) + eps;
    return friction_constraint;
}

template <typename Scalar>
Vector<Scalar> friction_constraint_pyramidal(
    const BalancedObject<Scalar>& object, const Vec3<Scalar>& alpha,
    const Vec3<Scalar>& beta, const Scalar eps) {
    /*** Pyramidal approximation ***/
    Vector<Scalar> friction_constraint(8);
    Scalar a = alpha(0);
    Scalar b = alpha(1);
    Scalar c = beta(2) / object.r_tau;
    Scalar d = object.mu * alpha(2);

    // clang-format off
    friction_constraint << d - ( a + b + c),
              d - (-a + b + c),
              d - ( a - b + c),
              d - (-a - b + c),
              d - ( a + b - c),
              d - (-a + b - c),
              d - ( a - b - c),
              d - (-a - b - c);
    // clang-format on

    return friction_constraint;
}

template <typename Scalar>
Vector<Scalar> zmp_constraint(const BalancedObject<Scalar>& object,
                              const Vec3<Scalar>& alpha,
                              const Vec3<Scalar>& beta, const Scalar eps) {
    Eigen::Matrix<Scalar, 2, 2> S;
    S << Scalar(0), Scalar(1), Scalar(-1), Scalar(0);

    Vec2<Scalar> zmp =
        (-object.com_height * alpha.head(2) - S * beta.head(2)) / alpha(2);
    return object.support_area_ptr->zmp_constraints(zmp);

    // Vec2<Scalar> az_zmp = -object.com_height * alpha.head(2) - S *
    // beta.head(2); return
    // object.support_area_ptr->zmp_constraints_scaled(az_zmp, alpha(2));
}

template <typename Scalar>
Vector<Scalar> balancing_constraints_single(
    const Mat3<Scalar>& orientation, const Vec3<Scalar>& angular_vel,
    const Vec3<Scalar>& linear_acc, const Vec3<Scalar>& angular_acc,
    const BalancedObject<Scalar>& object,
    const BalanceConstraintsEnabled& enabled) {
    // Tray inertia (in the tray's own frame)
    Mat3<Scalar> It = object.body.inertia;

    Mat3<Scalar> C_we = orientation;
    Mat3<Scalar> C_ew = C_we.transpose();

    Mat3<Scalar> S_angular_vel = skew3<Scalar>(angular_vel);
    Mat3<Scalar> S_angular_acc = skew3<Scalar>(angular_acc);
    Mat3<Scalar> ddC_we =
        (S_angular_acc + S_angular_vel * S_angular_vel) * C_we;

    Vec3<Scalar> g;
    g << Scalar(0), Scalar(0), Scalar(-9.81);

    Vec3<Scalar> alpha =
        object.body.mass * C_ew * (linear_acc + ddC_we * object.body.com - g);

    Mat3<Scalar> Iw = C_we * It * C_ew;
    Vec3<Scalar> beta =
        C_ew * S_angular_vel * Iw * angular_vel + It * C_ew * angular_acc;

    // NOTE: SLQ solver with soft constraints is sensitive to constraint
    // values, so having small values squared makes them too close to zero.
    Scalar eps(0.01);

    // normal contact constraint
    Vector<Scalar> g_con(1);
    g_con << alpha(2);

    // friction constraint
    Vector<Scalar> g_fric =
        friction_constraint_ellipsoidal(object, alpha, beta, eps);

    // tipping constraint
    Vector<Scalar> g_zmp = zmp_constraint(object, alpha, beta, eps);

    // Set disabled constraint values to unity so they are always satisfied.
    if (!enabled.friction) {
        g_fric.setZero(0);
    }
    if (!enabled.normal) {
        g_con.setZero(0);
    }
    if (!enabled.zmp) {
        g_zmp.setZero(0);
    }

    Vector<Scalar> g_bal(object.num_constraints(enabled));
    g_bal << g_con, g_fric, g_zmp;
    return g_bal;
}

template <typename Scalar>
size_t TrayBalanceConfiguration<Scalar>::num_constraints() const {
    size_t n = 0;
    for (const auto& obj : objects) {
        n += obj.num_constraints(enabled);
    }
    return n;
}

template <typename Scalar>
size_t TrayBalanceConfiguration<Scalar>::num_parameters() const {
    size_t n = 0;
    for (const auto& obj : objects) {
        n += obj.num_parameters();
    }
    return n;
}

template <typename Scalar>
Vector<Scalar> TrayBalanceConfiguration<Scalar>::get_parameters() const {
    Vector<Scalar> parameters(num_parameters());
    size_t index = 0;
    for (const auto& obj : objects) {
        Vector<Scalar> p = obj.get_parameters();
        size_t n = p.size();
        parameters.segment(index, n) = p;
        index += n;
    }
    return parameters;
}

template <typename Scalar>
template <typename T>
TrayBalanceConfiguration<T>
TrayBalanceConfiguration<Scalar>::cast_with_parameters(
    const Vector<T>& parameters) const {
    std::vector<BalancedObject<T>> objectsT;
    size_t index = 0;
    for (const auto& obj : objects) {
        size_t n = obj.num_parameters();
        auto objT =
            BalancedObject<T>::from_parameters(parameters.segment(index, n));
        objectsT.push_back(objT);
        index += n;
    }

    return TrayBalanceConfiguration<T>(objectsT, enabled);
}

template <typename Scalar>
Vector<Scalar> TrayBalanceConfiguration<Scalar>::balancing_constraints(
    const Mat3<Scalar>& orientation, const Vec3<Scalar>& angular_vel,
    const Vec3<Scalar>& linear_acc, const Vec3<Scalar>& angular_acc) {
    Vector<Scalar> constraints(num_constraints());
    size_t index = 0;
    for (const auto& object : objects) {
        Vector<Scalar> v = balancing_constraints_single(
            orientation, angular_vel, linear_acc, angular_acc, object, enabled);
        constraints.segment(index, v.rows()) = v;
        index += v.rows();
    }

    return constraints;
}
