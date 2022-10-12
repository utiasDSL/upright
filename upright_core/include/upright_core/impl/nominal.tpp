#pragma once

#include <Eigen/Eigen>

#include "upright_core/dynamics.h"
#include "upright_core/support_area.h"
#include "upright_core/types.h"
#include "upright_core/util.h"

namespace upright {

template <typename Scalar>
size_t BalancedObject<Scalar>::num_constraints(
    const BalanceConstraintsEnabled& enabled) const {
    const size_t num_normal = 1 * enabled.normal;
    const size_t num_fric = 8 * enabled.friction;
    const size_t num_zmp = support_area.num_constraints() * enabled.zmp;
    return num_normal + num_fric + num_zmp;
}

template <typename Scalar>
size_t BalancedObject<Scalar>::num_parameters() const {
    return 3 + body.num_parameters() + support_area.num_parameters();
}

template <typename Scalar>
VecX<Scalar> BalancedObject<Scalar>::get_parameters() const {
    VecX<Scalar> p(num_parameters());
    p << com_height, r_tau, mu, body.get_parameters(),
        support_area.get_parameters();
    return p;
}

template <typename Scalar>
BalancedObject<Scalar> BalancedObject<Scalar>::from_parameters(
    const VecX<Scalar>& p) {
    Scalar com_height = p(0);
    Scalar r_tau = p(1);
    Scalar mu = p(2);

    size_t start = 3;
    auto body = RigidBody<Scalar>::from_parameters(p, start);

    start += body.num_parameters();
    size_t num_params_remaining = p.size() - start;
    auto support_area = PolygonSupportArea<Scalar>::from_parameters(p, start);
    return BalancedObject<Scalar>(body, com_height, support_area, r_tau, mu);
}

template <typename Scalar>
template <typename T>
BalancedObject<T> BalancedObject<Scalar>::cast() const {
    VecX<Scalar> p = get_parameters();
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
    // std::unique_ptr<SupportAreaBase<Scalar>> support_area_ptr(
    //     objects[0].support_area_ptr->clone());
    // support_area_ptr->offset = delta.head(2);
    PolygonSupportArea<Scalar> support_area =
        objects[0].support_area.offset(delta.head(2));

    BalancedObject<Scalar> composite(body, com_height, support_area,
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
    S << Scalar(0), Scalar(-1), Scalar(1), Scalar(0);
    Vec2<Scalar> zmp =
        (-object.com_height * alpha.head(2) - S * beta.head(2)) / alpha(2);
    return zmp;
}

// USE AT YOUR PERIL
// The pyramidal (linear) approximation below typically works much better as a
// constraint in the optimizer
template <typename Scalar>
VecX<Scalar> friction_constraint_ellipsoidal(
    const BalancedObject<Scalar>& object, const Vec3<Scalar>& alpha,
    const Vec3<Scalar>& beta, const Scalar eps) {
    /*** Ellipsoidal approximation ***/
    VecX<Scalar> friction_constraint(1);
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
VecX<Scalar> friction_constraint_pyramidal(
    const BalancedObject<Scalar>& object, const Vec3<Scalar>& alpha,
    const Vec3<Scalar>& beta, const Scalar eps) {
    /*** Pyramidal approximation ***/
    VecX<Scalar> friction_constraint(8);
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

// TODO alpha + beta is just a wrench
template <typename Scalar>
VecX<Scalar> zmp_constraint(const BalancedObject<Scalar>& object,
                              const Vec3<Scalar>& alpha,
                              const Vec3<Scalar>& beta, const Scalar eps) {
    Eigen::Matrix<Scalar, 2, 2> S;
    S << Scalar(0), Scalar(-1), Scalar(1), Scalar(0);

    Vec2<Scalar> zmp =
        (-object.com_height * alpha.head(2) - S * beta.head(2)) / alpha(2);
    return object.support_area.zmp_constraints(zmp);
}

template <typename Scalar>
VecX<Scalar> balancing_constraints_single(
    const Mat3<Scalar>& orientation, const Vec3<Scalar>& angular_vel,
    const Vec3<Scalar>& linear_acc, const Vec3<Scalar>& angular_acc,
    const Vec3<Scalar>& gravity,
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

    Vec3<Scalar> alpha =
        object.body.mass * C_ew * (linear_acc + ddC_we * object.body.com - gravity);

    Mat3<Scalar> Iw = C_we * It * C_ew;
    Vec3<Scalar> beta =
        C_ew * S_angular_vel * Iw * angular_vel + It * C_ew * angular_acc;

    // NOTE: SLQ solver with soft constraints is sensitive to constraint
    // values, so having small values squared makes them too close to zero.
    Scalar eps(0.01);

    // normal contact constraint
    VecX<Scalar> g_con(1);
    g_con << alpha(2);

    // friction constraint
    VecX<Scalar> g_fric =
        friction_constraint_pyramidal(object, alpha, beta, eps);
        // friction_constraint_ellipsoidal(object, alpha, beta, eps);

    // tipping constraint
    VecX<Scalar> g_zmp = zmp_constraint(object, alpha, beta, eps);

    // Set disabled constraint values to unity so they are always satisfied.
    if (!enabled.friction) {
        g_fric.setZero();
    }
    if (!enabled.normal) {
        g_con.setZero();
    }
    if (!enabled.zmp) {
        g_zmp.setZero();
    }

    VecX<Scalar> g_bal(object.num_constraints(enabled));
    g_bal << g_con, g_fric, g_zmp;
    return g_bal;
}

template <typename Scalar>
size_t BalancedObjectArrangement<Scalar>::num_constraints() const {
    size_t n = 0;
    for (const auto& kv : objects) {
        n += kv.second.num_constraints(enabled);
    }
    return n;
}

template <typename Scalar>
size_t BalancedObjectArrangement<Scalar>::num_parameters() const {
    size_t n = 3;  // for gravity
    for (const auto& kv : objects) {
        n += kv.second.num_parameters();
    }
    return n;
}

template <typename Scalar>
VecX<Scalar> BalancedObjectArrangement<Scalar>::get_parameters() const {
    VecX<Scalar> parameters(num_parameters());
    parameters.head(3) = gravity;
    size_t index = 3;
    for (const auto& kv : objects) {
        VecX<Scalar> p = kv.second.get_parameters();
        size_t n = p.size();
        parameters.segment(index, n) = p;
        index += n;
    }
    return parameters;
}

template <typename Scalar>
template <typename T>
BalancedObjectArrangement<T>
BalancedObjectArrangement<Scalar>::cast() const {
    VecX<Scalar> parameters = get_parameters();
    VecX<T> parametersT = parameters.template cast<T>();

    Vec3<T> gravityT = parametersT.head(3);

    std::map<std::string, BalancedObject<T>> objectsT;
    size_t index = 3;
    for (const auto& kv : objects) {
        size_t n = kv.second.num_parameters();
        auto objT =
            BalancedObject<T>::from_parameters(parametersT.segment(index, n));
        objectsT.emplace(kv.first, objT);
        index += n;
    }

    return BalancedObjectArrangement<T>(objectsT, enabled, gravityT);
}

template <typename Scalar>
VecX<Scalar> BalancedObjectArrangement<Scalar>::balancing_constraints(
    const Mat3<Scalar>& orientation, const Vec3<Scalar>& angular_vel,
    const Vec3<Scalar>& linear_acc, const Vec3<Scalar>& angular_acc) {
    VecX<Scalar> constraints(num_constraints());
    size_t index = 0;
    for (const auto& kv : objects) {
        const auto& object = kv.second;
        VecX<Scalar> v = balancing_constraints_single(
            orientation, angular_vel, linear_acc, angular_acc, gravity, object, enabled);
        constraints.segment(index, v.rows()) = v;
        index += v.rows();
    }

    return constraints;
}

}  // namespace upright
