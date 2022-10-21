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

    // Offset support area relative to new CoM
    PolygonSupportArea<Scalar> support_area =
        objects[0].support_area.offset(delta.head(2));

    BalancedObject<Scalar> composite(body, com_height, support_area,
                                     objects[0].r_tau, objects[0].mu);
    return composite;
}

// USE AT YOUR PERIL
// The pyramidal (linear) approximation below typically works much better as a
// constraint in the optimizer
// template <typename Scalar>
// VecX<Scalar> friction_constraint_ellipsoidal(
//     const BalancedObject<Scalar>& object, const Vec3<Scalar>& alpha,
//     const Vec3<Scalar>& beta, const Scalar eps) {
//     /*** Ellipsoidal approximation ***/
//     VecX<Scalar> friction_constraint(1);
//     friction_constraint << object.mu * alpha(2) -
//                                sqrt(squared(alpha(0)) + squared(alpha(1)) +
//                                     squared(beta(2) / object.r_tau) + eps);
//     return friction_constraint;
// }

// Pyramidal (linear) approximation to the limit surface
template <typename Scalar>
VecX<Scalar> friction_constraint_pyramidal(const BalancedObject<Scalar>& object,
                                           const Wrench<Scalar>& giw) {
    VecX<Scalar> friction_constraint(8);
    Scalar a = giw.force(0);
    Scalar b = giw.force(1);
    Scalar c = giw.torque(2) / object.r_tau;
    Scalar d = object.mu * giw.force(2);

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
VecX<Scalar> zmp_constraint(const BalancedObject<Scalar>& object,
                            const Wrench<Scalar>& giw) {
    Eigen::Matrix<Scalar, 2, 2> S;
    S << Scalar(0), Scalar(-1), Scalar(1), Scalar(0);

    Vec2<Scalar> zmp =
        (-object.com_height * giw.force.head(2) + S * giw.torque.head(2)) /
        giw.force(2);
    return object.support_area.zmp_constraints(zmp);
}

template <typename Scalar>
VecX<Scalar> balancing_constraints_single(
    const RigidBodyState<Scalar>& state, const Vec3<Scalar>& gravity,
    const BalancedObject<Scalar>& object,
    const BalanceConstraintsEnabled& enabled) {
    Mat3<Scalar> C_we = state.pose.orientation;
    Mat3<Scalar> C_ew = C_we.transpose();

    Mat3<Scalar> Ie = object.body.inertia;
    Mat3<Scalar> Iw = C_we * Ie * C_ew;

    Mat3<Scalar> S_angular_vel = skew3<Scalar>(state.velocity.angular);
    Mat3<Scalar> ddC_we = dC_dtt(state);

    // gravito-inertial wrench
    Wrench<Scalar> giw;
    giw.force =
        object.body.mass * C_ew *
        (state.acceleration.linear + ddC_we * object.body.com - gravity);
    giw.torque = C_ew * S_angular_vel * Iw * state.velocity.angular +
                 Ie * C_ew * state.acceleration.angular;

    // normal contact constraint
    VecX<Scalar> g_con(1);
    g_con << giw.force(2);

    // friction constraint
    VecX<Scalar> g_fric = friction_constraint_pyramidal(object, giw);

    // tipping constraint
    VecX<Scalar> g_zmp = zmp_constraint(object, giw);

    // Set disabled constraint values to unity so they are always satisfied.
    if (!enabled.friction) {
        g_fric.setOnes();
    }
    if (!enabled.normal) {
        g_con.setOnes();
    }
    if (!enabled.zmp) {
        g_zmp.setOnes();
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
BalancedObjectArrangement<T> BalancedObjectArrangement<Scalar>::cast() const {
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
    const RigidBodyState<Scalar>& state) {
    VecX<Scalar> constraints(num_constraints());
    size_t index = 0;
    for (const auto& kv : objects) {
        const auto& object = kv.second;
        VecX<Scalar> v =
            balancing_constraints_single(state, gravity, object, enabled);
        constraints.segment(index, v.rows()) = v;
        index += v.rows();
    }

    return constraints;
}

}  // namespace upright
