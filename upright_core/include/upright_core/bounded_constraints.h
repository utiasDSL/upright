#pragma once

#include <numeric>

#include <Eigen/Eigen>

#include "upright_core/bounded.h"
#include "upright_core/dynamics.h"
#include "upright_core/ellipsoid.h"
#include "upright_core/support_area.h"
#include "upright_core/types.h"
#include "upright_core/util.h"

namespace upright {

// Type of optimization: Min[imization] or Max[imization]
enum class OptType { Min, Max };

template <typename Scalar>
Scalar max_beta_projection_approx(const Vec3<Scalar>& p, const Mat3<Scalar>& R2,
                                  const Mat3<Scalar>& C_ew,
                                  const Vec3<Scalar>& angular_vel,
                                  const Vec3<Scalar>& angular_acc,
                                  const Scalar& eps) {
    return epsilon_norm<Scalar>(p.cross(C_ew * angular_vel),
                                eps) *  // TODO note I added C_ew here
               epsilon_norm<Scalar>(R2 * C_ew * angular_vel, eps) +
           epsilon_norm<Scalar>(p, eps) *
               epsilon_norm<Scalar>(R2 * C_ew * angular_acc, eps);
}

// This should only be called if the radii of gyration of the body are exact
template <typename Scalar>
Scalar beta_projection_exact(const Vec3<Scalar>& p, const Mat3<Scalar>& R2,
                             const Mat3<Scalar>& C_ew,
                             const Vec3<Scalar>& angular_vel,
                             const Vec3<Scalar>& angular_acc) {
    return p.cross(C_ew * angular_vel).dot(R2 * C_ew * angular_vel) +
           p.dot(R2 * C_ew * angular_acc);
}

// optimize a.T * r + b s.t. r \in Ellipsoid
// return optimal r
template <typename Scalar>
Scalar optimize_linear_st_ellipsoid(const Vec3<Scalar>& a, const Scalar& b,
                                    const Ellipsoid<Scalar>& ellipsoid,
                                    const Scalar& eps, OptType type) {
    // TODO should we special-case ellipsoid with rank = 0 (i.e. just a point)
    Scalar v = sqrt(a.dot(ellipsoid.Einv() * a) + eps);
    if (type == OptType::Min) {
        v = -v;
    }
    return a.dot(ellipsoid.center()) + v + b;
}

// Compute the minimum value of p.T * alpha where the CoM is constrained to lie
// inside of an ellipsoid.
template <typename Scalar>
Scalar opt_alpha_projection(const Vec3<Scalar>& p, const Mat3<Scalar>& ddC_we,
                            const Mat3<Scalar>& C_ew,
                            const Vec3<Scalar>& linear_acc,
                            const Vec3<Scalar>& g,
                            const BoundedBalancedObject<Scalar>& object,
                            Scalar eps, OptType type) {
    Vec3<Scalar> a = ddC_we.transpose() * C_ew.transpose() * p;
    Scalar b = p.transpose() * C_ew * (linear_acc - g);
    return optimize_linear_st_ellipsoid(a, b, object.body.com_ellipsoid, eps,
                                        type);
}

template <typename Scalar>
Vector<Scalar> bounded_contact_constraint(
    const Mat3<Scalar>& ddC_we, const Mat3<Scalar>& C_ew,
    const Vec3<Scalar>& linear_acc, const Vec3<Scalar>& g,
    const BoundedBalancedObject<Scalar>& object, Scalar eps) {
    Vec3<Scalar> z = Vec3<Scalar>::UnitZ();
    Vector<Scalar> contact_constraint(1);
    contact_constraint << opt_alpha_projection(z, ddC_we, C_ew, linear_acc, g,
                                               object, eps, OptType::Min);
    return contact_constraint;
}

template <typename Scalar>
Mat3<Scalar> rotation_matrix_second_derivative(
    const Mat3<Scalar>& C, const Vec3<Scalar>& angular_vel,
    const Vec3<Scalar>& angular_acc) {
    Mat3<Scalar> S_angular_vel = skew3<Scalar>(angular_vel);
    Mat3<Scalar> S_angular_acc = skew3<Scalar>(angular_acc);
    return (S_angular_acc + S_angular_vel * S_angular_vel) * C;
}

template <typename Scalar>
Vector<Scalar> bounded_friction_constraint(
    const Mat3<Scalar>& ddC_we, const Mat3<Scalar>& C_ew,
    const Vec3<Scalar>& angular_vel, const Vec3<Scalar>& linear_acc,
    const Vec3<Scalar>& angular_acc, const Vec3<Scalar>& g,
    const BoundedBalancedObject<Scalar>& object, Scalar eps) {
    // Max torque about z-axis
    Vec3<Scalar> z = Vec3<Scalar>::UnitZ();
    Mat3<Scalar> R2 = object.body.radii_of_gyration_matrix();
    Scalar beta_z_max;
    if (object.body.has_exact_radii()) {
        // std::cerr << "BODY HAS EXACT RADII" << std::endl;
        beta_z_max =
            beta_projection_exact(z, R2, C_ew, angular_vel, angular_acc);
    } else {
        beta_z_max = max_beta_projection_approx(z, R2, C_ew, angular_vel,
                                                angular_acc, eps);
    }

    // clang-format off
    Vec3<Scalar> c1, c2, c3, c4;
    c1 << Scalar(-1), Scalar(-1), object.mu_min;
    c2 << Scalar(1),  Scalar(-1), object.mu_min;
    c3 << Scalar(-1), Scalar(1),  object.mu_min;
    c4 << Scalar(1),  Scalar(1),  object.mu_min;
    // clang-format on

    Scalar min1 = opt_alpha_projection(c1, ddC_we, C_ew, linear_acc, g, object,
                                       eps, OptType::Min);
    Scalar min2 = opt_alpha_projection(c2, ddC_we, C_ew, linear_acc, g, object,
                                       eps, OptType::Min);
    Scalar min3 = opt_alpha_projection(c3, ddC_we, C_ew, linear_acc, g, object,
                                       eps, OptType::Min);
    Scalar min4 = opt_alpha_projection(c4, ddC_we, C_ew, linear_acc, g, object,
                                       eps, OptType::Min);

    Vector<Scalar> friction_constraint = Vector<Scalar>::Ones(4);

    // TODO hopefully come up with a more elegant way to handle the exact case
    if (object.body.has_exact_radii()) {
        // For the exact beta, we need to handle the sign (since in this case
        // beta is an absolute value) but can get away with sqrt(x**2 + eps),
        // since the max and min values cannot be different
        Scalar beta_positive = sqrt(squared(beta_z_max) + eps);
        friction_constraint(0) = min1 - beta_positive / object.r_tau_min;
        friction_constraint(1) = min2 - beta_positive / object.r_tau_min;
        friction_constraint(2) = min3 - beta_positive / object.r_tau_min;
        friction_constraint(3) = min4 - beta_positive / object.r_tau_min;
    } else {
        // Approximate beta_z_max is always non-negative, so we don't need to
        // handle different signs here.
        friction_constraint(0) = min1 - beta_z_max / object.r_tau_min;
        friction_constraint(1) = min2 - beta_z_max / object.r_tau_min;
        friction_constraint(2) = min3 - beta_z_max / object.r_tau_min;
        friction_constraint(3) = min4 - beta_z_max / object.r_tau_min;
    }

    return friction_constraint;
}

// TODO can I sub in the nominal ZMP constraint here?

template <typename Scalar>
Vector<Scalar> bounded_zmp_constraint(
    const Mat3<Scalar>& ddC_we, const Mat3<Scalar>& C_ew,
    const Vec3<Scalar>& angular_vel, const Vec3<Scalar>& linear_acc,
    const Vec3<Scalar>& angular_acc, const Vec3<Scalar>& g,
    const BoundedBalancedObject<Scalar>& object, Scalar eps) {
    // Four constraints per edge
    // std::vector<PolygonEdge<Scalar>> edges = object.support_area_min.edges();
    // Vector<Scalar> zmp_constraints(edges.size() * 4);

    // Vec3<Scalar> z = Vec3<Scalar>::UnitZ();
    // Eigen::Matrix<Scalar, 2, 3> S;
    // S << Scalar(0), Scalar(1), Scalar(0), Scalar(-1), Scalar(0), Scalar(0);
    Mat3<Scalar> R2 = object.body.radii_of_gyration_matrix();

    ////

    // TODO need alpha and beta
    Scalar m = object.body.mass_min;
    Vec3<Scalar> com = object.body.com_ellipsoid.center();
    Vec3<Scalar> alpha =
        m * C_ew * (linear_acc + ddC_we * com - g);

    Mat3<Scalar> C_we = C_we.transpose();
    Mat3<Scalar> S_angular_vel = skew3<Scalar>(angular_vel);
    Mat3<Scalar> It = m * R2;
    Mat3<Scalar> Iw = C_we * It * C_ew;
    Vec3<Scalar> beta =
        C_ew * S_angular_vel * Iw * angular_vel + It * C_ew * angular_acc;

    Eigen::Matrix<Scalar, 2, 2> S;
    S << Scalar(0), Scalar(1), Scalar(-1), Scalar(0);
    Vec2<Scalar> zmp =
        (-object.com_height * alpha.head(2) - S * beta.head(2)) / alpha(2);
    return object.support_area_min.zmp_constraints(zmp);

    ////

    // for (int i = 0; i < edges.size(); ++i) {
    //     Vec3<Scalar> normal3;
    //     normal3 << edges[i].normal, Scalar(0);
    //     Scalar alpha_xy_max = opt_alpha_projection(
    //         normal3, ddC_we, C_ew, linear_acc, g, object, eps, OptType::Max);
    //
    //     // NOTE: very important to use a small epsilon here
    //     // TODO: ideally, we could handle this at a lower level in CppAD
    //     Scalar r_xy_max = optimize_linear_st_ellipsoid(
    //         normal3,
    //         -edges[i].normal.dot(object.body.com_ellipsoid.center().head(2) +
    //                              edges[i].v1),
    //         object.body.com_ellipsoid, Scalar(1e-6), OptType::Max);
    //
    //     Vec3<Scalar> p = S.transpose() * edges[i].normal;
    //     Scalar beta_xy_max;
    //     if (object.body.has_exact_radii()) {
    //         beta_xy_max =
    //             beta_projection_exact(p, R2, C_ew, angular_vel, angular_acc);
    //     } else {
    //         beta_xy_max = max_beta_projection_approx(p, R2, C_ew, angular_vel,
    //                                                  angular_acc, Scalar(1e-6));
    //     }
    //
    //     Scalar alpha_z_min = opt_alpha_projection(z, ddC_we, C_ew, linear_acc,
    //                                               g, object, eps, OptType::Min);
    //     Scalar alpha_z_max = opt_alpha_projection(z, ddC_we, C_ew, linear_acc,
    //                                               g, object, eps, OptType::Max);
    //
    //     if (object.body.has_exact_radii()) {
    //         // When radii of gyration are exact, we remove the negative sign
    //         // because we want to use the exact value of beta, rather than an
    //         // upper bound. TODO as with the friction case, this can be handled
    //         // better
    //         zmp_constraints(i * 4) = beta_xy_max -
    //                                  object.max_com_height() * alpha_xy_max -
    //                                  alpha_z_max * r_xy_max;
    //         zmp_constraints(i * 4 + 1) =
    //             beta_xy_max - object.min_com_height() * alpha_xy_max -
    //             alpha_z_max * r_xy_max;
    //         zmp_constraints(i * 4 + 2) =
    //             beta_xy_max - object.max_com_height() * alpha_xy_max -
    //             alpha_z_min * r_xy_max;
    //         zmp_constraints(i * 4 + 3) =
    //             beta_xy_max - object.min_com_height() * alpha_xy_max -
    //             alpha_z_min * r_xy_max;
    //     } else {
    //         zmp_constraints(i * 4) = -beta_xy_max -
    //                                  object.max_com_height() * alpha_xy_max -
    //                                  alpha_z_max * r_xy_max;
    //         zmp_constraints(i * 4 + 1) =
    //             -beta_xy_max - object.min_com_height() * alpha_xy_max -
    //             alpha_z_max * r_xy_max;
    //         zmp_constraints(i * 4 + 2) =
    //             -beta_xy_max - object.max_com_height() * alpha_xy_max -
    //             alpha_z_min * r_xy_max;
    //         zmp_constraints(i * 4 + 3) =
    //             -beta_xy_max - object.min_com_height() * alpha_xy_max -
    //             alpha_z_min * r_xy_max;
    //     }
    // }
    //
    // return zmp_constraints;
}

// TODO make this a member of the object class
template <typename Scalar>
Vector<Scalar> bounded_balancing_constraints_single(
    const Mat3<Scalar>& orientation, const Vec3<Scalar>& angular_vel,
    const Vec3<Scalar>& linear_acc, const Vec3<Scalar>& angular_acc,
    const BoundedBalancedObject<Scalar>& object, const Vec3<Scalar>& gravity,
    const BalanceConstraintsEnabled& enabled) {
    Mat3<Scalar> C_we = orientation;
    Mat3<Scalar> C_ew = C_we.transpose();
    Mat3<Scalar> ddC_we =
        rotation_matrix_second_derivative(C_we, angular_vel, angular_acc);

    // NOTE: SLQ solver with soft constraints is sensitive to constraint
    // values, so having small values squared makes them too close to zero.
    Scalar eps(1e-6);

    // normal contact constraint
    Vector<Scalar> g_con = bounded_contact_constraint(ddC_we, C_ew, linear_acc,
                                                      gravity, object, eps);

    // friction constraint
    Vector<Scalar> g_fric =
        bounded_friction_constraint(ddC_we, C_ew, angular_vel, linear_acc,
                                    angular_acc, gravity, object, eps);

    // tipping constraint
    Vector<Scalar> g_zmp =
        bounded_zmp_constraint(ddC_we, C_ew, angular_vel, linear_acc,
                               angular_acc, gravity, object, eps);

    if (!enabled.normal) {
        g_con.setZero();
    }
    if (!enabled.friction) {
        g_fric.setZero();
    }
    if (!enabled.zmp) {
        g_zmp.setZero();
    }

    Vector<Scalar> g_bal(object.num_constraints());
    g_bal << g_con, g_fric, g_zmp;
    return g_bal;
}

template <typename Scalar>
size_t num_balancing_constraints(
    const std::vector<BoundedBalancedObject<Scalar>>& objects) {
    size_t num_constraints = 0;
    for (const auto& obj : objects) {
        num_constraints += obj.num_constraints();
    }
    return num_constraints;
}

template <typename Scalar>
size_t num_balancing_constraints(
    const std::map<std::string, BoundedBalancedObject<Scalar>>& objects) {
    size_t num_constraints = 0;
    for (const auto& kv : objects) {
        num_constraints += kv.second.num_constraints();
    }
    return num_constraints;
}

template <typename Scalar>
Vector<Scalar> balancing_constraints(
    const std::vector<BoundedBalancedObject<Scalar>>& objects,
    const Vec3<Scalar>& gravity, const BalanceConstraintsEnabled& enabled,
    const Mat3<Scalar>& orientation, const Vec3<Scalar>& angular_vel,
    const Vec3<Scalar>& linear_acc, const Vec3<Scalar>& angular_acc) {
    Vector<Scalar> constraints(num_balancing_constraints(objects));

    size_t index = 0;
    for (const auto& object : objects) {
        Vector<Scalar> v = bounded_balancing_constraints_single(
            orientation, angular_vel, linear_acc, angular_acc, object, gravity,
            enabled);
        constraints.segment(index, v.rows()) = v;
        index += v.rows();
    }
    return constraints;
}

}  // namespace upright
