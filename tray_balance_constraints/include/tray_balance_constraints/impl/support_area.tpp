#pragma once

#include <Eigen/Eigen>

#include "tray_balance_constraints/types.h"
#include "tray_balance_constraints/util.h"

template <typename Scalar>
std::vector<Vec2<Scalar>> cuboid_support_vertices(Scalar sx, Scalar sy) {
    Scalar rx = 0.5 * sx;
    Scalar ry = 0.5 * sy;

    std::vector<Vec2<Scalar>> vertices;
    vertices.push_back(Vec2<Scalar>(rx, ry));
    vertices.push_back(Vec2<Scalar>(-rx, ry));
    vertices.push_back(Vec2<Scalar>(-rx, -ry));
    vertices.push_back(Vec2<Scalar>(rx, -ry));
    return vertices;
}

template <typename Scalar>
std::vector<Vec2<Scalar>> cuboid_support_vertices(
    const Vec3<Scalar>& side_lengths) {
    return cuboid_support_vertices(side_lengths(0), side_lengths(1));
}

template <typename Scalar>
Scalar equilateral_triangle_inscribed_radius(Scalar side_length) {
    return side_length / (2 * sqrt(3.0));
}

// TODO this is not a fully general implementation, since it doesn't account
// for different orientations
template <typename Scalar>
std::vector<Vec2<Scalar>> equilateral_triangle_support_vertices(
    Scalar side_length) {
    Scalar r = equilateral_triangle_inscribed_radius(side_length);
    std::vector<Vec2<Scalar>> vertices;
    vertices.push_back(Vec2<Scalar>(2 * r, 0));
    vertices.push_back(Vec2<Scalar>(-r, 0.5 * side_length));
    vertices.push_back(Vec2<Scalar>(-r, -0.5 * side_length));
    return vertices;
}

template <typename Scalar>
Vector<Scalar> CircleSupportArea<Scalar>::zmp_constraints(
    const Vec2<Scalar>& zmp) const {
    Vec2<Scalar> e = zmp - this->offset;
    Vector<Scalar> constraints(num_constraints());
    // In the squared case here, the constraint is easily violated. In the
    // non-squared case below, the controller does not generate a stable
    // rollout.
    constraints << squared(radius - this->margin) - e.dot(e);
    // Scalar eps(0.01);
    // constraints << radius - this->margin - sqrt(e.dot(e) + eps);
    return constraints;
}

template <typename Scalar>
Vector<Scalar> CircleSupportArea<Scalar>::zmp_constraints_scaled(
    const Vec2<Scalar>& az_zmp, Scalar& az) const {
    // e is scaled by az
    Vec2<Scalar> e = az_zmp - az * this->offset;

    Vector<Scalar> constraints(num_constraints());
    constraints << squared(az * (radius - this->margin)) - e.dot(e);
    return constraints;
}

template <typename Scalar>
Vector<Scalar> CircleSupportArea<Scalar>::get_parameters() const {
    Vector<Scalar> p(num_parameters());
    p << this->offset, this->margin, radius;
    return p;
}

template <typename Scalar>
CircleSupportArea<Scalar> CircleSupportArea<Scalar>::from_parameters(
    const Vector<Scalar>& p, const size_t index) {
    size_t n = p.size() - index;
    if (n < 4) {
        throw std::runtime_error(
            "[CircleSupportArea] Parameter vector is wrong size.");
    }

    Vec2<Scalar> offset = p.template segment<2>(index);
    Scalar margin = p(index + 2);
    Scalar radius = p(index + 3);
    return CircleSupportArea<Scalar>(radius, offset, margin);
}

template <typename Scalar>
Vector<Scalar> PolygonSupportArea<Scalar>::zmp_constraints(
    const Vec2<Scalar>& zmp) const {
    const size_t n = num_constraints();
    Vector<Scalar> constraints(n);
    for (int i = 0; i < n - 1; ++i) {
        constraints(i) = edge_zmp_constraint(zmp - this->offset, vertices[i],
                                             vertices[i + 1]);
    }
    constraints(n - 1) =
        edge_zmp_constraint(zmp - this->offset, vertices[n - 1], vertices[0]);
    return constraints;
}

template <typename Scalar>
Vector<Scalar> PolygonSupportArea<Scalar>::zmp_constraints_scaled(
    const Vec2<Scalar>& az_zmp, Scalar& az) const {
    const size_t n = num_constraints();
    Vector<Scalar> constraints(n);
    for (int i = 0; i < n - 1; ++i) {
        constraints(i) = edge_zmp_constraint_scaled(
            az_zmp - az * this->offset, vertices[i], vertices[i + 1], az);
    }
    constraints(n - 1) = edge_zmp_constraint_scaled(
        az_zmp - az * this->offset, vertices[n - 1], vertices[0], az);
    return constraints;
}

template <typename Scalar>
Vector<Scalar> PolygonSupportArea<Scalar>::get_parameters() const {
    Vector<Scalar> p(num_parameters());
    Vector<Scalar> v(vertices.size() * 2);
    for (int i = 0; i < vertices.size(); ++i) {
        v.segment(i * 2, 2) = vertices[i];
    }
    p << this->offset, this->margin, v;
    return p;
}

template <typename Scalar>
PolygonSupportArea<Scalar> PolygonSupportArea<Scalar>::from_parameters(
    const Vector<Scalar>& p, const size_t index) {
    // Need at least three vertices in the support area
    size_t n = p.size() - index;
    if (n < 3 + 2 * 3) {
        throw std::runtime_error(
            "[PolygonSupportArea] Parameter vector is wrong size.");
    }

    Vec2<Scalar> offset = p.template segment<2>(index);
    Scalar margin = p(index + 2);

    std::vector<Vec2<Scalar>> vertices;
    for (int i = 0; i < (n - 3) / 2; ++i) {
        vertices.push_back(p.template segment<2>(index + 3 + i * 2));
    }
    return PolygonSupportArea(vertices, offset, margin);
}

// Square support area approximation to a circle
template <typename Scalar>
PolygonSupportArea<Scalar> PolygonSupportArea<Scalar>::circle(Scalar radius,
                                                              Scalar margin) {
    Scalar side_length = Scalar(sqrt(2.0)) * radius;
    Vec2<Scalar> offset = Vec2<Scalar>::Zero();
    std::vector<Vec2<Scalar>> vertices =
        cuboid_support_vertices(side_length, side_length);
    return PolygonSupportArea<Scalar>(vertices, offset, margin);
}

// Equilateral triangle support area
template <typename Scalar>
PolygonSupportArea<Scalar> PolygonSupportArea<Scalar>::equilateral_triangle(
    Scalar side_length, Scalar margin) {
    Vec2<Scalar> offset = Vec2<Scalar>::Zero();
    std::vector<Vec2<Scalar>> vertices =
        equilateral_triangle_support_vertices(side_length);
    return PolygonSupportArea<Scalar>(vertices, offset, margin);
}

template <typename Scalar>
PolygonSupportArea<Scalar> PolygonSupportArea<Scalar>::axis_aligned_rectangle(
    Scalar sx, Scalar sy, Scalar margin) {
    Vec2<Scalar> offset = Vec2<Scalar>::Zero();
    std::vector<Vec2<Scalar>> vertices = cuboid_support_vertices(sx, sy);
    return PolygonSupportArea<Scalar>(vertices, offset, margin);
}

template <typename Scalar>
Scalar PolygonSupportArea<Scalar>::edge_zmp_constraint(
    const Vec2<Scalar>& zmp, const Vec2<Scalar>& v1,
    const Vec2<Scalar>& v2) const {
    Mat2<Scalar> S;
    S << Scalar(0), Scalar(1), Scalar(-1), Scalar(0);

    Vec2<Scalar> normal = S * (v2 - v1);  // inward-facing normal vector

    // Normalize manually rather than calling normal.normalize(), because
    // that doesn't play nice with CppAD variables. Multiple vertices
    // should never be equal, so this should always be well-defined.
    normal = normal / normal.norm();

    return -(zmp - v1).dot(normal) - this->margin;
}

template <typename Scalar>
Scalar PolygonSupportArea<Scalar>::edge_zmp_constraint_scaled(
    const Vec2<Scalar>& az_zmp, const Vec2<Scalar>& v1, const Vec2<Scalar>& v2,
    Scalar& az) const {
    Mat2<Scalar> S;
    S << Scalar(0), Scalar(1), Scalar(-1), Scalar(0);

    Vec2<Scalar> normal = S * (v2 - v1);  // inward-facing normal vector
    normal.normalize();

    // TODO margin should be scaled up for scaled version
    return -(az_zmp - az * v1).dot(normal) - this->margin;
}