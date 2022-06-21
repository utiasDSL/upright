#pragma once

#include <Eigen/Eigen>

#include "upright_core/types.h"
#include "upright_core/util.h"

namespace upright {

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
Vector<Scalar> PolygonSupportArea<Scalar>::zmp_constraints(
    const Vec2<Scalar>& zmp, const Scalar& margin) const {
    const size_t n = num_constraints();
    Vector<Scalar> constraints(n);
    for (int i = 0; i < n - 1; ++i) {
        constraints(i) =
            edge_zmp_constraint(zmp, vertices[i], vertices[i + 1], margin);
    }
    constraints(n - 1) =
        edge_zmp_constraint(zmp, vertices[n - 1], vertices[0], margin);
    return constraints;
}

// template <typename Scalar>
// Vector<Scalar> PolygonSupportArea<Scalar>::zmp_constraints_scaled(
//     const Vec2<Scalar>& az_zmp, Scalar& az) const {
//     const size_t n = num_constraints();
//     Vector<Scalar> constraints(n);
//     for (int i = 0; i < n - 1; ++i) {
//         constraints(i) = edge_zmp_constraint_scaled(
//             az_zmp - az, vertices[i], vertices[i + 1], az);
//     }
//     constraints(n - 1) = edge_zmp_constraint_scaled(
//         az_zmp - az, vertices[n - 1], vertices[0], az);
//     return constraints;
// }

template <typename Scalar>
Vector<Scalar> PolygonSupportArea<Scalar>::get_parameters() const {
    Vector<Scalar> p(num_parameters());
    for (int i = 0; i < vertices.size(); ++i) {
        p.segment(i * 2, 2) = vertices[i];
    }
    return p;
}

template <typename Scalar>
PolygonSupportArea<Scalar> PolygonSupportArea<Scalar>::offset(
    const Vec2<Scalar>& offset) const {
    std::vector<Vec2<Scalar>> offset_vertices;
    for (int i = 0; i < vertices.size(); ++i) {
        offset_vertices.push_back(vertices[i] + offset);
    }
    return PolygonSupportArea<Scalar>(offset_vertices);
}

template <typename Scalar>
PolygonSupportArea<Scalar> PolygonSupportArea<Scalar>::from_parameters(
    const Vector<Scalar>& p, const size_t index) {
    // Need at least three vertices in the support area
    size_t n = p.size() - index;
    if (n < 2 * 3) {
        throw std::runtime_error(
            "[PolygonSupportArea] Parameter vector is too small.");
    }

    std::vector<Vec2<Scalar>> vertices;
    for (int i = 0; i < n / 2; ++i) {
        vertices.push_back(p.template segment<2>(index + i * 2));
    }
    return PolygonSupportArea(vertices);
}

// Square support area approximation to a circle
template <typename Scalar>
PolygonSupportArea<Scalar> PolygonSupportArea<Scalar>::circle(Scalar radius) {
    Scalar side_length = Scalar(sqrt(2.0)) * radius;
    std::vector<Vec2<Scalar>> vertices =
        cuboid_support_vertices(side_length, side_length);
    return PolygonSupportArea<Scalar>(vertices);
}

// Equilateral triangle support area
template <typename Scalar>
PolygonSupportArea<Scalar> PolygonSupportArea<Scalar>::equilateral_triangle(
    Scalar side_length) {
    std::vector<Vec2<Scalar>> vertices =
        equilateral_triangle_support_vertices(side_length);
    return PolygonSupportArea<Scalar>(vertices);
}

template <typename Scalar>
PolygonSupportArea<Scalar> PolygonSupportArea<Scalar>::axis_aligned_rectangle(
    Scalar sx, Scalar sy) {
    std::vector<Vec2<Scalar>> vertices = cuboid_support_vertices(sx, sy);
    return PolygonSupportArea<Scalar>(vertices);
}

template <typename Scalar>
Scalar PolygonSupportArea<Scalar>::edge_zmp_constraint(
    const Vec2<Scalar>& zmp, const Vec2<Scalar>& v1, const Vec2<Scalar>& v2,
    const Scalar& margin) const {
    Mat2<Scalar> S;
    S << Scalar(0), Scalar(1), Scalar(-1), Scalar(0);

    Vec2<Scalar> normal = S * (v2 - v1);  // inward-facing normal vector

    // Normalize manually rather than calling normal.normalize(), because
    // that doesn't play nice with CppAD variables. Multiple vertices
    // should never be equal, so this should always be well-defined.
    normal = normal / normal.norm();

    return -(zmp - v1).dot(normal) - margin;
}

// template <typename Scalar>
// Scalar PolygonSupportArea<Scalar>::edge_zmp_constraint_scaled(
//     const Vec2<Scalar>& az_zmp, const Vec2<Scalar>& v1, const Vec2<Scalar>&
//     v2, Scalar& az) const { Mat2<Scalar> S; S << Scalar(0), Scalar(1),
//     Scalar(-1), Scalar(0);
//
//     Vec2<Scalar> normal = S * (v2 - v1);  // inward-facing normal vector
//     normal.normalize();
//
//     // TODO margin should be scaled up for scaled version
//     return -(az_zmp - az * v1).dot(normal) - this->margin;
// }

}  // namespace upright