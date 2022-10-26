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
Scalar distance_point_to_line(const Vec2<Scalar>& point, const Vec2<Scalar>& v1,
                              const Vec2<Scalar>& v2) {
    // Compute inward-facing normal vector
    Mat2<Scalar> S;
    S << Scalar(0), Scalar(-1), Scalar(1), Scalar(0);
    Vec2<Scalar> normal = S * (v2 - v1);

    // Normalize manually rather than calling normal.normalize(), because
    // that doesn't play nice with CppAD variables. Multiple vertices
    // should never be equal, so this should always be well-defined.
    normal = normal / normal.norm();

    return normal.dot(point - v1);
}

template <typename Scalar>
Scalar distance_point_to_line_segment(const Vec2<Scalar>& point,
                                      const Vec2<Scalar>& v1,
                                      const Vec2<Scalar>& v2) {
    Vec2<Scalar> a = v2 - v1;
    Scalar d = a.norm();
    Scalar p = a.dot(point - v1);
    if (p < 0) {
        // closest point is v1
        return (point - v1).norm();
    } else if (p > d) {
        // closest point is v2
        return (point - v2).norm();
    } else {
        // closest point is on the line in between
        return std::abs(distance_point_to_line(point, v1, v2));
    }
}

template <typename Scalar>
Scalar distance_point_to_line_scaled(const Vec2<Scalar>& scaled_point,
                                     Scalar scale, const Vec2<Scalar>& v1,
                                     const Vec2<Scalar>& v2) {
    // Compute inward-facing normal vector
    Mat2<Scalar> S;
    S << Scalar(0), Scalar(-1), Scalar(1), Scalar(0);
    Vec2<Scalar> normal = S * (v2 - v1);
    normal = normal / normal.norm();

    return normal.dot(scaled_point - scale * v1);
}

template <typename Scalar>
VecX<Scalar> PolygonSupportArea<Scalar>::inner_distances_to_edges(
    const Vec2<Scalar>& point) const {
    const size_t n = vertices_.size();
    VecX<Scalar> distances(n);
    for (int i = 0; i < n - 1; ++i) {
        distances(i) =
            distance_point_to_line(point, vertices_[i], vertices_[i + 1]);
    }
    distances(n - 1) =
        distance_point_to_line(point, vertices_[n - 1], vertices_[0]);
    return distances;
}

template <typename Scalar>
Vec2<Scalar> PolygonSupportArea<Scalar>::project_onto_support_plane(
    const Vec3<Scalar>& point) const {
    return span_ * point;
}

template <typename Scalar>
VecX<Scalar> PolygonSupportArea<Scalar>::zmp_constraints(
    const Vec3<Scalar>& zmp) const {
    Vec2<Scalar> zmp_proj = project_onto_support_plane(zmp);
    return inner_distances_to_edges(zmp_proj);
}

template <typename Scalar>
VecX<Scalar> PolygonSupportArea<Scalar>::zmp_constraints_scaled(
    const Vec2<Scalar>& az_zmp, Scalar& az) const {
    const size_t n = num_constraints();
    VecX<Scalar> constraints(n);
    for (int i = 0; i < n - 1; ++i) {
        constraints(i) = distance_point_to_line_scaled(az_zmp, az, vertices_[i],
                                                       vertices_[i + 1]);
    }
    constraints(n - 1) = distance_point_to_line_scaled(
        az_zmp, az, vertices_[n - 1], vertices_[0]);
    return constraints;
}

// Minimum distance of a point from the polygon (negative if inside)
template <typename Scalar>
Scalar PolygonSupportArea<Scalar>::distance(const Vec3<Scalar>& point) const {
    Vec2<Scalar> point_proj = project_onto_support_plane(point);
    VecX<Scalar> e_dists = inner_distances_to_edges(point_proj);
    Scalar min_e_dist = e_dists.minCoeff();

    // Check if point is inside the polygon: then it is just the negative
    // minimum distance to an edge
    if (min_e_dist > 0) {
        return -min_e_dist;
    }

    // Otherwise the point is outside the polygon, and we have to check each
    // line segment
    const size_t n = vertices_.size();
    VecX<Scalar> distances(n);
    for (int i = 0; i < n - 1; ++i) {
        distances(i) = distance_point_to_line_segment(point_proj, vertices_[i],
                                                      vertices_[i + 1]);
    }
    distances(n - 1) = distance_point_to_line_segment(
        point_proj, vertices_[n - 1], vertices_[0]);
    return distances.minCoeff();
}

template <typename Scalar>
PolygonSupportArea<Scalar> PolygonSupportArea<Scalar>::offset(
    const Vec2<Scalar>& offset) const {
    std::vector<Vec2<Scalar>> offset_vertices;
    for (int i = 0; i < vertices_.size(); ++i) {
        offset_vertices.push_back(vertices_[i] + offset);
    }
    return PolygonSupportArea<Scalar>(offset_vertices, normal_, span_);
}

template <typename Scalar>
VecX<Scalar> PolygonSupportArea<Scalar>::get_parameters() const {
    size_t n = num_parameters();
    VecX<Scalar> p(n);
    p.head(3) = normal_;
    p.segment(3, 3) = span_.row(0).transpose();
    p.segment(6, 3) = span_.row(1).transpose();
    for (int i = 0; i < vertices_.size(); ++i) {
        p.segment(9 + i * 2, 2) = vertices_[i];
    }
    return p;
}

template <typename Scalar>
PolygonSupportArea<Scalar> PolygonSupportArea<Scalar>::from_parameters(
    const VecX<Scalar>& p, const size_t index) {
    // Need a normal and at least three vertices in the support area
    const size_t n = p.size() - index;
    if (n < 9 + 2 * 3) {
        throw std::runtime_error(
            "[PolygonSupportArea] Parameter vector is too small.");
    }

    Vec3<Scalar> normal = p.segment(index, 3);
    Mat23<Scalar> span;
    span << p.segment(index + 3, 3).transpose(),
        p.segment(index + 6, 3).transpose();

    const size_t num_vertices = n - 9;
    std::vector<Vec2<Scalar>> vertices;
    for (int i = 0; i < num_vertices / 2; ++i) {
        vertices.push_back(p.segment(index + 9 + i * 2, 2));
    }
    return PolygonSupportArea(vertices, normal, span);
}

// // Square support area approximation to a circle
// template <typename Scalar>
// PolygonSupportArea<Scalar> PolygonSupportArea<Scalar>::circle(Scalar radius)
// {
//     Scalar side_length = Scalar(sqrt(2.0)) * radius;
//     std::vector<Vec2<Scalar>> vertices =
//         cuboid_support_vertices(side_length, side_length);
//     return PolygonSupportArea<Scalar>(vertices);
// }
//
// // Equilateral triangle support area
// template <typename Scalar>
// PolygonSupportArea<Scalar> PolygonSupportArea<Scalar>::equilateral_triangle(
//     Scalar side_length) {
//     std::vector<Vec2<Scalar>> vertices =
//         equilateral_triangle_support_vertices(side_length);
//     return PolygonSupportArea<Scalar>(vertices);
// }
//
// template <typename Scalar>
// PolygonSupportArea<Scalar>
// PolygonSupportArea<Scalar>::axis_aligned_rectangle(
//     Scalar sx, Scalar sy) {
//     std::vector<Vec2<Scalar>> vertices = cuboid_support_vertices(sx, sy);
//     return PolygonSupportArea<Scalar>(vertices);
// }

}  // namespace upright
