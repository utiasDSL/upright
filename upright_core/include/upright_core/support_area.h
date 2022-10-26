#pragma once

#include <Eigen/Eigen>

#include "upright_core/types.h"
#include "upright_core/util.h"

namespace upright {

template <typename Scalar>
std::vector<Vec2<Scalar>> cuboid_support_vertices(Scalar sx, Scalar sy);

template <typename Scalar>
std::vector<Vec2<Scalar>> cuboid_support_vertices(
    const Vec3<Scalar>& side_lengths);

template <typename Scalar>
Scalar equilateral_triangle_inscribed_radius(Scalar side_length);

// TODO this is not a fully general implementation, since it doesn't account
// for different orientations
// - this could be solved by including an angle parameter
template <typename Scalar>
std::vector<Vec2<Scalar>> equilateral_triangle_support_vertices(
    Scalar side_length);

template <typename Scalar>
std::vector<Vec2<Scalar>> regular_polygon_vertices(
    size_t n_sides, Scalar r, Scalar start_angle = Scalar(0)) {
    Scalar angle_incr = Scalar(2 * M_PI / n_sides);
    Scalar angle = start_angle;

    std::vector<Vec2<Scalar>> vertices;
    for (size_t i = 0; i < n_sides; ++i) {
        vertices.push_back(r * Vec2<Scalar>(cos(angle), sin(angle)));
        angle += angle_incr;
    }
    return vertices;
}

template <typename Scalar>
struct PolygonSupportArea {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

   public:
    PolygonSupportArea(const std::vector<Vec2<Scalar>>& vertices,
                       const Vec3<Scalar>& normal, const Mat23<Scalar>& span)
        : vertices_(vertices), normal_(normal), span_(span) {}

    PolygonSupportArea* clone() const {
        return new PolygonSupportArea(vertices_, normal_, span_);
    }

    size_t num_constraints() const { return vertices_.size(); }

    size_t num_parameters() const { return 9 + vertices_.size() * 2; }

    const std::vector<Vec2<Scalar>>& vertices() const { return vertices_; }

    const Vec3<Scalar> normal() const { return normal_; }

    Vec2<Scalar> project_onto_support_plane(const Vec3<Scalar>& point) const;

    // Constraints on the ZMP
    VecX<Scalar> zmp_constraints(const Vec3<Scalar>& zmp) const;

    // Constraints on the ZMP scaled by normal force az
    VecX<Scalar> zmp_constraints_scaled(const Vec2<Scalar>& az_zmp,
                                        Scalar& az) const;

    // Compute distance of a point from the support polygon. Negative if inside
    // the polygon.
    Scalar distance(const Vec3<Scalar>& point) const;

    VecX<Scalar> get_parameters() const;

    // Create a new support polygon with vertices offset by specified amount:
    // new_vertices = old_vertices + offset
    PolygonSupportArea<Scalar> offset(const Vec2<Scalar>& offset) const;

    static PolygonSupportArea<Scalar> from_parameters(const VecX<Scalar>& p,
                                                      const size_t index = 0);

    // // Square support area approximation to a circle
    // static PolygonSupportArea<Scalar> circle(Scalar radius);
    //
    // // Equilateral triangle support area
    // static PolygonSupportArea<Scalar> equilateral_triangle(Scalar
    // side_length);
    //
    // static PolygonSupportArea<Scalar> axis_aligned_rectangle(Scalar sx,
    //                                                          Scalar sy);

   private:
    VecX<Scalar> inner_distances_to_edges(const Vec2<Scalar>& point) const;

    std::vector<Vec2<Scalar>> vertices_;
    Vec3<Scalar> normal_;
    Mat23<Scalar> span_;
};

}  // namespace upright

#include "impl/support_area.tpp"
