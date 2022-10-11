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
struct PolygonEdge {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    PolygonEdge(const Vec2<Scalar>& v1, const Vec2<Scalar>& v2)
        : v1(v1), v2(v2) {
        // TODO would be nice to revise all this to use inward-facing normals
        Mat2<Scalar> S;
        S << Scalar(0), Scalar(1), Scalar(-1), Scalar(0);
        normal = S * (v2 - v1);
        normal = normal / normal.norm();
    }

    Vec2<Scalar> v1;
    Vec2<Scalar> v2;
    Vec2<Scalar> normal;  // outward-facing
};

template <typename Scalar>
struct PolygonSupportArea {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

   public:
    PolygonSupportArea(const std::vector<Vec2<Scalar>>& vertices)
        : vertices_(vertices) {}

    PolygonSupportArea* clone() const {
        return new PolygonSupportArea(vertices_);
    }

    size_t num_constraints() const { return vertices_.size(); }

    size_t num_parameters() const { return vertices_.size() * 2; }

    const std::vector<Vec2<Scalar>>& vertices() const { return vertices_; }

    // Get the edges of the polygon composing the support area
    std::vector<PolygonEdge<Scalar>> edges() const {
        std::vector<PolygonEdge<Scalar>> es;
        for (int i = 0; i < vertices_.size() - 1; ++i) {
            es.push_back(PolygonEdge<Scalar>(vertices_[i], vertices_[i + 1]));
        }
        es.push_back(PolygonEdge<Scalar>(vertices_.back(), vertices_.front()));
        return es;
    }

    // TODO combine and generalize with implementation of distance outside
    // polygon below
    Vector<Scalar> zmp_constraints(const Vec2<Scalar>& zmp) const;

    Vector<Scalar> zmp_constraints_scaled(const Vec2<Scalar>& az_zmp,
                                          Scalar& az) const {
        const size_t n = num_constraints();
        Vector<Scalar> constraints(n);
        for (int i = 0; i < n - 1; ++i) {
            constraints(i) = edge_zmp_constraint_scaled(az_zmp, vertices_[i],
                                                        vertices_[i + 1], az);
        }
        constraints(n - 1) = edge_zmp_constraint_scaled(az_zmp, vertices_[n - 1],
                                                        vertices_[0], az);
        return constraints;
    }

    // This is used in the bindings
    Scalar distance_outside(const Vec2<Scalar>& point) const {
        const size_t n = vertices_.size();
        Scalar dist_inside = 100;  // arbitrary large value for now
        Scalar dist_inside_edge = 0;
        for (int i = 0; i < n - 1; ++i) {
            dist_inside_edge =
                edge_zmp_constraint(point, vertices_[i], vertices_[i + 1]);
            if (dist_inside_edge < dist_inside) {
                dist_inside = dist_inside_edge;
            }
        }
        dist_inside_edge =
            edge_zmp_constraint(point, vertices_[n - 1], vertices_[0]);
        if (dist_inside_edge < dist_inside) {
            dist_inside = dist_inside_edge;
        }
        Scalar dist_outside = -dist_inside;
        return dist_outside;
    }

    Vector<Scalar> get_parameters() const;

    // Create a new support polygon with vertices offset by specified amount:
    // new_vertices = old_vertices + offset
    PolygonSupportArea<Scalar> offset(const Vec2<Scalar>& offset) const;

    static PolygonSupportArea<Scalar> from_parameters(const Vector<Scalar>& p,
                                                      const size_t index = 0);

    // Square support area approximation to a circle
    static PolygonSupportArea<Scalar> circle(Scalar radius);

    // Equilateral triangle support area
    static PolygonSupportArea<Scalar> equilateral_triangle(Scalar side_length);

    static PolygonSupportArea<Scalar> axis_aligned_rectangle(Scalar sx,
                                                             Scalar sy);

   private:
    Scalar edge_zmp_constraint(const Vec2<Scalar>& zmp, const Vec2<Scalar>& v1,
                               const Vec2<Scalar>& v2) const;

    Scalar edge_zmp_constraint_scaled(const Vec2<Scalar>& az_zmp,
                                      const Vec2<Scalar>& v1,
                                      const Vec2<Scalar>& v2,
                                      Scalar& az) const {
        Mat2<Scalar> S;
        S << Scalar(0), Scalar(-1), Scalar(1), Scalar(0);

        Vec2<Scalar> normal = S * (v2 - v1);  // inward-facing normal vector
        normal = normal / normal.norm();

        return normal.dot(az_zmp - az * v1);
    }

    std::vector<Vec2<Scalar>> vertices_;
};

}  // namespace upright

#include "impl/support_area.tpp"
