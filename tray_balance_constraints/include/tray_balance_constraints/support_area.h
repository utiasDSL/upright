#pragma once

#include <Eigen/Eigen>

#include "tray_balance_constraints/types.h"
#include "tray_balance_constraints/util.h"

template <typename Scalar>
std::vector<Vec2<Scalar>> cuboid_support_vertices(Scalar sx, Scalar sy);

template <typename Scalar>
std::vector<Vec2<Scalar>> cuboid_support_vertices(
    const Vec3<Scalar>& side_lengths);

template <typename Scalar>
Scalar equilateral_triangle_inscribed_radius(Scalar side_length);

// TODO this is not a fully general implementation, since it doesn't account
// for different orientations
template <typename Scalar>
std::vector<Vec2<Scalar>> equilateral_triangle_support_vertices(
    Scalar side_length);


template <typename Scalar>
struct PolygonEdge {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    PolygonEdge(const Vec2<Scalar>& v1, const Vec2<Scalar>& v2)
        : v1(v1), v2(v2) {
        Mat2<Scalar> S;
        S << Scalar(0), Scalar(1), Scalar(-1), Scalar(0);
        normal = S * (v2 - v1);
    }

    Vec2<Scalar> v1;
    Vec2<Scalar> v2;
    Vec2<Scalar> normal;  // inward-facing
};

template <typename Scalar>
struct PolygonSupportArea {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

   public:
    PolygonSupportArea(const std::vector<Vec2<Scalar>>& vertices)
        : vertices(vertices) {}

    size_t num_constraints() const { return vertices.size(); }

    PolygonSupportArea* clone() const {
        return new PolygonSupportArea(vertices);
    }

    size_t num_parameters() const { return vertices.size() * 2; }

    // Get the edges of the polygon composing the support area
    std::vector<PolygonEdge<Scalar>> edges() const {
        std::vector<PolygonEdge<Scalar>> es;
        for (int i = 0; i < vertices.size() - 1; ++i) {
            es.push_back(PolygonEdge<Scalar>(vertices[i], vertices[i + 1]));
        }
        es.push_back(PolygonEdge<Scalar>(vertices.back(), vertices.front()));
        return es;
    }

    Vector<Scalar> zmp_constraints(const Vec2<Scalar>& zmp,
                                   const Scalar& margin = Scalar(0)) const;

    // Vector<Scalar> zmp_constraints_scaled(const Vec2<Scalar>& az_zmp,
    //                                       Scalar& az) const;

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

    std::vector<Vec2<Scalar>> vertices;

   private:
    Scalar edge_zmp_constraint(const Vec2<Scalar>& zmp, const Vec2<Scalar>& v1,
                               const Vec2<Scalar>& v2,
                               const Scalar& margin) const;

    // Scalar edge_zmp_constraint_scaled(const Vec2<Scalar>& az_zmp,
    //                                   const Vec2<Scalar>& v1,
    //                                   const Vec2<Scalar>& v2, Scalar& az)
    //                                   const;
};

#include "impl/support_area.tpp"
