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
struct SupportAreaBase {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    SupportAreaBase(const Vec2<Scalar>& offset, Scalar margin)
        : offset(offset), margin(margin) {}

    virtual size_t num_constraints() const = 0;

    virtual Vector<Scalar> zmp_constraints(const Vec2<Scalar>& zmp) const = 0;

    // Scaled version of the ZMP constraints that avoids division by az to
    // calculate the ZMP explicitly.
    virtual Vector<Scalar> zmp_constraints_scaled(const Vec2<Scalar>& az_zmp,
                                                  Scalar& az) const = 0;

    virtual SupportAreaBase* clone() const = 0;

    virtual size_t num_parameters() const = 0;
    virtual Vector<Scalar> get_parameters() const = 0;

    /**
     * Offset is the vector pointing from the CoM of the object to the centroid
     * of the support area. ZMP is computed relative to the CoM but to test for
     * inclusion in the support area, we need it relative to the centroid. Thus
     * we have: r^{zmp, centroid} = r^{zmp, CoM} - r^{centroid, CoM}
     */
    Vec2<Scalar> offset;

    Scalar margin;
};

// Circular support area.
template <typename Scalar>
struct CircleSupportArea : public SupportAreaBase<Scalar> {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    CircleSupportArea(Scalar radius, const Vec2<Scalar>& offset, Scalar margin)
        : SupportAreaBase<Scalar>(offset, margin), radius(radius) {}

    CircleSupportArea* clone() const override {
        return new CircleSupportArea(radius, this->offset, this->margin);
    }

    size_t num_parameters() const override { return 2 + 1 + 1; }

    size_t num_constraints() const override { return 1; }

    Vector<Scalar> zmp_constraints(const Vec2<Scalar>& zmp) const override;

    Vector<Scalar> zmp_constraints_scaled(const Vec2<Scalar>& az_zmp,
                                          Scalar& az) const override;

    Vector<Scalar> get_parameters() const override;

    static CircleSupportArea<Scalar> from_parameters(const Vector<Scalar>& p,
                                                     const size_t index = 0);

    Scalar radius;
};

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
struct PolygonSupportArea : public SupportAreaBase<Scalar> {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

   public:
    PolygonSupportArea(const std::vector<Vec2<Scalar>>& vertices,
                       const Vec2<Scalar>& offset, Scalar margin = 0)
        : SupportAreaBase<Scalar>(offset, margin), vertices(vertices) {}

    size_t num_constraints() const override { return vertices.size(); }

    PolygonSupportArea* clone() const override {
        return new PolygonSupportArea(vertices, this->offset, this->margin);
    }

    size_t num_parameters() const override {
        return 2 + 1 + vertices.size() * 2;
    }

    // Get the edges of the polygon composing the support area
    std::vector<PolygonEdge<Scalar>> edges() const {
        std::vector<PolygonEdge<Scalar>> es;
        for (int i = 0; i < vertices.size() - 1; ++i) {
            es.push_back(PolygonEdge<Scalar>(vertices[i], vertices[i + 1]));
        }
        es.push_back(PolygonEdge<Scalar>(vertices.back(), vertices.front()));
        return es;
    }

    Vector<Scalar> zmp_constraints(const Vec2<Scalar>& zmp) const override;

    Vector<Scalar> zmp_constraints_scaled(const Vec2<Scalar>& az_zmp,
                                          Scalar& az) const override;

    Vector<Scalar> get_parameters() const override;

    static PolygonSupportArea<Scalar> from_parameters(const Vector<Scalar>& p,
                                                      const size_t index = 0);

    // Square support area approximation to a circle
    static PolygonSupportArea<Scalar> circle(Scalar radius, Scalar margin = 0);

    // Equilateral triangle support area
    static PolygonSupportArea<Scalar> equilateral_triangle(Scalar side_length,
                                                           Scalar margin = 0);

    static PolygonSupportArea<Scalar> axis_aligned_rectangle(Scalar sx,
                                                             Scalar sy,
                                                             Scalar margin = 0);

    std::vector<Vec2<Scalar>> vertices;

   private:
    Scalar edge_zmp_constraint(const Vec2<Scalar>& zmp, const Vec2<Scalar>& v1,
                               const Vec2<Scalar>& v2) const;

    Scalar edge_zmp_constraint_scaled(const Vec2<Scalar>& az_zmp,
                                      const Vec2<Scalar>& v1,
                                      const Vec2<Scalar>& v2, Scalar& az) const;
};

#include "impl/support_area.tpp"
