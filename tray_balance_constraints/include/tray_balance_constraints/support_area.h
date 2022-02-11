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

    size_t num_constraints() const override { return 1; }

    Vector<Scalar> zmp_constraints(const Vec2<Scalar>& zmp) const override {
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

    Vector<Scalar> zmp_constraints_scaled(const Vec2<Scalar>& az_zmp,
                                          Scalar& az) const override {
        // e is scaled by az
        Vec2<Scalar> e = az_zmp - az * this->offset;

        Vector<Scalar> constraints(num_constraints());
        constraints << squared(az * (radius - this->margin)) - e.dot(e);
        return constraints;
    }

    CircleSupportArea* clone() const override {
        return new CircleSupportArea(radius, this->offset, this->margin);
    }

    size_t num_parameters() const override { return 2 + 1 + 1; }

    Vector<Scalar> get_parameters() const override {
        Vector<Scalar> p(num_parameters());
        p << this->offset, this->margin, radius;
        return p;
    }

    static CircleSupportArea<Scalar> from_parameters(const Vector<Scalar>& p,
                                                     const size_t index = 0) {
        size_t n = p.size() - index;
        if (n < 4) {
            throw std::runtime_error("[CircleSupportArea] Parameter vector is wrong size.");
        }

        Vec2<Scalar> offset = p.template segment<2>(index);
        Scalar margin = p(index + 2);
        Scalar radius = p(index + 3);
        return CircleSupportArea<Scalar>(radius, offset, margin);
    }

    Scalar radius;
};

template <typename Scalar>
struct PolygonSupportArea : public SupportAreaBase<Scalar> {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

   public:
    PolygonSupportArea(const std::vector<Vec2<Scalar>>& vertices,
                       const Vec2<Scalar>& offset, Scalar margin = 0)
        : SupportAreaBase<Scalar>(offset, margin), vertices(vertices) {}

    size_t num_constraints() const override { return vertices.size(); }

    Vector<Scalar> zmp_constraints(const Vec2<Scalar>& zmp) const override {
        const size_t n = num_constraints();
        Vector<Scalar> constraints(n);
        for (int i = 0; i < n - 1; ++i) {
            constraints(i) = edge_zmp_constraint(zmp - this->offset,
                                                 vertices[i], vertices[i + 1]);
        }
        constraints(n - 1) = edge_zmp_constraint(zmp - this->offset,
                                                 vertices[n - 1], vertices[0]);
        return constraints;
    }

    Vector<Scalar> zmp_constraints_scaled(const Vec2<Scalar>& az_zmp,
                                          Scalar& az) const override {
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

    PolygonSupportArea* clone() const override {
        return new PolygonSupportArea(vertices, this->offset, this->margin);
    }

    size_t num_parameters() const override {
        return 2 + 1 + vertices.size() * 2;
    }

    Vector<Scalar> get_parameters() const override {
        Vector<Scalar> p(num_parameters());
        Vector<Scalar> v(vertices.size() * 2);
        for (int i = 0; i < vertices.size(); ++i) {
            v.segment(i * 2, 2) = vertices[i];
        }
        p << this->offset, this->margin, v;
        return p;
    }

    static PolygonSupportArea<Scalar> from_parameters(const Vector<Scalar>& p,
                                                      const size_t index = 0) {
        // Need at least three vertices in the support area
        size_t n = p.size() - index;
        if (n < 3 + 2 * 3) {
            throw std::runtime_error("[PolygonSupportArea] Parameter vector is wrong size.");
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
    static PolygonSupportArea<Scalar> circle(Scalar radius, Scalar margin = 0) {
        Scalar side_length = Scalar(sqrt(2.0)) * radius;
        Vec2<Scalar> offset = Vec2<Scalar>::Zero();
        std::vector<Vec2<Scalar>> vertices =
            cuboid_support_vertices(side_length, side_length);
        return PolygonSupportArea<Scalar>(vertices, offset, margin);
    }

    // Equilateral triangle support area
    static PolygonSupportArea<Scalar> equilateral_triangle(Scalar side_length,
                                                           Scalar margin = 0) {
        Vec2<Scalar> offset = Vec2<Scalar>::Zero();
        std::vector<Vec2<Scalar>> vertices =
            equilateral_triangle_support_vertices(side_length);
        return PolygonSupportArea<Scalar>(vertices, offset, margin);
    }

    static PolygonSupportArea<Scalar> axis_aligned_rectangle(Scalar sx,
                                                             Scalar sy,
                                                             Scalar margin = 0) {
        Vec2<Scalar> offset = Vec2<Scalar>::Zero();
        std::vector<Vec2<Scalar>> vertices = cuboid_support_vertices(sx, sy);
        return PolygonSupportArea<Scalar>(vertices, offset, margin);
    }

    std::vector<Vec2<Scalar>> vertices;

   private:
    Scalar edge_zmp_constraint(const Vec2<Scalar>& zmp, const Vec2<Scalar>& v1,
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

    Scalar edge_zmp_constraint_scaled(const Vec2<Scalar>& az_zmp,
                                      const Vec2<Scalar>& v1,
                                      const Vec2<Scalar>& v2,
                                      Scalar& az) const {
        Mat2<Scalar> S;
        S << Scalar(0), Scalar(1), Scalar(-1), Scalar(0);

        Vec2<Scalar> normal = S * (v2 - v1);  // inward-facing normal vector
        normal.normalize();

        // TODO margin should be scaled up for scaled version
        return -(az_zmp - az * v1).dot(normal) - this->margin;
    }
};
