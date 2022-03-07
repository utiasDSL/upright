#pragma once

#include <Eigen/Eigen>

#include "tray_balance_constraints/dynamics.h"
#include "tray_balance_constraints/support_area.h"
#include "tray_balance_constraints/types.h"

template <typename Scalar>
Scalar random_scalar() {
    Scalar x = Eigen::Matrix<Scalar, 1, 1>::Random()(0);
    return 0.5 * (x + 1.0);
}

template <typename Scalar>
struct Ellipsoid {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    Ellipsoid(const Vec3<Scalar>& center,
              const std::vector<Scalar>& half_lengths_vec,
              const std::vector<Vec3<Scalar>>& directions_vec,
              const size_t rank)
        : center(center), rank(rank) {
        // Sort indices of half lengths such that half lengths are in
        // decreasing order.
        std::vector<int> indices(3);
        std::iota(indices.begin(), indices.end(), 0);
        std::sort(indices.begin(), indices.end(), [&](int i, int j) -> bool {
            return half_lengths_vec[i] > half_lengths_vec[j];
        });

        // Construct properly-ordered Eigen versions.
        for (int i = 0; i < 3; ++i) {
            half_lengths(i) = half_lengths_vec[indices[i]];
            directions.col(i) = directions_vec[indices[i]];
        }

        init();
    }

    Mat3<Scalar> Einv() const {
        return directions * Dinv * directions.transpose();
    }

    Mat3<Scalar> E() const { return directions * D * directions.transpose(); }

    // Return a new ellipsoid that is scaled by the scalar a.
    Ellipsoid<Scalar> scaled(Scalar a) const {
        if (abs(a) < 1e-8) {
            return Ellipsoid<Scalar>::zero(center);
        }
        return Ellipsoid<Scalar>(a * center, a * half_lengths, directions, rank);
    }

    // Constructor a rank-zero ellipsoid, which is just a point.
    static Ellipsoid<Scalar> zero(const Vec3<Scalar>& center)
        : Ellipsoid<Scalar>(center, Vec3<Scalar>::Zero(), Mat3<Scalar>::Zero(),
                            Scalar(0)) {}

    // Randomly sample a point from the ellipsoid. Set boundary = true to only
    // sample points from the boundary of the ellipsoid.
    Vec3<Scalar> sample(bool boundary = false) const {
        Vec3<Scalar> direction = Vec3<Scalar>::Random().normalized();
        Scalar max_dist_squared = direction.transpose() * E() * direction;

        if (max_dist_squared < 1e-8) {
            return center;
        }

        // If we are on the boundary, then just return the boundary point in
        // the direction. Otherwise, generate a random distance between 0 and
        // max_dist to return.
        Scalar max_dist = sqrt(max_dist_squared);
        Scalar dist = max_dist;
        if (!boundary) {
            dist = max_dist * random_scalar();
        }
        return center + dist * direction;
    }

    // Compute the rank of multiple combined ellipsoids. This is the dimension
    // of the space spanned by the different ellipsoids, taking into account
    // their center points.
    static size_t combined_rank(
        const std::vector<Ellipsoid<Scalar>>& ellipsoids) {
        Matrix<Scalar> A(4 * ellipsoids.size(), 3);

        Vec3<Scalar> c = ellipsoids[0].center;

        for (int i = 0; i < ellipsoids.size(); ++i) {
            A.middleRows<3>(i * 4) = ellipsoids[i].Einv();
            A.row(i * 4 + 3) = ellipsoids[i].center - c;
        }
        return A.fullPivHouseholderQr().rank();
    }

    Vec3<Scalar> center;
    Vec3<Scalar> half_lengths;  // should be in decreasing order
    Mat3<Scalar> directions;    // directions stored as columns

    Mat3<Scalar> Dinv;  // diagonal matrix of squared half lengths
    Mat3<Scalar> D;  // inverse of D, except zero where half lengths near zero

    size_t rank;

   private:
    // Constructor for when half_lengths and directions are already correctly
    // ordered. Only allowed to be called internally for now.
    Ellipsoid(const Vec3<Scalar>& center, const Vec3<Scalar>& half_lengths,
              const Mat3<Scalar>& directions, const size_t rank)
        : center(center),
          half_lengths(half_lengths),
          directions(directions),
          rank(rank) {
        init();
    }

    void init() {
        Dinv = Mat3<Scalar>::Zero();
        for (int i = 0; i < rank; ++i) {
            Dinv(i, i) = half_lengths(i) * half_lengths(i);
        }

        // rank = 3 - n means n half lengths are zero. We set entries of D to
        // zero if the half length is zero.
        D = Mat3<Scalar>::Zero();
        for (int i = 0; i < rank; ++i) {
            D(i, i) = 1. / Dinv(i, i);
        }
    }
};

template <typename Scalar>
struct RigidBodyBounds {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    RigidBodyBounds(const Scalar& mass_min, const Scalar& mass_max,
                    const Scalar& r_gyr, const Ellipsoid<Scalar>& com_ellipsoid)
        : mass_min(mass_min),
          mass_max(mass_max),
          r_gyr(r_gyr),
          com_ellipsoid(com_ellipsoid) {}

    // Compose multiple rigid bodies into one.
    // static RigidBodyBounds<Scalar> compose(
    //     const std::vector<RigidBodyBounds<Scalar>>& bodies);

    // Create a RigidBody from a parameter vector
    // TODO
    // static RigidBodyBounds<Scalar> from_parameters(
    //     const Vector<Scalar>& parameters, const size_t index = 0);

    // Sample a random mass and center of mass within the bounds. If boundary =
    // true, then the CoM is generate on the boundary of the bounding
    // ellipsoid.
    std::tuple<Scalar, Vec3<Scalar>> sample(bool boundary = false) {
        Scalar m = mass_min + random_scalar() * (mass_max - mass_min);
        Vec3<Scalar> r = com_ellipsoid.sample(boundary);
        return std::tuple<Scalar, Vec3<Scalar>>(m, r);
    }

    // Combined rank of multiple rigid bodies.
    static size_t combined_rank(
        const std::vector<RigidBodyBounds<Scalar>>& bodies) {
        std::vector<Ellipsoid<Scalar>> ellipsoids;
        for (const auto& body : bodies) {
            ellipsoids.push_back(body.com_ellipsoid.scaled(body.mass_min));
            ellipsoids.push_back(body.com_ellipsoid.scaled(body.mass_max));
        }
        return Ellipsoid<Scalar>::combined_rank(ellipsoids);
    }

    size_t num_parameters() const { return 2 + 1 /*+ TODO # for ellipsoid */; }

    Vector<Scalar> get_parameters() const;

    Scalar mass_min;
    Scalar mass_max;
    Scalar r_gyr;
    Ellipsoid<Scalar> com_ellipsoid;
}
