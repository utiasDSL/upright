#pragma once

#include <Eigen/Eigen>

#include "tray_balance_constraints/types.h"
#include "tray_balance_constraints/util.h"

template <typename Scalar>
struct Ellipsoid {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    Ellipsoid(const Vec3<Scalar>& center,
              const std::vector<Scalar>& half_lengths_vec,
              const std::vector<Vec3<Scalar>>& directions_vec)
        : center_(center) {
        // Sort indices of half lengths such that half lengths are in
        // decreasing order.
        std::vector<int> indices(half_lengths_vec.size());
        std::iota(indices.begin(), indices.end(), 0);
        std::sort(indices.begin(), indices.end(), [&](int i, int j) -> bool {
            return half_lengths_vec[i] > half_lengths_vec[j];
        });

        // Construct properly-ordered Eigen versions.
        half_lengths_ = Vec3<Scalar>::Zero();
        directions_ = Mat3<Scalar>::Zero();
        rank_ = 0;
        for (int i = 0; i < std::min<size_t>(3, half_lengths_vec.size()); ++i) {
            Scalar hl = half_lengths_vec[indices[i]];
            if (near_zero(hl)) {
                break;
            } else if (hl < Scalar(0)) {
                throw std::runtime_error("Ellipsoid half lengths must be non-negative.");
            }
            half_lengths_(i) = hl;
            directions_.col(i) = directions_vec[indices[i]];
            rank_++;
        }

        // Fill remaining directions with the nullspace vectors
        if (rank_ < 3) {
            Matrix<Scalar> kernel =
                directions_.leftCols(rank_).transpose().fullPivLu().kernel();
            directions_.rightCols(3 - rank_) = kernel;
        }

        init();
    }

    Mat3<Scalar> Einv() const {
        return directions_ * Dinv * directions_.transpose();
    }

    // TODO possibly rename to .matrix() and remove Einv method
    Mat3<Scalar> E() const { return directions_ * D * directions_.transpose(); }

    Matrix<Scalar> rangespace() const { return directions_.leftCols(rank_); }

    Matrix<Scalar> nullspace() const {
        return directions_.rightCols(3 - rank_);
    }

    // Return a new ellipsoid that is scaled by the scalar a.
    Ellipsoid<Scalar> scaled(Scalar a) const {
        if (abs(a) < NEAR_ZERO) {
            return Ellipsoid<Scalar>::point(center_);
        }
        return Ellipsoid<Scalar>(a * center_, a * half_lengths_, directions_,
                                 rank_);
    }

    // Construct a rank-zero ellipsoid, which is just a point.
    static Ellipsoid<Scalar> point(const Vec3<Scalar>& center) {
        return Ellipsoid<Scalar>(center, Vec3<Scalar>::Zero(),
                                 Mat3<Scalar>::Zero(), 0);
    }

    // Construct a rank-one ellipsoid, which is a line segment with vertices v1
    // and v2.
    static Ellipsoid<Scalar> segment(const Vec3<Scalar>& v1,
                                     const Vec3<Scalar>& v2) {
        Vec3<Scalar> center = 0.5 * (v1 + v2);
        Scalar half_length = (v2 - center).norm();
        Vec3<Scalar> direction = (v2 - center) / half_length;
        return Ellipsoid<Scalar>(center, {half_length}, {direction});
    }

    // Returns true if the ellipsoid contains the point x
    bool contains(const Vec3<Scalar>& x) {
        Vec3<Scalar> delta = x - center();
        bool outside_nullspace = true;
        if (rank_ < 3) {
            Vector<Scalar> nullspace_projection =
                nullspace().transpose() * delta;
            outside_nullspace = nullspace_projection.isZero(NEAR_ZERO);
        }
        bool inside_rangespace =
            (delta.transpose() * E() * delta <= 1.0 + NEAR_ZERO);
        return outside_nullspace && inside_rangespace;
    }

    // Randomly sample a point from the ellipsoid. Set boundary = true to only
    // sample points from the boundary of the ellipsoid.
    Vec3<Scalar> sample(bool boundary = false) const {
        // Generate a random unit vector in the range space of the ellipsoid
        Vector<Scalar> rand = Vector<Scalar>::Random(rank_);
        Vec3<Scalar> direction =
            (directions_.leftCols(rank_) * rand).normalized();

        // Inverse of squared distance to edge of ellipsoid along the random
        // direction
        Scalar max_dist_inv_squared = direction.transpose() * E() * direction;
        if (max_dist_inv_squared < NEAR_ZERO) {
            return center();
        }

        // If we are on the boundary, then just return the boundary point in
        // the direction. Otherwise, generate a random distance between 0 and
        // max_dist to return.
        Scalar max_dist = 1. / sqrt(max_dist_inv_squared);
        Scalar dist = max_dist;
        if (!boundary) {
            dist = max_dist * random_scalar<Scalar>();
        }
        return center() + dist * direction;
    }

    // Compute the rank of multiple combined ellipsoids. This is the dimension
    // of the space spanned by the different ellipsoids, taking into account
    // their center points.
    static size_t combined_rank(
        const std::vector<Ellipsoid<Scalar>>& ellipsoids) {
        Matrix<Scalar> A(4 * ellipsoids.size(), 3);

        Vec3<Scalar> c = ellipsoids[0].center();

        for (int i = 0; i < ellipsoids.size(); ++i) {
            A.template middleRows<3>(i * 4) = ellipsoids[i].Einv();
            A.row(i * 4 + 3) = ellipsoids[i].center() - c;
        }
        return A.fullPivHouseholderQr().rank();
    }

    // Get the rank of the ellipsoid
    size_t rank() const { return rank_; }

    // Get the center point of the ellipsoid
    Vec3<Scalar> center() const { return center_; }

    // Get the half lengths of the ellipsoid
    Vec3<Scalar> half_lengths() const { return half_lengths_; }

    // Get the unit vectors representing the semi-major axes
    Mat3<Scalar> directions() const { return directions_; }

    static const size_t num_parameters() { return 3 + 3 + 9; }

    Vector<Scalar> get_parameters() const {
        Vector<Scalar> p(num_parameters());
        Vector<Scalar> directions_vec(Eigen::Map<const Vector<Scalar>>(
            directions_.data(), directions_.size()));
        p << center_, half_lengths_, directions_vec;
        return p;
    }

    static Ellipsoid<Scalar> from_parameters(const Vector<Scalar>& parameters) {
        Vec3<Scalar> center = parameters.template head<3>();
        Vec3<Scalar> half_lengths = parameters.template segment<3>(3);
        Vector<Scalar> directions_vec = parameters.template segment<9>(6);
        Mat3<Scalar> directions(
            Eigen::Map<Mat3<Scalar>>(directions_vec.data(), 3, 3));

        size_t rank = 3;
        for (int i = 0; i < 3; ++i) {
            if (half_lengths(i) < NEAR_ZERO) {
                rank = i;
                break;
            }
        }
        return Ellipsoid<Scalar>(center, half_lengths, directions, rank);
    }

    static Ellipsoid<Scalar> bounding(const std::vector<Vec3<Scalar>>& points,
                                      const Scalar eps);

   private:
    // Constructor for when half_lengths and directions are already correctly
    // ordered. Only allowed to be called internally for now.
    Ellipsoid(const Vec3<Scalar>& center, const Vec3<Scalar>& half_lengths,
              const Mat3<Scalar>& directions, const size_t rank)
        : center_(center),
          half_lengths_(half_lengths),
          directions_(directions),
          rank_(rank) {
        init();
    }

    void init() {
        Dinv = Mat3<Scalar>::Zero();
        for (int i = 0; i < rank_; ++i) {
            Dinv(i, i) = half_lengths_(i) * half_lengths_(i);
        }

        // rank = 3 - n means n half lengths are zero. We set entries of D to
        // zero if the half length is zero.
        D = Mat3<Scalar>::Zero();
        for (int i = 0; i < rank_; ++i) {
            D(i, i) = 1. / Dinv(i, i);
        }
    }

    Vec3<Scalar> center_;
    Vec3<Scalar> half_lengths_;  // should be in decreasing order
    Mat3<Scalar> directions_;    // directions stored as columns

    Mat3<Scalar> Dinv;  // diagonal matrix of squared half lengths
    Mat3<Scalar> D;  // inverse of D, except zero where half lengths near zero

    size_t rank_;
};

#include "impl/bounding_ellipsoid.tpp"
