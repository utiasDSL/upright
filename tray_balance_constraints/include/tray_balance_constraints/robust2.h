#pragma once

#include <Eigen/Eigen>

#include "tray_balance_constraints/dynamics.h"
#include "tray_balance_constraints/support_area.h"
#include "tray_balance_constraints/types.h"

template <typename Scalar>
struct Ellipsoid {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    Ellipsoid(const Vec3<Scalar>& center,
              const std::vector<Scalar>& half_lengths_vec;
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

        Dinv = Mat3<Scalar>::Zero();
        for (int i = 0; i < 3; ++i) {
            Dinv(i, i) = half_lengths(i) * half_lengths(i);
        }

        // rank = 3 - n means n half lengths are zero. We set entries of D to
        // zero if the half length is zero.
        D = Mat3<Scalar>::Zero();
        for (int i = 0; i < rank; ++i) {
            D(i, i) = 1. / Dinv(i, i);
        }
    }

    Mat3<Scalar> Einv() const {
        return directions * Dinv * directions.transpose();
    }

    Mat3<Scalar> E() const { return directions * D * directions.tranpose(); }

    Vec3<Scalar> center;
    Vec3<Scalar> half_lengths;  // should be in decreasing order
    Mat3<Scalar> directions;    // directions stored as columns

    Mat3<Scalar> Dinv;  // diagonal matrix of squared half lengths
    Mat3<Scalar> D;  // inverse of D, except zero where half lengths near zero

    size_t rank;
};
