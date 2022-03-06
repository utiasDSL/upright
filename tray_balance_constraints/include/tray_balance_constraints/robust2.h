#pragma once

#include <Eigen/Eigen>

#include "tray_balance_constraints/dynamics.h"
#include "tray_balance_constraints/support_area.h"
#include "tray_balance_constraints/types.h"

template <typename Scalar>
struct Ellipsoid {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    Ellipsoid(const Vec3<Scalar>& center, const Vec3<Scalar>& half_lengths;
              const Mat3<Scalar>& directions, const Scalar eps = 1e-4)
        : center(center), half_lengths(half_lengths), directions(directions) {
        Dinv = Mat3<Scalar>::Zero();
        for (int i = 0; i < 3; ++i) {
            Dinv(i, i) = half_lengths(i) * half_lengths(i);
        }

        // we set entries of D to zero if the half length approaches zero
        D = Mat3<Scalar>::Zero();
        for (int i = 0; i < 3; ++i) {
            if (half_lengths(i) >= eps) {
                D(i, i) = 1. / Dinv(i, i);
            } else {
                rank = i;
                break;
            }
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

    uint32_t rank = 3;
};
