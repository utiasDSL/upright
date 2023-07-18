#pragma once

#include <iostream>

#include <upright_core/types.h>
#include <upright_core/util.h>

#include <Eigen/Eigen>

namespace upright {

template <typename Scalar>
using Mat36 = Eigen::Matrix<Scalar, 3, 6>;

template <typename Scalar>
using Vec6 = Eigen::Matrix<Scalar, 6, 1>;

template <typename Scalar>
using Mat6 = Eigen::Matrix<Scalar, 6, 6>;

template <typename Scalar>
using Mat6_10 = Eigen::Matrix<Scalar, 6, 10>;

template <typename Scalar>
Mat36<Scalar> lift3(const Vec3<Scalar>& x) {
    Mat36<Scalar> L;
    // clang-format off
    L << x(0), x(1), x(2), 0, 0, 0,
         0, x(0), 0, x(1), x(2), 0,
         0, 0, x(0), 0, x(1), x(2);
    // clang-format on
    return L;
}

template <typename Scalar>
Mat6<Scalar> skew6(const Twist<Scalar>& V) {
    Mat3<Scalar> Sv = skew3(V.linear);
    Mat3<Scalar> Sw = skew3(V.angular);
    Mat6<Scalar> S;
    S << Sw, Mat3<Scalar>::Zero(), Sv, Sw;
    return S;
}

template <typename Scalar>
Mat6_10<Scalar> lift6(const Twist<Scalar>& A) {
    Mat6_10<Scalar> L;
    // clang-format off
    L << A.linear, skew3(A.angular), Mat36<Scalar>::Zero(),
         Vec3<Scalar>::Zero(), -skew3(A.linear), lift3(A.angular);
    // clang-format on
    return L;
}

template <typename Scalar>
MatX<Scalar> body_regressor(const Twist<Scalar>& V, const Twist<Scalar>& A) {
    return lift6(A) + skew6(V) * lift6(V);
}

template <typename Scalar>
class RobustBounds {
   public:
    RobustBounds(const MatX<Scalar>& RT, const MatX<Scalar>& F,
                 const MatX<Scalar>& A_ineq)
        : RT_(RT), F_(F), A_ineq_(A_ineq), no_(F.cols() / 6) {}

    Scalar compute_scale(const Vec6<Scalar>& V, const Vec6<Scalar>& G,
                         const Vec6<Scalar>& A) {

        Twist<Scalar> Vt;
        Vt.linear = V.head(3);
        Vt.angular = V.tail(3);
        Twist<Scalar> Gt;
        Gt.linear = G.head(3);
        Gt.angular = G.tail(3);

        const MatX<Scalar> Y0 = body_regressor(Vt, Gt);
        // MatX<Scalar> B = MatX<Scalar>::Zero(F_.rows(), RT_.cols());
        // for (int i = 0; i < no_; ++i) {
        //     B.noalias() += F_.middleCols(i * 6, 6) * Y0 * RT_.middleRows(i * 10, 10);
        // }

        MatX<Scalar> BT = MatX<Scalar>::Zero(RT_.cols(), F_.rows());
        for (int i = 0; i < no_; ++i) {
            BT.noalias() += RT_.middleRows(i * 10, 10).transpose() * Y0.transpose() * F_.middleCols(i * 6, 6).transpose();
        }

        // MatX<Scalar> BT = B.transpose();  // TODO
        // Eigen::Map<VecX<Scalar>> b(BT.data(), BT.size());
        Eigen::Map<Eigen::Array<Scalar, -1, 1>> b(BT.data(), BT.size());

        VecX<Scalar> x = A_ineq_ * A;
        Scalar scale(1.0);
        for (int i = 0; i < x.size(); ++i) {
            if (x(i) > b(i)) {
                if (abs(x(i)) < 1e-8) {
                    throw std::runtime_error("Division by zero!");
                }
                Scalar s = b(i) / x(i);
                if (s < 0) {
                    throw std::runtime_error("Different signs in inequality constraints!");
                }
                if (s < scale) {
                    scale = s;
                }
            }
        }
        return scale;
    }

   private:
    const size_t no_;
    const MatX<Scalar> RT_;
    const MatX<Scalar> F_;
    const MatX<Scalar> A_ineq_;
};

}  // namespace upright
