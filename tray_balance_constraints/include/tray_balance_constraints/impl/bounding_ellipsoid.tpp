#pragma once

#include <CGAL/Approximate_min_ellipsoid_d.h>
#include <CGAL/Approximate_min_ellipsoid_d_traits_d.h>
#include <CGAL/Cartesian_d.h>
#include <CGAL/MP_Float.h>
#include <CGAL/point_generators_d.h>

namespace upright {

// Compute rank and orthogonal basis vectors for the set of points.
template <typename Scalar>
std::tuple<size_t, Matrix<Scalar>> points_basis(
    const std::vector<Vec3<Scalar>>& points) {
    size_t n = points.size();
    Matrix<Scalar> P = Matrix<Scalar>::Zero(n, 3);
    for (int i = 0; i < n; ++i) {
        P.row(i) = points[i] - points[0];
    }

    Eigen::JacobiSVD<Matrix<Scalar>> svd(
        P, Eigen::ComputeThinU | Eigen::ComputeThinV);
    size_t rank = svd.rank();
    Matrix<Scalar> R = svd.matrixV().leftCols(rank);

    return std::tuple<size_t, Matrix<Scalar>>(rank, R);
}

template <typename Scalar>
Ellipsoid<Scalar> Ellipsoid<Scalar>::bounding(
    const std::vector<Vec3<Scalar>>& points, const Scalar eps) {
    using Kernel = CGAL::Cartesian_d<Scalar>;
    using Traits =
        CGAL::Approximate_min_ellipsoid_d_traits_d<Kernel, CGAL::MP_Float>;
    using AME = CGAL::Approximate_min_ellipsoid_d<Traits>;

    Matrix<Scalar> R;
    size_t rank;
    std::tie(rank, R) = points_basis(points);

    if (rank == 0) {
        // All the points are the same and we needn't do anything
        return Ellipsoid<Scalar>::point(points[0]);
    } else if (rank == 1) {
        // The points form a line along the direction R. Sort to find the
        // points farthest apart on the line (TODO could probably do this more
        // efficiently).
        std::vector<Vec3<Scalar>> sorted_points = points;
        std::sort(sorted_points.begin(), sorted_points.end(),
                  [&](Vec3<Scalar> a, Vec3<Scalar> b) -> bool {
                      // R is has a single column (i.e. it is a vector), but
                      // the compiler doesn't know this, so we explicitly
                      // access the column
                      return R.col(0).dot(a - b) < 0;
                  });
        return Ellipsoid<Scalar>::segment(sorted_points.front(),
                                          sorted_points.back());
    }

    // Project points onto basis and convert to CGAL format
    std::vector<typename Traits::Point> ps;
    for (int i = 0; i < points.size(); ++i) {
        Vec3<Scalar> p = R.transpose() * points[i];
        ps.push_back(
            typename Traits::Point(rank, p.data(), p.data() + p.size()));
    }

    // Compute the bounding ellipsoid for projected points
    Traits traits;
    AME ame(eps, ps.begin(), ps.end(), traits);

    // Copy center point
    Vector<Scalar> c = Vector<Scalar>::Zero(rank);
    auto c_it = ame.center_cartesian_begin();
    for (int i = 0; i < rank; ++i) {
        c(i) = *c_it++;
    }
    Vec3<Scalar> center = R * c;

    // Copy axes half lengths and directions
    std::vector<Scalar> half_lengths_vec;
    std::vector<Vec3<Scalar>> directions_vec;
    auto a_it = ame.axes_lengths_begin();
    for (int i = 0; i < rank; ++i) {
        half_lengths_vec.push_back(*a_it++);
        auto d_it = ame.axis_direction_cartesian_begin(i);
        Vector<Scalar> v = Vector<Scalar>::Zero(rank);
        for (int j = 0; j < rank; ++j) {
            v(j) = *d_it++;
        }
        directions_vec.push_back(R * v);
    }

    return Ellipsoid<Scalar>(center, half_lengths_vec, directions_vec);
}

}  // namespace upright
