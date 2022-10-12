#pragma once

#include <numeric>

#include <Eigen/Eigen>

#include "upright_core/dynamics.h"
#include "upright_core/ellipsoid.h"
#include "upright_core/support_area.h"
#include "upright_core/types.h"
#include "upright_core/util.h"

namespace upright {

template <typename Scalar>
class BoundedRigidBody {
   public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    BoundedRigidBody(const Scalar& mass_min, const Scalar& mass_max,
                     const Vec3<Scalar>& radii_of_gyration_min,
                     const Vec3<Scalar>& radii_of_gyration_max,
                     const Ellipsoid<Scalar>& com_ellipsoid)
        : mass_min(mass_min),
          mass_max(mass_max),
          radii_of_gyration_min(radii_of_gyration_min),
          radii_of_gyration_max(radii_of_gyration_max),
          com_ellipsoid(com_ellipsoid) {
        if ((mass_min < 0) || (mass_max < mass_min)) {
            throw std::runtime_error(
                "Masses must be positive and max mass must be >= min mass.");
        }
        if ((radii_of_gyration_min.array() < Scalar(0)).any() ||
            ((radii_of_gyration_max - radii_of_gyration_min).array() <
             Scalar(0))
                .any()) {
            throw std::runtime_error(
                "Radii of gyration must be positive and max radii must be >= "
                "min radii.");
        }
        if (near_zero(mass_max - mass_min)) {
            exact_mass = true;
        }
        if ((radii_of_gyration_max - radii_of_gyration_min).isZero()) {
            exact_radii = true;
        }
    }

    // Sample a random mass and center of mass within the bounds. If boundary =
    // true, then the CoM is generate on the boundary of the bounding
    // ellipsoid.
    std::tuple<Scalar, Vec3<Scalar>> sample(bool boundary = false) {
        Scalar rand = random_scalar<Scalar>();
        Scalar m = mass_min + rand * (mass_max - mass_min);
        // On the boundary, we just choose either mass_min or mass_max randomly
        // for the mass
        if (boundary) {
            if (rand < 0.5) {
                m = mass_min;
            } else {
                m = mass_max;
            }
        }
        Vec3<Scalar> r = com_ellipsoid.sample(boundary);
        return std::tuple<Scalar, Vec3<Scalar>>(m, r);
    }

    // Combined rank of multiple rigid bodies.
    static size_t combined_rank(
        const std::vector<BoundedRigidBody<Scalar>>& bodies) {
        std::vector<Ellipsoid<Scalar>> ellipsoids;
        for (const auto& body : bodies) {
            ellipsoids.push_back(body.com_ellipsoid.scaled(body.mass_min));
            ellipsoids.push_back(body.com_ellipsoid.scaled(body.mass_max));
        }
        return Ellipsoid<Scalar>::combined_rank(ellipsoids);
    }

    size_t num_parameters() const {
        return 2 + 6 + Ellipsoid<Scalar>::num_parameters();
    }

    VecX<Scalar> get_parameters() const {
        VecX<Scalar> p(num_parameters());
        p << mass_min, mass_max, radii_of_gyration_min, radii_of_gyration_max,
            com_ellipsoid.get_parameters();
        return p;
    }

    // Note that this is the squared matrix; i.e., the diagonal contains the
    // squared radii of gyration.
    // TODO add squared to name
    Mat3<Scalar> radii_of_gyration_matrix() const {
        Mat3<Scalar> R = Mat3<Scalar>::Zero();
        R.diagonal() << radii_of_gyration_max;
        return R * R;
    }

    bool is_exact() const {
        return has_exact_mass() && has_exact_radii() && has_exact_com();
    }
    bool has_exact_mass() const { return exact_mass; }
    bool has_exact_radii() const { return exact_radii; }
    bool has_exact_com() const { return com_ellipsoid.rank() == 0; }

    // Create a RigidBody from a parameter vector
    static BoundedRigidBody<Scalar> from_parameters(
        const VecX<Scalar>& parameters, const size_t index = 0) {
        Scalar mass_min = parameters(index);
        Scalar mass_max = parameters(index + 1);

        Vec3<Scalar> radii_of_gyration_min = parameters.segment(index + 2, 3);
        Vec3<Scalar> radii_of_gyration_max = parameters.segment(index + 5, 3);

        const size_t num_ell_params = Ellipsoid<Scalar>::num_parameters();
        Ellipsoid<Scalar> com_ellipsoid = Ellipsoid<Scalar>::from_parameters(
            parameters.segment(index + 8, num_ell_params));
        return BoundedRigidBody(mass_min, mass_max, radii_of_gyration_min,
                                radii_of_gyration_max, com_ellipsoid);
    }

    Scalar mass_min;
    Scalar mass_max;
    Vec3<Scalar> radii_of_gyration_min;
    Vec3<Scalar> radii_of_gyration_max;
    Ellipsoid<Scalar> com_ellipsoid;

   private:
    // True if parameters are certain, false otherwise.
    bool exact_mass = false;
    bool exact_radii = false;
};

template <typename Scalar>
struct BoundedBalancedObject {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    BoundedBalancedObject(const BoundedRigidBody<Scalar>& body,
                          Scalar com_height,
                          const PolygonSupportArea<Scalar>& support_area_min,
                          Scalar r_tau_min, Scalar mu_min)
        : body(body),
          com_height(com_height),
          support_area_min(support_area_min),
          r_tau_min(r_tau_min),
          mu_min(mu_min) {}

    // Copy constructor
    BoundedBalancedObject(const BoundedBalancedObject& other)
        : body(other.body),
          com_height(other.com_height),
          support_area_min(other.support_area_min),
          r_tau_min(other.r_tau_min),
          mu_min(other.mu_min) {}

    // Copy assignment operator
    BoundedBalancedObject<Scalar>& operator=(
        const BoundedBalancedObject& other) {
        return *this;
    }

    ~BoundedBalancedObject() = default;

    size_t num_constraints() const {
        const size_t num_normal = 1;
        const size_t num_fric = 4;
        // const size_t num_zmp = 4 * support_area_min.num_constraints();
        const size_t num_zmp = support_area_min.num_constraints();
        return num_normal + num_fric + num_zmp;
    }

    size_t num_parameters() const {
        return 3 + body.num_parameters() + support_area_min.num_parameters();
    }

    VecX<Scalar> get_parameters() const {
        VecX<Scalar> p(num_parameters());
        p << com_height, r_tau_min, mu_min, body.get_parameters(),
            support_area_min.get_parameters();
        return p;
    }

    Scalar max_com_height() const {
        Vec3<Scalar> z = Vec3<Scalar>::UnitZ();
        Scalar v = z.dot(body.com_ellipsoid.Einv() * z);
        return com_height + sqrt(v);
    }

    Scalar min_com_height() const {
        Vec3<Scalar> z = Vec3<Scalar>::UnitZ();
        Scalar v = z.dot(body.com_ellipsoid.Einv() * z);
        // CoM cannot be below the bottom of the object (it is always >= 0)
        return std::max(com_height - sqrt(v), Scalar(0));
    }

    static BoundedBalancedObject<Scalar> from_parameters(
        const VecX<Scalar>& p) {
        Scalar com_height = p(0);
        Scalar r_tau_min = p(1);
        Scalar mu_min = p(2);

        size_t index = 3;
        auto body = BoundedRigidBody<Scalar>::from_parameters(p, index);

        index += body.num_parameters();
        auto support_area_min =
            PolygonSupportArea<Scalar>::from_parameters(p, index);

        return BoundedBalancedObject<Scalar>(body, com_height, support_area_min,
                                             r_tau_min, mu_min);
    }

    // Cast to another underlying scalar type
    template <typename T>
    BoundedBalancedObject<T> cast() const {
        VecX<Scalar> p = get_parameters();
        return BoundedBalancedObject<T>::from_parameters(p.template cast<T>());
    }

    // Dynamic parameters
    BoundedRigidBody<Scalar> body;

    // Geometry
    // TODO we are assuming object frame is same as EE at the moment
    Scalar com_height;  // nominal CoM height
    PolygonSupportArea<Scalar> support_area_min;

    // Friction
    Scalar r_tau_min;
    Scalar mu_min;
};

}  // namespace upright
