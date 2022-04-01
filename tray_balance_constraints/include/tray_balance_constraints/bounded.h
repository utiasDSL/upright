#pragma once

#include <Eigen/Eigen>

#include "tray_balance_constraints/dynamics.h"
#include "tray_balance_constraints/ellipsoid.h"
#include "tray_balance_constraints/support_area.h"
#include "tray_balance_constraints/types.h"
#include "tray_balance_constraints/util.h"

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

    Vector<Scalar> get_parameters() const {
        Vector<Scalar> p(num_parameters());
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
        const Vector<Scalar>& parameters, const size_t index = 0) {
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
        const size_t num_zmp = 4 * support_area_min.num_constraints();
        return num_normal + num_fric + num_zmp;
    }

    size_t num_parameters() const {
        return 3 + body.num_parameters() + support_area_min.num_parameters();
    }

    Vector<Scalar> get_parameters() const {
        Vector<Scalar> p(num_parameters());
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
        const Vector<Scalar>& p) {
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
        Vector<Scalar> p = get_parameters();
        return BoundedBalancedObject<T>::from_parameters(p.template cast<T>());
    }

    // Dynamic parameters
    BoundedRigidBody<Scalar> body;

    // Geometry
    Scalar com_height;  // nominal CoM height
    PolygonSupportArea<Scalar> support_area_min;

    // Friction
    Scalar r_tau_min;
    Scalar mu_min;
};

template <typename Scalar>
Scalar max_beta_projection_approx(const Vec3<Scalar>& p, const Mat3<Scalar>& R2,
                                  const Mat3<Scalar>& C_ew,
                                  const Vec3<Scalar>& angular_vel,
                                  const Vec3<Scalar>& angular_acc,
                                  const Scalar& eps) {
    return epsilon_norm<Scalar>(p.cross(C_ew * angular_vel), eps) *  // TODO note I added C_ew here
               epsilon_norm<Scalar>(R2 * C_ew * angular_vel, eps) +
           epsilon_norm<Scalar>(p, eps) *
               epsilon_norm<Scalar>(R2 * C_ew * angular_acc, eps);
}

// This should only be called if the radii of gyration of the body are exact
template <typename Scalar>
Scalar max_beta_projection_exact(const Vec3<Scalar>& p, const Mat3<Scalar>& R2,
                                 const Mat3<Scalar>& C_ew,
                                 const Vec3<Scalar>& angular_vel,
                                 const Vec3<Scalar>& angular_acc) {
    return p.cross(C_ew * angular_vel).dot(R2 * C_ew * angular_vel) +
           p.dot(R2 * C_ew * angular_acc);
}

enum class OptType { Min, Max };

template <typename Scalar>
Scalar optimize_linear_st_ellipsoid(const Vec3<Scalar>& a, const Scalar& b,
                                    const Ellipsoid<Scalar>& ellipsoid,
                                    const Scalar& eps, OptType type) {
    Scalar v = sqrt(a.dot(ellipsoid.Einv() * a) + eps);
    if (type == OptType::Min) {
        v = -v;
    }
    return a.dot(ellipsoid.center()) + v + b;
}

// Compute the minimum value of p.T * alpha where the CoM is constrained to lie
// inside of an ellipsoid.
template <typename Scalar>
Scalar opt_alpha_projection(const Vec3<Scalar>& p, const Mat3<Scalar>& ddC_we,
                            const Mat3<Scalar>& C_ew,
                            const Vec3<Scalar>& linear_acc,
                            const Vec3<Scalar>& g,
                            const BoundedBalancedObject<Scalar>& object,
                            Scalar eps, OptType type) {
    Vec3<Scalar> a = ddC_we.transpose() * C_ew.transpose() * p;
    Scalar b = p.transpose() * C_ew * (linear_acc - g);
    return optimize_linear_st_ellipsoid(a, b, object.body.com_ellipsoid, eps,
                                        type);
}

template <typename Scalar>
Vector<Scalar> bounded_contact_constraint(
    const Mat3<Scalar>& ddC_we, const Mat3<Scalar>& C_ew,
    const Vec3<Scalar>& linear_acc, const Vec3<Scalar>& g,
    const BoundedBalancedObject<Scalar>& object, Scalar eps) {
    Vec3<Scalar> z = Vec3<Scalar>::UnitZ();
    Vector<Scalar> contact_constraint(1);
    contact_constraint << opt_alpha_projection(z, ddC_we, C_ew, linear_acc, g,
                                               object, eps, OptType::Min);
    return contact_constraint;
}

template <typename Scalar>
Mat3<Scalar> rotation_matrix_second_derivative(
    const Mat3<Scalar>& C, const Vec3<Scalar>& angular_vel,
    const Vec3<Scalar>& angular_acc) {
    Mat3<Scalar> S_angular_vel = skew3<Scalar>(angular_vel);
    Mat3<Scalar> S_angular_acc = skew3<Scalar>(angular_acc);
    return (S_angular_acc + S_angular_vel * S_angular_vel) * C;
}

template <typename Scalar>
Vector<Scalar> bounded_friction_constraint(
    const Mat3<Scalar>& ddC_we, const Mat3<Scalar>& C_ew,
    const Vec3<Scalar>& angular_vel, const Vec3<Scalar>& linear_acc,
    const Vec3<Scalar>& angular_acc, const Vec3<Scalar>& g,
    const BoundedBalancedObject<Scalar>& object, Scalar eps) {
    // Max torque about z-axis
    Vec3<Scalar> z = Vec3<Scalar>::UnitZ();
    Mat3<Scalar> R2 = object.body.radii_of_gyration_matrix();
    Scalar beta_z_max;
    if (object.body.has_exact_radii()) {
        beta_z_max = max_beta_projection_exact(z, R2, C_ew, angular_vel, angular_acc);
    } else {
        beta_z_max = max_beta_projection_approx(z, R2, C_ew, angular_vel, angular_acc, eps);
    }

    // clang-format off
    Vec3<Scalar> c1, c2, c3, c4;
    c1 << Scalar(-1), Scalar(-1), object.mu_min;
    c2 << Scalar(1),  Scalar(-1), object.mu_min;
    c3 << Scalar(-1), Scalar(1),  object.mu_min;
    c4 << Scalar(1),  Scalar(1),  object.mu_min;
    // clang-format on

    Scalar min1 = opt_alpha_projection(c1, ddC_we, C_ew, linear_acc, g, object,
                                       eps, OptType::Min);
    Scalar min2 = opt_alpha_projection(c2, ddC_we, C_ew, linear_acc, g, object,
                                       eps, OptType::Min);
    Scalar min3 = opt_alpha_projection(c3, ddC_we, C_ew, linear_acc, g, object,
                                       eps, OptType::Min);
    Scalar min4 = opt_alpha_projection(c4, ddC_we, C_ew, linear_acc, g, object,
                                       eps, OptType::Min);

    Vector<Scalar> friction_constraint = Vector<Scalar>::Ones(4);

    // TODO hopefully come up with a more elegant way to handle the exact case
    if (object.body.has_exact_radii()) {
        // For the exact beta, we need to handle the sign but can get away with
        // sqrt(x**2 + eps), since the max and min values cannot be different
        Scalar beta_positive = sqrt(squared(beta_z_max) + eps);
        friction_constraint(0) = min1 - beta_positive / object.r_tau_min;
        friction_constraint(1) = min2 - beta_positive / object.r_tau_min;
        friction_constraint(2) = min3 - beta_positive / object.r_tau_min;
        friction_constraint(3) = min4 - beta_positive / object.r_tau_min;
    } else {
        // Approximate beta_z_max is always non-negative, so we don't need to
        // handle different signs here.
        friction_constraint(0) = min1 - beta_z_max / object.r_tau_min;
        friction_constraint(1) = min2 - beta_z_max / object.r_tau_min;
        friction_constraint(2) = min3 - beta_z_max / object.r_tau_min;
        friction_constraint(3) = min4 - beta_z_max / object.r_tau_min;
    }

    return friction_constraint;
}

template <typename Scalar>
Vector<Scalar> bounded_zmp_constraint(
    const Mat3<Scalar>& ddC_we, const Mat3<Scalar>& C_ew,
    const Vec3<Scalar>& angular_vel, const Vec3<Scalar>& linear_acc,
    const Vec3<Scalar>& angular_acc, const Vec3<Scalar>& g,
    const BoundedBalancedObject<Scalar>& object, Scalar eps) {
    // Four constraints per edge
    std::vector<PolygonEdge<Scalar>> edges = object.support_area_min.edges();
    Vector<Scalar> zmp_constraints(edges.size() * 4);

    Vec3<Scalar> z = Vec3<Scalar>::UnitZ();
    Eigen::Matrix<Scalar, 2, 3> S;
    S << Scalar(0), Scalar(1), Scalar(0), Scalar(-1), Scalar(0), Scalar(0);
    Mat3<Scalar> R2 = object.body.radii_of_gyration_matrix();

    for (int i = 0; i < edges.size(); ++i) {
        Vec3<Scalar> normal3;
        normal3 << edges[i].normal, Scalar(0);
        Scalar alpha_xy_max = opt_alpha_projection(
            normal3, ddC_we, C_ew, linear_acc, g, object, eps, OptType::Max);

        // NOTE: very important to use a small epsilon here
        // TODO: ideally, we could handle this at a lower level in CppAD
        Scalar r_xy_max = optimize_linear_st_ellipsoid(
            normal3,
            -edges[i].normal.dot(object.body.com_ellipsoid.center().head(2) +
                                 edges[i].v1),
            object.body.com_ellipsoid, Scalar(1e-6), OptType::Max);

        Vec3<Scalar> p = S.transpose() * edges[i].normal;
        Scalar beta_xy_max;
        if (object.body.has_exact_radii()) {
            beta_xy_max = max_beta_projection_exact(p, R2, C_ew, angular_vel, angular_acc);
        } else {
            beta_xy_max = max_beta_projection_approx(p, R2, C_ew, angular_vel, angular_acc, Scalar(1e-6));
        }

        Scalar alpha_z_min = opt_alpha_projection(z, ddC_we, C_ew, linear_acc,
                                                  g, object, eps, OptType::Min);
        Scalar alpha_z_max = opt_alpha_projection(z, ddC_we, C_ew, linear_acc,
                                                  g, object, eps, OptType::Max);

        if (object.body.has_exact_radii()) {
            // When radii of gyration are exact, we remove the negative sign
            // because we want to use the exact value of beta, rather than an
            // upper bound. TODO as with the friction case, this can be handled
            // better
            zmp_constraints(i * 4) = beta_xy_max -
                                     object.max_com_height() * alpha_xy_max -
                                     alpha_z_max * r_xy_max;
            zmp_constraints(i * 4 + 1) = beta_xy_max -
                                         object.min_com_height() * alpha_xy_max -
                                         alpha_z_max * r_xy_max;
            zmp_constraints(i * 4 + 2) = beta_xy_max -
                                         object.max_com_height() * alpha_xy_max -
                                         alpha_z_min * r_xy_max;
            zmp_constraints(i * 4 + 3) = beta_xy_max -
                                         object.min_com_height() * alpha_xy_max -
                                         alpha_z_min * r_xy_max;
        } else {
            zmp_constraints(i * 4) = -beta_xy_max -
                                     object.max_com_height() * alpha_xy_max -
                                     alpha_z_max * r_xy_max;
            zmp_constraints(i * 4 + 1) = -beta_xy_max -
                                         object.min_com_height() * alpha_xy_max -
                                         alpha_z_max * r_xy_max;
            zmp_constraints(i * 4 + 2) = -beta_xy_max -
                                         object.max_com_height() * alpha_xy_max -
                                         alpha_z_min * r_xy_max;
            zmp_constraints(i * 4 + 3) = -beta_xy_max -
                                         object.min_com_height() * alpha_xy_max -
                                         alpha_z_min * r_xy_max;

        }
    }

    return zmp_constraints;
}

template <typename Scalar>
Vector<Scalar> bounded_balancing_constraints_single(
    const Mat3<Scalar>& orientation, const Vec3<Scalar>& angular_vel,
    const Vec3<Scalar>& linear_acc, const Vec3<Scalar>& angular_acc,
    const BoundedBalancedObject<Scalar>& object) {
    Mat3<Scalar> C_we = orientation;
    Mat3<Scalar> C_ew = C_we.transpose();
    Mat3<Scalar> ddC_we =
        rotation_matrix_second_derivative(C_we, angular_vel, angular_acc);

    // gravity
    Vec3<Scalar> g = Scalar(-9.81) * Vec3<Scalar>::UnitZ();

    // NOTE: SLQ solver with soft constraints is sensitive to constraint
    // values, so having small values squared makes them too close to zero.
    Scalar eps(1e-6);

    // normal contact constraint
    Vector<Scalar> g_con =
        bounded_contact_constraint(ddC_we, C_ew, linear_acc, g, object, eps);

    // friction constraint
    Vector<Scalar> g_fric = bounded_friction_constraint(
        ddC_we, C_ew, angular_vel, linear_acc, angular_acc, g, object, eps);

    // tipping constraint
    Vector<Scalar> g_zmp = bounded_zmp_constraint(
        ddC_we, C_ew, angular_vel, linear_acc, angular_acc, g, object, eps);

    Vector<Scalar> g_bal(object.num_constraints());
    g_bal << g_con, g_fric, g_zmp;
    return g_bal;
}

template <typename Scalar>
struct BoundedTrayBalanceConfiguration {
    BoundedTrayBalanceConfiguration() {}

    BoundedTrayBalanceConfiguration(
        const std::vector<BoundedBalancedObject<Scalar>>& objects)
        : objects(objects) {}

    BoundedTrayBalanceConfiguration(
        const BoundedTrayBalanceConfiguration& other)
        : objects(other.objects) {}

    // Number of balancing constraints.
    size_t num_constraints() const {
        size_t n = 0;
        for (const auto& obj : objects) {
            n += obj.num_constraints();
        }
        return n;
    }

    // Size of parameter vector.
    size_t num_parameters() const {
        size_t n = 0;
        for (const auto& obj : objects) {
            n += obj.num_parameters();
        }
        return n;
    }

    // Get the parameter vector representing all objects in the configuration.
    Vector<Scalar> get_parameters() const {
        Vector<Scalar> parameters(num_parameters());
        size_t index = 0;
        for (const auto& obj : objects) {
            Vector<Scalar> p = obj.get_parameters();
            size_t n = p.size();
            parameters.segment(index, n) = p;
            index += n;
        }
        return parameters;
    }

    // Cast the configuration to a different underlying scalar type, creating
    // the objects from the supplied parameter vector.
    template <typename T>
    BoundedTrayBalanceConfiguration<T> cast_with_parameters(
        const Vector<T>& parameters) const {
        std::vector<BoundedBalancedObject<T>> objectsT;
        size_t index = 0;
        for (const auto& obj : objects) {
            size_t n = obj.num_parameters();
            auto objT = BoundedBalancedObject<T>::from_parameters(
                parameters.segment(index, n));
            objectsT.push_back(objT);
            index += n;
        }
        return BoundedTrayBalanceConfiguration<T>(objectsT);
    }

    template <typename T>
    BoundedTrayBalanceConfiguration<T> cast() const {
        std::vector<BoundedBalancedObject<T>> objectsT;
        for (const auto& obj : objects) {
            objectsT.push_back(obj.template cast<T>());
        }
        return BoundedTrayBalanceConfiguration<T>(objectsT);
    }

    // Compute the nominal balancing constraints for this configuration.
    Vector<Scalar> balancing_constraints(const Mat3<Scalar>& orientation,
                                         const Vec3<Scalar>& angular_vel,
                                         const Vec3<Scalar>& linear_acc,
                                         const Vec3<Scalar>& angular_acc) {
        Vector<Scalar> constraints(num_constraints());
        size_t index = 0;
        for (const auto& object : objects) {
            Vector<Scalar> v = bounded_balancing_constraints_single(
                orientation, angular_vel, linear_acc, angular_acc, object);
            constraints.segment(index, v.rows()) = v;
            index += v.rows();
        }
        return constraints;
    }

    std::vector<BoundedBalancedObject<Scalar>> objects;
};
