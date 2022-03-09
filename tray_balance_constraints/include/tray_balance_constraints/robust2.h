#pragma once

#include <Eigen/Eigen>

#include "tray_balance_constraints/dynamics.h"
#include "tray_balance_constraints/ellipsoid.h"
#include "tray_balance_constraints/support_area.h"
#include "tray_balance_constraints/types.h"
#include "tray_balance_constraints/util.h"

template <typename Scalar>
struct BoundedRigidBody {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    BoundedRigidBody(const Scalar& mass_min, const Scalar& mass_max,
                     const Scalar& r_gyr,
                     const Ellipsoid<Scalar>& com_ellipsoid)
        : mass_min(mass_min),
          mass_max(mass_max),
          r_gyr(r_gyr),
          com_ellipsoid(com_ellipsoid) {}

    // Compose multiple rigid bodies into one.
    // static BoundedRigidBody<Scalar> compose(
    //     const std::vector<BoundedRigidBody<Scalar>>& bodies);

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
    // TODO is this correct?
    static size_t combined_rank(
        const std::vector<BoundedRigidBody<Scalar>>& bodies) {
        std::vector<Ellipsoid<Scalar>> ellipsoids;
        for (const auto& body : bodies) {
            ellipsoids.push_back(body.com_ellipsoid.scaled(body.mass_min));
            ellipsoids.push_back(body.com_ellipsoid.scaled(body.mass_max));
        }
        return Ellipsoid<Scalar>::combined_rank(ellipsoids);
    }

    size_t num_parameters() const { return 2 + 1 + 15; }

    Vector<Scalar> get_parameters() const {
        Vector<Scalar> p;
        p << mass_min, mass_max, r_gyr, com_ellipsoid.get_parameters();
        return p;
    }

    // Create a RigidBody from a parameter vector
    static BoundedRigidBody<Scalar> from_parameters(
        const Vector<Scalar>& parameters, const size_t index = 0) {
        Scalar mass_min = parameters(index);
        Scalar mass_max = parameters(index + 1);
        Scalar r_gyr = parameters(index + 2);
        Ellipsoid<Scalar> com_ellipsoid = Ellipsoid<Scalar>::from_parameters(
            parameters.template segment<15>(index + 3));
        return BoundedRigidBody(mass_min, mass_max, r_gyr, com_ellipsoid);
    }

    Scalar mass_min;
    Scalar mass_max;
    Scalar r_gyr;
    Ellipsoid<Scalar> com_ellipsoid;
};

template <typename Scalar>
struct BoundedBalancedObject {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    BoundedBalancedObject(const BoundedRigidBody<Scalar>& body,
                          Scalar com_height_max,
                          const PolygonSupportArea<Scalar>& support_area_min,
                          Scalar r_tau_min, Scalar mu_min)
        : body(body),
          com_height_max(com_height_max),
          support_area_min(support_area_min),
          r_tau_min(r_tau_min),
          mu_min(mu_min) {}

    // Copy constructor
    BoundedBalancedObject(const BoundedBalancedObject& other)
        : body(other.body),
          com_height_max(other.com_height_max),
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
        const size_t num_fric = 1;
        const size_t num_zmp = support_area_min.num_constraints();
        return num_normal + num_fric + num_zmp;
    }

    size_t num_parameters() const {
        return 3 + body.num_parameters() + support_area_min.num_parameters();
    }

    Vector<Scalar> get_parameters() const {
        Vector<Scalar> p(num_parameters());
        p << com_height_max, r_tau_min, mu_min, body.get_parameters(),
            support_area_min.get_parameters();
        return p;
    }

    static BoundedBalancedObject<Scalar> from_parameters(
        const Vector<Scalar>& p) {
        Scalar com_height_max = p(0);
        Scalar r_tau_min = p(1);
        Scalar mu_min = p(2);

        size_t index = 3;
        auto body = BoundedRigidBody<Scalar>::from_parameters(p, index);

        index += body.num_parameters();
        auto support_area_min =
            PolygonSupportArea<Scalar>::from_parameters(p, index);
        return BoundedBalancedObject<Scalar>(
            body, com_height_max, support_area_min, r_tau_min, mu_min);
    }

    // Cast to another underlying scalar type
    template <typename T>
    BoundedBalancedObject<T> cast() const {
        Vector<Scalar> p = get_parameters();
        return BoundedBalancedObject<T>::from_parameters(p.template cast<T>());
    }

    // Compose multiple balanced objects. The first one is assumed to be the
    // bottom-most.
    // TODO
    // static BalancedObject<Scalar> compose(
    //     const std::vector<BalancedObject<Scalar>>& objects);

    // Dynamic parameters
    BoundedRigidBody<Scalar> body;

    // Geometry
    Scalar com_height_max;
    Scalar com_height_min;
    PolygonSupportArea<Scalar> support_area_min;

    // Friction
    Scalar r_tau_min;
    Scalar mu_min;
};

template <typename Scalar>
Scalar min_alpha_projection(const Vec3<Scalar> p, const Mat3<Scalar>& ddC_we,
                            const Mat3<Scalar>& C_ew,
                            const Vec3<Scalar>& linear_acc,
                            const Vec3<Scalar>& g,
                            const BoundedBalancedObject& object, Scalar eps) {
    Vec3<Scalar> a = ddC_we.transpose() * C_ew.transpose() * p;
    Scalar b = z.transpose() * C_ew * (linear_acc - g);
    Vec3<Scalar> e = object.com_ellipsoid.center();
    Mat3<Scalar> Einv = object.com_ellipsoid.Einv();
    return a.dot(e) - sqrt(a.transpose() * Einv() * a + eps) + b;
}

template <typename Scalar>
Scalar max_alpha_projection(const Vec3<Scalar> p, const Mat3<Scalar>& ddC_we,
                            const Mat3<Scalar>& C_ew,
                            const Vec3<Scalar>& linear_acc,
                            const Vec3<Scalar>& g,
                            const BoundedBalancedObject& object, Scalar eps) {
    Vec3<Scalar> a = ddC_we.transpose() * C_ew.transpose() * p;
    Scalar b = z.transpose() * C_ew * (linear_acc - g);
    Vec3<Scalar> e = object.com_ellipsoid.center();
    Mat3<Scalar> Einv = object.com_ellipsoid.Einv();
    // TODO only difference from min case is the sign here. refactor
    return a.dot(e) + sqrt(a.transpose() * Einv() * a + eps) + b;
}

template <typename Scalar>
Vector<Scalar> bounded_contact_constraint(const Mat3<Scalar>& ddC_we,
                                          const Mat3<Scalar>& C_ew,
                                          const Vec3<Scalar>& linear_acc,
                                          const Vec3<Scalar>& g,
                                          const BoundedBalancedObject& object,
                                          Scalar eps) {
    Vec3<Scalar> z;
    z << Scalar(0), Scalar(0), Scalar(1);

    Vector<Scalar> contact_constraint(1);
    contact_constraint << min_alpha_projection(z, ddC_we, C_ew, linear_acc, g,
                                               object, eps);
    return contact_constraint;
}

template <typename Scalar>
Vector<Scalar> bounded_friction_constraint(const Mat3<Scalar>& ddC_we,
                                           const Mat3<Scalar>& C_ew,
                                           const Vec3<Scalar>& linear_acc,
                                           const Vec3<Scalar>& g,
                                           const BoundedBalancedObject& object,
                                           Scalar beta_max, Scalar eps) {
    Mat3<Scalar> I = Mat3<Scalar>::Identity();
    Vec3<Scalar> x = I.col(0);
    Vec3<Scalar> y = I.col(1);
    Vec3<Scalar> z = I.col(2);

    Scalar alpha_z_min =
        min_alpha_projection(z, ddC_we, C_ew, linear_acc, g, object, eps);
    Scalar alpha_x_min =
        min_alpha_projection(x, ddC_we, C_ew, linear_acc, g, object, eps);
    Scalar alpha_x_max =
        min_alpha_projection(x, ddC_we, C_ew, linear_acc, g, object, eps);
    Scalar alpha_y_min =
        min_alpha_projection(y, ddC_we, C_ew, linear_acc, g, object, eps);
    Scalar alpha_y_max =
        min_alpha_projection(y, ddC_we, C_ew, linear_acc, g, object, eps);

    Vector<Scalar> friction_constraint(4);
    // clang-format off
    friction_constraint <<
        object.mu_min * alpha_z_min - alpha_x_max - alpha_y_max - beta_max / object.r_tau_min,
        object.mu_min * alpha_z_min - alpha_x_max + alpha_y_min - beta_max / object.r_tau_min,
        object.mu_min * alpha_z_min + alpha_x_min - alpha_y_max - beta_max / object.r_tau_min,
        object.mu_min * alpha_z_min + alpha_x_min + alpha_y_min - beta_max / object.r_tau_min;
    // clang-format on
    return friction_constraint;
}


template <typename Scalar>
Vec3<Scalar> compute_p(const Scalar h, const Vec2<Scalar>& normal, const Vec2<Scalar>& vertex) {
    Vec3<Scalar> p;
    p << -h * normal, normal.dot(vertex);
    return p;
}

template <typename Scalar>
Vector<Scalar> bounded_zmp_constraint(const Mat3<Scalar>& ddC_we,
                                      const Mat3<Scalar>& C_ew,
                                      const Vec3<Scalar>& linear_acc,
                                      const Vec3<Scalar>& g,
                                      const BoundedBalancedObject& object,
                                      Scalar beta_max, Scalar eps) {
    auto edges = object.support_area_min.edges();

    // Two constraints per edge
    Vector<Scalar> zmp_constraints(edges.size() * 2);

    for (int i = 0; i < edges.size(); ++i) {
        Vec3<Scalar> p1 = compute_p(object.com_height_max, edges[i].normal, edges[i].v1);
        Vec3<Scalar> p2 = compute_p(object.com_height_min, edges[i].normal, edges[i].v1);

        Scalar alpha_max1 = max_alpha_projection(p1, ddC_we, C_ew, linear_acc, g, object, eps);
        Scalar alpha_max2 = max_alpha_projection(p2, ddC_we, C_ew, linear_acc, g, object, eps);

        zmp_constraints(i*2) = alpha_max1 - beta_max;
        zmp_constraints(i*2 + 1) = alpha_max2 - beta_max;
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

    Mat3<Scalar> S_angular_vel = skew3<Scalar>(angular_vel);
    Mat3<Scalar> S_angular_acc = skew3<Scalar>(angular_acc);
    Mat3<Scalar> ddC_we =
        (S_angular_acc + S_angular_vel * S_angular_vel) * C_we;

    Vec3<Scalar> g;
    g << Scalar(0), Scalar(0), Scalar(-9.81);

    Scalar beta_max =
        squared(object.r_gyr) *
        (angular_vel.dot(angular_vel) + epsilon_norm<Scalar>(angular_acc, eps));

    // NOTE: SLQ solver with soft constraints is sensitive to constraint
    // values, so having small values squared makes them too close to zero.
    Scalar eps(0.01);

    // normal contact constraint
    Vector<Scalar> g_con =
        bounded_contact_constraint(ddC_we, C_ew, angular_acc, g, object, eps);

    // friction constraint
    Vector<Scalar> g_fric = bounded_friction_constraint(
        ddC_we, C_ew, angular_acc, g, object, beta_max, eps);

    // tipping constraint
    Vector<Scalar> g_zmp = bounded_zmp_constraint(object, alpha, beta, eps);

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
        return TrayBalanceConfiguration<T>(objectsT);
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
