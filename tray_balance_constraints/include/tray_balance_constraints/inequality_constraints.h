#pragma once

#include <Eigen/Eigen>

#include "tray_balance_constraints/support_area.h"
#include "tray_balance_constraints/types.h"
#include "tray_balance_constraints/util.h"

template <typename Scalar>
Scalar circle_r_tau(Scalar radius) {
    return Scalar(2.0) * radius / Scalar(3.0);
}

template <typename Scalar>
Scalar alpha_rect(Scalar w, Scalar h) {
    // alpha_rect for half of the rectangle
    Scalar d = sqrt(h * h + w * w);
    return (w * h * d + w * w * w * (log(h + d) - log(w))) / 12.0;
}

template <typename Scalar>
Scalar rectangle_r_tau(Scalar w, Scalar h) {
    // (see pushing notes)
    return (alpha_rect(w, h) + alpha_rect(h, w)) / (w * h);
}

template <typename Scalar>
Mat3<Scalar> cylinder_inertia_matrix(Scalar mass, Scalar radius,
                                     Scalar height) {
    // diagonal elements
    Scalar xx =
        mass * (Scalar(3.0) * radius * radius + height * height) / Scalar(12.0);
    Scalar yy = xx;
    Scalar zz = Scalar(0.5) * mass * radius * radius;

    // construct the inertia matrix
    Mat3<Scalar> I = Mat3<Scalar>::Zero();
    I.diagonal() << xx, yy, zz;
    return I;
}

template <typename Scalar>
Mat3<Scalar> cuboid_inertia_matrix(Scalar mass,
                                   const Vec3<Scalar>& side_lengths) {
    Scalar lx = side_lengths(0);
    Scalar ly = side_lengths(1);
    Scalar lz = side_lengths(2);

    Scalar xx = mass * (squared(ly) + squared(lz)) / Scalar(12.0);
    Scalar yy = mass * (squared(lx) + squared(lz)) / Scalar(12.0);
    Scalar zz = mass * (squared(lx) + squared(ly)) / Scalar(12.0);

    Mat3<Scalar> I = Mat3<Scalar>::Zero();
    I.diagonal() << xx, yy, zz;
    return I;
}

template <typename Scalar>
struct RigidBody {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    RigidBody(const Scalar mass, const Mat3<Scalar>& inertia,
              const Vec3<Scalar>& com)
        : mass(mass), inertia(inertia), com(com) {}

    // Compose multiple rigid bodies into one.
    static RigidBody<Scalar> compose(
        const std::vector<RigidBody<Scalar>>& bodies) {
        // Compute new mass and center of mass.
        Scalar mass = Scalar(0);
        Vec3<Scalar> com = Vec3<Scalar>::Zero();
        for (int i = 0; i < bodies.size(); ++i) {
            mass += bodies[i].mass;
            com += bodies[i].mass * bodies[i].com;
        }
        com = com / mass;

        // Parallel axis theorem to compute new moment of inertia.
        Mat3<Scalar> inertia = Mat3<Scalar>::Zero();
        for (int i = 0; i < bodies.size(); ++i) {
            Vec3<Scalar> r = bodies[i].com - com;
            Mat3<Scalar> R = skew3(r);
            inertia += bodies[i].inertia - bodies[i].mass * R * R;
        }

        return RigidBody<Scalar>(mass, inertia, com);
    }

    // Create a RigidBody from a parameter vector
    static RigidBody<Scalar> from_parameters(const Vector<Scalar>& parameters,
                                             const size_t index = 0) {
        Scalar mass(parameters(index));
        Vec3<Scalar> com(parameters.template segment<3>(index + 1));
        Vector<Scalar> I_vec(parameters.template segment<9>(index + 4));
        Mat3<Scalar> inertia(Eigen::Map<Mat3<Scalar>>(I_vec.data(), 3, 3));
        return RigidBody(mass, inertia, com);
    }

    size_t num_parameters() const { return 1 + 3 + 9; }

    Vector<Scalar> get_parameters() const {
        Vector<Scalar> p(num_parameters());
        Vector<Scalar> I_vec(
            Eigen::Map<const Vector<Scalar>>(inertia.data(), inertia.size()));
        p << mass, com, I_vec;
        return p;
    }

    Scalar mass;
    Mat3<Scalar> inertia;
    Vec3<Scalar> com;
};

template <typename Scalar>
struct BalancedObject {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    BalancedObject(const RigidBody<Scalar>& body, Scalar com_height,
                   const SupportAreaBase<Scalar>& support_area, Scalar r_tau,
                   Scalar mu)
        : body(body),
          com_height(com_height),
          support_area_ptr(support_area.clone()),
          r_tau(r_tau),
          mu(mu) {}

    // Copy constructor
    // NOTE: move constructor would instead use
    // std::move(other.support_area_ptr)
    BalancedObject(const BalancedObject& other)
        : body(other.body),
          com_height(other.com_height),
          support_area_ptr(other.support_area_ptr->clone()),
          r_tau(other.r_tau),
          mu(other.mu) {}

    // Copy assignment operator
    BalancedObject<Scalar>& operator=(const BalancedObject& other) {
        // TODO does this handle the unique_ptr properly?
        return *this;
    }

    ~BalancedObject() = default;

    size_t num_constraints() const {
        return 2 + support_area_ptr->num_constraints();
    }

    size_t num_parameters() const {
        return 3 + body.num_parameters() + support_area_ptr->num_parameters();
    }

    Vector<Scalar> get_parameters() const {
        Vector<Scalar> p(num_parameters());
        p << com_height, r_tau, mu, body.get_parameters(),
            support_area_ptr->get_parameters();
        return p;
    }

    static BalancedObject<Scalar> from_parameters(const Vector<Scalar>& p) {
        Scalar com_height = p(0);
        Scalar r_tau = p(1);
        Scalar mu = p(2);

        size_t start = 3;
        auto body = RigidBody<Scalar>::from_parameters(p, start);

        start += body.num_parameters();
        size_t num_params_remaining = p.size() - start - 1;

        if (num_params_remaining == 4) {
            auto support = CircleSupportArea<Scalar>::from_parameters(p, start);
            return BalancedObject<Scalar>(body, com_height, support, r_tau, mu);
        } else {
            auto support =
                PolygonSupportArea<Scalar>::from_parameters(p, start);
            return BalancedObject<Scalar>(body, com_height, support, r_tau, mu);
        }

        // return BalancedObject<Scalar>(body, com_height, *support_ptr,
        //                               r_tau, mu);
    }

    // Cast to another underlying scalar type
    template <typename T>
    BalancedObject<T> cast() const {
        Vector<Scalar> p = get_parameters();
        return BalancedObject<T>::from_parameters(p.template cast<T>());
    }

    // Compose multiple balanced objects. The first one is assumed to be the
    // bottom-most.
    static BalancedObject<Scalar> compose(
        const std::vector<BalancedObject<Scalar>>& objects) {
        std::vector<RigidBody<Scalar>> bodies;
        for (int i = 0; i < objects.size(); ++i) {
            bodies.push_back(objects[i].body);
        }

        // Relative to objects[0] since it is the one on the bottom (and so its
        // com_height is relative to the composite object's support plane)
        RigidBody<Scalar> body = RigidBody<Scalar>::compose(bodies);
        Vec3<Scalar> delta = objects[0].body.com - body.com;
        Scalar com_height = objects[0].com_height - delta(2);

        // Using a smart pointer here ensures that the cloned object is cleaned
        // up after we pass in the reference (which is itself cloned)
        std::unique_ptr<SupportAreaBase<Scalar>> support_area_ptr(
            objects[0].support_area_ptr->clone());
        support_area_ptr->offset = delta.head(2);

        BalancedObject<Scalar> composite(body, com_height, *support_area_ptr,
                                         objects[0].r_tau, objects[0].mu);
        return composite;
    }

    // Dynamic parameters
    RigidBody<Scalar> body;

    // Geometry
    Scalar com_height;
    std::unique_ptr<SupportAreaBase<Scalar>> support_area_ptr;

    // Friction
    Scalar r_tau;
    Scalar mu;
};

template <typename Scalar>
Vec2<Scalar> compute_zmp(const Mat3<Scalar>& orientation,
                         const Vec3<Scalar>& angular_vel,
                         const Vec3<Scalar>& linear_acc,
                         const Vec3<Scalar>& angular_acc,
                         const BalancedObject<Scalar>& object) {
    // Tray inertia (in the tray's own frame)
    Mat3<Scalar> It = object.body.inertia;

    Mat3<Scalar> C_we = orientation;
    Mat3<Scalar> C_ew = C_we.transpose();

    Mat3<Scalar> S_angular_vel = skew3<Scalar>(angular_vel);
    Mat3<Scalar> S_angular_acc = skew3<Scalar>(angular_acc);
    Mat3<Scalar> ddC_we =
        (S_angular_acc + S_angular_vel * S_angular_vel) * C_we;

    Vec3<Scalar> g;
    g << Scalar(0), Scalar(0), Scalar(-9.81);

    Vec3<Scalar> alpha =
        object.body.mass * C_ew * (linear_acc + ddC_we * object.body.com - g);

    Mat3<Scalar> Iw = C_we * It * C_ew;
    Vec3<Scalar> beta =
        C_ew * S_angular_vel * Iw * angular_vel + It * C_ew * angular_acc;

    // tipping constraint
    Eigen::Matrix<Scalar, 2, 2> S;
    S << Scalar(0), Scalar(1), Scalar(-1), Scalar(0);
    Vec2<Scalar> zmp =
        (-object.com_height * alpha.head(2) - S * beta.head(2)) / alpha(2);
    // Vector<Scalar> h3 = object.support_area_ptr->zmp_constraints(zmp);
    return zmp;
}

template <typename Scalar>
Vector<Scalar> inequality_constraints(const Mat3<Scalar>& orientation,
                                      const Vec3<Scalar>& angular_vel,
                                      const Vec3<Scalar>& linear_acc,
                                      const Vec3<Scalar>& angular_acc,
                                      const BalancedObject<Scalar>& object) {
    // Tray inertia (in the tray's own frame)
    Mat3<Scalar> It = object.body.inertia;

    Mat3<Scalar> C_we = orientation;
    Mat3<Scalar> C_ew = C_we.transpose();

    Mat3<Scalar> S_angular_vel = skew3<Scalar>(angular_vel);
    Mat3<Scalar> S_angular_acc = skew3<Scalar>(angular_acc);
    Mat3<Scalar> ddC_we =
        (S_angular_acc + S_angular_vel * S_angular_vel) * C_we;

    Vec3<Scalar> g;
    g << Scalar(0), Scalar(0), Scalar(-9.81);

    Vec3<Scalar> alpha =
        object.body.mass * C_ew * (linear_acc + ddC_we * object.body.com - g);

    Mat3<Scalar> Iw = C_we * It * C_ew;
    Vec3<Scalar> beta =
        C_ew * S_angular_vel * Iw * angular_vel + It * C_ew * angular_acc;

    // friction constraint
    // NOTE: SLQ solver with soft constraints is sensitive to constraint
    // values, so having small values squared makes them too close to zero.
    Scalar eps(0.01);
    // Scalar h1 = sqrt(squared(object.mu * alpha(2)) - squared(alpha(0)) -
    //             squared(alpha(1)) - squared(beta(2) / object.r_tau) + eps);
    Scalar h1 =
        object.mu * alpha(2) - sqrt(squared(alpha(0)) + squared(alpha(1)) +
                                    squared(beta(2) / object.r_tau) + eps);
    // Scalar h1 =
    //     squared(object.mu * alpha(2)) - squared(alpha(0)) + squared(alpha(1))
    //     -
    //                                 squared(beta(2) / object.r_tau) + eps;

    // normal constraint
    Scalar h2 = alpha(2);

    // tipping constraint
    Eigen::Matrix<Scalar, 2, 2> S;
    S << Scalar(0), Scalar(1), Scalar(-1), Scalar(0);
    Vec2<Scalar> zmp =
        (-object.com_height * alpha.head(2) - S * beta.head(2)) / alpha(2);
    Vector<Scalar> h3 = object.support_area_ptr->zmp_constraints(zmp);
    // Vec2<Scalar> az_zmp = -object.com_height * alpha.head(2) - S *
    // beta.head(2); Vector<Scalar> h3 =
    //     object.support_area_ptr->zmp_constraints_scaled(az_zmp, alpha(2));

    // Vector<Scalar> h3_ones = Vector<Scalar>::Ones(h3.rows());

    Vector<Scalar> constraints(object.num_constraints());
    constraints << h1, h2, h3;
    return constraints;
}

// This is the external API to be called for multiple objects
template <typename Scalar>
Vector<Scalar> balancing_constraints(
    const Mat3<Scalar>& orientation, const Vec3<Scalar>& angular_vel,
    const Vec3<Scalar>& linear_acc, const Vec3<Scalar>& angular_acc,
    const std::vector<BalancedObject<Scalar>>& objects) {
    size_t num_constraints = 0;
    for (const auto& object : objects) {
        num_constraints += object.num_constraints();
    }

    Vector<Scalar> constraints(num_constraints);
    size_t index = 0;
    for (const auto& object : objects) {
        Vector<Scalar> v = inequality_constraints(
            orientation, angular_vel, linear_acc, angular_acc, object);
        constraints.segment(index, v.rows()) = v;
        index += v.rows();
    }

    return constraints;
}
