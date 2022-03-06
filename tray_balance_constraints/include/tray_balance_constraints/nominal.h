#pragma once

#include <Eigen/Eigen>

#include "tray_balance_constraints/dynamics.h"
#include "tray_balance_constraints/support_area.h"
#include "tray_balance_constraints/types.h"


// TODO make pyramidal friction an option here
struct BalanceConstraintsEnabled {
    bool normal = true;
    bool friction = true;
    bool zmp = true;
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

    size_t num_constraints(const BalanceConstraintsEnabled& enabled) const;

    size_t num_parameters() const;

    Vector<Scalar> get_parameters() const;

    static BalancedObject<Scalar> from_parameters(const Vector<Scalar>& p);

    // Cast to another underlying scalar type
    template <typename T>
    BalancedObject<T> cast() const;

    // Compose multiple balanced objects. The first one is assumed to be the
    // bottom-most.
    static BalancedObject<Scalar> compose(
        const std::vector<BalancedObject<Scalar>>& objects);

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
                         const BalancedObject<Scalar>& object);

template <typename Scalar>
struct TrayBalanceConfiguration {
    TrayBalanceConfiguration() {}

    TrayBalanceConfiguration(const std::vector<BalancedObject<Scalar>>& objects,
                             const BalanceConstraintsEnabled& enabled)
        : objects(objects), enabled(enabled) {}

    // Number of balancing constraints.
    size_t num_constraints() const;

    // Size of parameter vector.
    size_t num_parameters() const;

    // Get the parameter vector representing all objects in the configuration.
    Vector<Scalar> get_parameters() const;

    // Cast the configuration to a different underlying scalar type, creating
    // the objects from the supplied parameter vector.
    template <typename T>
    TrayBalanceConfiguration<T> cast_with_parameters(
        const Vector<T>& parameters) const;

    // Compute the nominal balancing constraints for this configuration.
    Vector<Scalar> balancing_constraints(const Mat3<Scalar>& orientation,
                                         const Vec3<Scalar>& angular_vel,
                                         const Vec3<Scalar>& linear_acc,
                                         const Vec3<Scalar>& angular_acc);

    std::vector<BalancedObject<Scalar>> objects;
    BalanceConstraintsEnabled enabled;
};

#include "impl/nominal.h"
