#pragma once

#include <Eigen/Eigen>

#include "upright_core/dynamics.h"
#include "upright_core/support_area.h"
#include "upright_core/types.h"

namespace upright {

template <typename Scalar>
struct BalancedObject {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    BalancedObject(const RigidBody<Scalar>& body, Scalar com_height,
                   const PolygonSupportArea<Scalar>& support_area, Scalar r_tau,
                   Scalar mu)
        : body(body),
          com_height(com_height),
          support_area(support_area),
          r_tau(r_tau),
          mu(mu) {}

    // Copy constructor
    BalancedObject(const BalancedObject& other)
        : body(other.body),
          com_height(other.com_height),
          support_area(other.support_area),
          r_tau(other.r_tau),
          mu(other.mu) {}

    // Copy assignment operator
    BalancedObject<Scalar>& operator=(const BalancedObject& other) {
        return *this;
    }

    ~BalancedObject() = default;

    size_t num_constraints(const BalanceConstraintsEnabled& enabled) const;

    size_t num_parameters() const;

    VecX<Scalar> get_parameters() const;

    static BalancedObject<Scalar> from_parameters(const VecX<Scalar>& p);

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
    PolygonSupportArea<Scalar> support_area;

    // Friction
    Scalar r_tau;
    Scalar mu;
};

template <typename Scalar>
struct BalancedObjectArrangement {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    BalancedObjectArrangement() {}

    BalancedObjectArrangement(
        const std::map<std::string, BalancedObject<Scalar>>& objects,
        const BalanceConstraintsEnabled& enabled, const Vec3<Scalar>& gravity)
        : objects(objects), enabled(enabled), gravity(gravity) {}

    // Constructor where all constraints are enabled by default (useful for
    // examining constraint values outside of the controller)
    BalancedObjectArrangement(
        const std::map<std::string, BalancedObject<Scalar>>& objects,
        const Vec3<Scalar>& gravity) : objects(objects), gravity(gravity) {}

    // Number of balancing constraints.
    size_t num_constraints() const;

    // Size of parameter vector.
    size_t num_parameters() const;

    // Get the parameter vector representing all objects in the configuration.
    VecX<Scalar> get_parameters() const;

    // Cast the configuration to a different underlying scalar type.
    template <typename T>
    BalancedObjectArrangement<T> cast() const;

    // Compute the nominal balancing constraints for this configuration.
    VecX<Scalar> balancing_constraints(const RigidBodyState<Scalar>& state);

    std::map<std::string, BalancedObject<Scalar>> objects;
    BalanceConstraintsEnabled enabled;
    Vec3<Scalar> gravity;
};

}  // namespace upright

#include "impl/nominal.tpp"
