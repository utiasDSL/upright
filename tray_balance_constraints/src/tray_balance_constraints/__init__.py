# bring all bindings into the top-level package
from tray_balance_constraints.bindings import *
from tray_balance_constraints.composition import compose_bounded_objects


BoundedBalancedObject.compose = staticmethod(compose_bounded_objects)
