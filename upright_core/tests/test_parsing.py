import numpy as np
import pytest
import yaml
from pathlib import Path
import upright_core as core
from upright_core.util import allclose_unordered


def load_config():
    path = Path(__file__).parent / "config.yaml"
    with open(path) as f:
        return yaml.safe_load(f)


def get_points_by_name(names, contacts):
    points = {name: [] for name in names}
    for c in contacts:
        points[c.object1_name].append(c.r_co_o1)
        points[c.object2_name].append(c.r_co_o2)
    return {name: np.array(points[name]) for name in names}


def test_parse_number():
    x = "1e-2"
    y = core.parsing.parse_number(x)
    assert y == 1e-2

    x = "2pi"
    y = core.parsing.parse_number(x)
    assert y == 2 * np.pi

    x = 1
    y = core.parsing.parse_number(x)
    assert np.isclose(x, y)


def test_parse_array():
    x = [1, 2, 3]
    y = core.parsing.parse_array(x)
    assert np.allclose(x, y)

    x = ["1rep3"]
    y = core.parsing.parse_array(x)
    assert np.allclose(y, [1, 1, 1])

    x = ["1rep3", "1pi"]
    y = core.parsing.parse_array(x)
    assert np.allclose(y, [1, 1, 1, np.pi])


def test_parse_diag_matrix_dict():
    d = {"scale": 2, "diag": ["1rep3"]}
    y = core.parsing.parse_diag_matrix_dict(d)
    assert np.allclose(y, 2 * np.eye(3))


def test_parse_support_offset():
    # Cartesian only
    d = {"x": 1, "y": 1}
    y = core.parsing.parse_support_offset(d)
    assert np.allclose(y, [1, 1])

    # polar only
    θ = 0.25 * np.pi
    d = {"r": 1, "θ": θ}
    y = core.parsing.parse_support_offset(d)
    assert np.allclose(y, [np.cos(θ), np.sin(θ)])

    # Cartesian and polar
    d = {"x": 1, "y": 1, "r": 1, "θ": θ}
    y = core.parsing.parse_support_offset(d)
    assert np.allclose(y, [np.cos(θ) + 1, np.sin(θ) + 1])

    # missing radius or angle
    with pytest.raises(ValueError):
        core.parsing.parse_support_offset({"r": 1})
    with pytest.raises(ValueError):
        core.parsing.parse_support_offset({"θ": 1})


def test_parse_box():
    ctrl_config = load_config()
    ctrl_config["balancing"]["arrangement"] = "box"
    ctrl_objects, contacts = core.parsing.parse_control_objects(ctrl_config)

    box = ctrl_objects["box"]
    assert np.isclose(box.mass, 1.0)
    assert np.allclose(box.com, [0, 0, 0.1])
    assert np.allclose(
        box.inertia, core.math.cuboid_inertia_matrix(1.0, [0.2, 0.2, 0.2])
    )

    assert len(contacts) == 4
    for c in contacts:
        assert np.allclose(c.normal, [0, 0, -1])
        assert np.allclose(c.span @ c.normal, [0, 0])
        assert np.isclose(c.mu, 0.45)  # includes the margin
        assert c.object1_name == "ee"
        assert c.object2_name == "box"

    points1 = np.array([c.r_co_o1 for c in contacts])
    points2 = np.array([c.r_co_o2 for c in contacts])

    points_expected = np.array(
        [[0.1, 0.1, 0], [0.1, -0.1, 0], [-0.1, -0.1, 0], [-0.1, 0.1, 0]]
    )

    # both sets of points should be the same since we use a common reference
    # point
    assert allclose_unordered(points1, points_expected)
    assert allclose_unordered(points2, points_expected)


def test_parse_cylinder_box():
    ctrl_config = load_config()
    ctrl_config["balancing"]["arrangement"] = "cylinder_box"
    _, contacts = core.parsing.parse_control_objects(ctrl_config)

    assert len(contacts) == 10
    for c in contacts:
        # all contacts with EE point upward
        # the others (between box and cylinder) are aligned with x-axis
        if c.object1_name == "ee":
            assert np.allclose(c.normal, [0, 0, -1])
        else:
            assert np.allclose(np.abs(c.normal), [1, 0, 0])

    # gather all contact points for each object
    points = get_points_by_name(["ee", "box", "cylinder"], contacts)

    # fmt: off
    ee_points_expected = np.array([
        [0, 0, 0], [-0.03, 0.03, 0], [-0.06, 0, 0], [-0.03, -0.03, 0],  # cylinder
        [0, -0.1, 0], [0.2, -0.1, 0], [0.2, 0.1, 0], [0, 0.1, 0]])  # box
    box_points_expected = np.array([
        [0, -0.1, 0], [0.2, -0.1, 0], [0.2, 0.1, 0], [0, 0.1, 0],
        [0, 0, 0], [0, 0, 0.2]])
    cylinder_points_expected = np.array([
        [0, 0, 0], [-0.03, 0.03, 0], [-0.06, 0, 0], [-0.03, -0.03, 0],
        [0, 0, 0], [0, 0, 0.2]])
    # fmt: on

    assert allclose_unordered(points["ee"], ee_points_expected)
    assert allclose_unordered(points["box"], box_points_expected)
    assert allclose_unordered(points["cylinder"], cylinder_points_expected)


def test_parse_wedge_box():
    ctrl_config = load_config()
    ctrl_config["balancing"]["arrangement"] = "wedge_box"
    ctrl_objects, contacts = core.parsing.parse_control_objects(ctrl_config)

    # check that wedge has CoM computed correctly
    assert np.allclose(ctrl_objects["wedge"].com, [-0.05, 0, 0.1])

    assert len(contacts) == 8
    z = np.array([0, 0, 1])
    C = core.math.roty(np.pi / 4)
    Cz = C @ z
    for c in contacts:
        # all contacts with EE point upward
        # the others (between box and cylinder) are are at 45-deg tilt
        if c.object1_name == "ee":
            assert np.allclose(c.normal, -z)
        else:
            assert np.allclose(c.normal, -Cz)

    points = get_points_by_name(["ee", "wedge", "box"], contacts)

    a = np.sqrt(0.02)

    # fmt: off
    wedge_points_expected = np.array([
        [0.15, 0.15, 0], [0.15, -0.15, 0], [-0.15, -0.15, 0], [-0.15, 0.15, 0],
        [0, 0.1, 0.15], [0, -0.1, 0.15], [-a, 0.1, 0.15 + a],
        [-a, -0.1, 0.15 + a]])
    box_points_expected = np.array([
        [0, 0.1, 0.15], [0, -0.1, 0.15], [-a, -0.1, 0.15 + a], [-a, 0.1, 0.15 + a]])
    # fmt: on

    assert allclose_unordered(points["wedge"], wedge_points_expected)
    assert allclose_unordered(points["box"], box_points_expected)
