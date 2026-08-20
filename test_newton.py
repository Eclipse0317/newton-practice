import pytest
import math
import newton

def test_linear():
    """ test a simple linear function on the newton optimization function """
    assert newton.optimize(5, lambda x: x + 1) is None

def test_quadratic():
    """ test a simple quadratic function """
    assert abs(newton.optimize(0, lambda x: x**2 + 2 * x + 1) + 1) < 1e-4

def test_sine():
    """ test a trigonometric function, systematically vulnerable because second derivative would be zero"""
    assert abs(newton.optimize(0, math.sin) + math.pi/2) < 1e-4 

if __name__ == "__main__":
    test_linear()
    test_quadratic()
    test_sine()