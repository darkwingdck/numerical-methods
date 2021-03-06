# File "Newton's method, tasks 5"

import sole  # To solve a system of linear equations
import numpy as np
import math
import copy
from colorama import Fore, Style
import matplotlib.pyplot as plt


def f(x):  # Scalar function
    return (math.log10(x) - 7 / (2 * x + 6))


def df(x):  # Scalar function (derivative)
    return 1 / (x * math.log(10)) + 14 / ((2 * x + 6) * (2 * x + 6))


def newtonScalar():  # Newton's method for scalar equation
    # Localization of roots
    a = 1
    b = 20
    n = 40
    h = (b - a) / n
    xk = 0
    xk1 = 0
    for i in range(n):
        xk = a + i * h
        xk1 = a + h * (i + 1)
        if f(xk) * f(xk1) < 0:
            print("Root is located between", xk, "and", xk1)
            break
    a = xk
    b = xk1

    # Finding the actual root
    x0 = (a + b) / 2
    xn = f(x0)
    xn1 = xn - f(xn) / df(xn)
    while abs(xn1 - xn) > 1e-4:
        xn = xn1
        xn1 = xn - f(xn) / df(xn)
    return xn1


def nolinF(x):  # System of nolinear equations
    return np.array([math.sin(x[0]) + 2 * x[1] - 2, x[0] + math.cos(x[1] - 1) - 0.7], dtype=float)


def jacobian(x):  # Jacobian for system
    return np.array([[math.cos(x[0]), 2], [1, -math.sin(x[1] - 1)]],  dtype=float)


def newtonNolinearSys():  # Solving SONE with Newton's method

    # Graph
    x = np.linspace(-0.2, 1.5, 50)
    y1 = [(2 - math.sin(i)) / 2 for i in x]
    y2 = [math.acos(0.7 - i) - 1 for i in x]
    plt.grid()
    plt.plot(x, y1, x, y2)
    plt.show()

    x0 = np.array([0.5, 1], dtype=float)
    x1 = x0 + np.array([0.5, 1], dtype=float)
    while abs(x0[0] - x1[0]) > 1e-4:
        J = jacobian(x0)
        func = -nolinF(x0)
        mx = np.array([[J[0][0], J[0][1], func[0]], [
                      J[1][0], J[1][1], func[1]]], dtype=float)
        deltaX0 = sole.triangularMatrix(mx)
        x1 = copy.deepcopy(x0)
        x0 += deltaX0
    return x0


def main():
    print(Fore.BLUE + "--------------------Newton's method--------------------", end='')
    print(Style.RESET_ALL)
    print("Answer is", newtonScalar(), "\n")
    print(Fore.BLUE + "----------Newton's method (nolinear equations)---------", end='')
    print(Style.RESET_ALL)
    print(newtonNolinearSys())


if __name__ == "__main__":
    main()
