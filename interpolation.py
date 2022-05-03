import numpy as np
import matplotlib.pyplot as plt
import math


def func(x):
    return x ** 3 - math.exp(x) + 1


def func2(x):
    return abs(x) * func(x)


def lagrange_polynomial(points, values):
    assert len(points) == len(values)

    def l(x):
        s = 0
        for k in range(len(points)):
            p = 1
            for i in range(len(points)):
                if k != i:
                    p *= x - points[i]
                    p /= points[k] - points[i]
            s += p*values[k]
        return s
    return l


# takes interpolation boundaries and number of nodes
def random_nodes(a, b, n):
    x = list(np.linspace(a, b, n))
    y = [func2(i) for i in x]
    return (x, y)


def particular_nodes(a, b, n):
    x = np.array([0.5 * ((b - a) * math.cos((2 * i + 1)/(2 * (n + 1))
                 * math.pi) + b + a) for i in range(n)], dtype=float)
    y = np.array([func2(x[i]) for i in range(n)], dtype=float)
    return (x, y)


def r(x, p1, p2):
    delta = p1(x) - p2(x)
    return abs(delta)


def main():
    a, b, n = -1, 4, 8
    (x_random, y_random) = random_nodes(a, b, n)

    l = lagrange_polynomial(x_random, y_random)
    x = np.linspace(a, b, 100)
    y_origin = [func2(i) for i in x]
    y_lagrange = [l(i) for i in x]
    (x_particular, y_particular) = particular_nodes(a, b, n)
    x_particular_l = lagrange_polynomial(x_particular, y_particular)
    y_particular_y = [x_particular_l(i) for i in x]

    k = 100
    diff_origin_and_random = 0
    diff_origin_and_particular = 0
    diff_origin_and_random_values = []
    diff_origin_and_particular_values = []
    for i in np.linspace(a, b, k):
        diff_origin_and_random += r(i, func2, l)
        diff_origin_and_particular += r(i, func2, x_particular_l)

        diff_origin_and_random_values.append(r(i, func2, l))
        diff_origin_and_particular_values.append(r(i, func2, x_particular_l))

    plt.plot(x, diff_origin_and_random_values, label="f - random")
    plt.plot(x, diff_origin_and_particular_values, label="f - particular")
    plt.plot(x, y_origin, label="origin")
    plt.plot(x, y_lagrange, label="random")
    plt.plot(x, y_particular_y, label="particular")
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()
