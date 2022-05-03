from sympy.abc import x
import math
from typing import List
import numpy as np
import matplotlib.pyplot as plt
from sympy import integrate, exp


def f(x: float):
    return x ** 3 + math.exp(x)  # ТУТ МЕНЯТЬ


x = [-1, -0.5, 0, 0.5, 1]
y = [f(i) for i in x]

# функция которая вернет коэфициенты для наименьшеквадратичного полинома


def mnk(x: List[float], y: List[float], n: int):
    assert len(x) == len(y)
    b: List[float] = []  # это будет вектор б при решении СЛАУ
    for i in range(0, n + 1):
        b.append(sum([(x ** i) * y for x, y in zip(x, y)]))
    xPows = []  # коэфициенты для слау
    for i in range(0, 2 * n + 1):
        xPows.append(sum(xi ** i for xi in x))

    A = np.zeros((n + 1, n + 1))  # матрица для слау
    for i in range(n + 1):
        for j in range(n + 1):
            A[i][j] = xPows[i + j]

    coefs = np.linalg.solve(A, b)
    return coefs


polinoms = []
# эта функция делает функции полиномы


def poly(coefs: List[float]):
    def p(x: float) -> float:
        poly_str = ''
        r = 0  # степени помноженные на коэфициенты
        for i in range(len(coefs)):
            poly_str += 'x ^ ' + str(i) + ' * ' + str(coefs[i])
            r += (x ** i) * coefs[i]
        polinoms.append(poly_str)
        return r  # значение в точке
    return p


coefs = mnk(x, y, 3)  # по исходным данным ищем коэфициенты
P = poly(coefs)  # строим полином

X = np.linspace(-1, 1, 100)
Y_origin = [f(i) for i in X]
Y_apr = [P(i) for i in X]

plt.plot(X, Y_origin, 'r', label='origin')
plt.plot(X, Y_apr, 'b', label='aproximated')

##### Лежандр #####


# полиномы лежандра до 3 степени
legendre = [1, x, (3*x**2-1)/2, (5*x**3-3*x)/2]
f = x ** 3 + exp(x)


def prod(f1, f2):  # скалярное произведение в L2 пространстве
    return integrate(f1 * f2, (x, -1, 1))


ck = [float(prod(f, l) / prod(l, l)) for l in legendre]

# опять функция которая делает полиномы по коэфициентам


def poly(coefs):
    def p(t: float) -> float:
        r = coefs[0]
        poly_str = ''
        for i in range(1, len(coefs)):
            r += float(legendre[i].subs(x, t)) * coefs[i]
            poly_str = legendre[i]
        print(poly_str)
        return r
    return p


P = poly(ck)

Y_l = [P(i) for i in X]
print("Полином 1: x ^ 0 * 0.9944154101731908 + x ^ 1 * 0.9978537501020599 + x ^ 2 * 0.5477344596709182 + x ^ 3 * 1.1773474435417415")
print("Полином 2: x ^ 1 * 1.7036383235143269 + x ^ 2 * 0.35781435064737244 + x ^ 3 * 0.470455633668489")


plt.plot(X, Y_l, 'g', label='legendre')

plt.grid()
plt.legend()
plt.show()
