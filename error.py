# File "Theory of errors", page 12, task 5

import math
from tabulate import tabulate

# exact values of functions


def exactFiFunc(x):
    return math.sinh(2 * x + 0.45)


def exactPsiFunc(x):
    return math.atan(6 * x + 1)


def exactFunc(x):
    return math.sqrt(exactFiFunc(x)) / exactPsiFunc(x)


def main():
    # errors of Fi, Psi and Func
    e1 = 1e-6 / 6.3
    e2 = 1e-6 / 5.4
    e3 = 1e-6 / 3

    errors = []  # for errors analysis

    x = 0.01
    step = 0.005
    num = 0  # amount of arguments

    fiFunc = 0  # approximate value
    fiFunc_lastEl = 1  # last element to check error with Taylor
    psiFunc = 0  # approximate value

    column_list = ["id", "x", 'approximate value',
                   'exact value', 'error']  # for final table
    value_list = []  # for final table

    while x <= 0.06:
        fiFunc = 0
        fiFunc_lastEl = 1
        psiFunc = 0
        num += 1
        # fiFunc with sinh Taylor series
        n = 0
        y = 2 * x + 0.45
        while fiFunc_lastEl > e1:
            fiFunc_lastEl = (y ** (2 * n + 1)) / math.factorial(2 * n + 1)
            fiFunc += fiFunc_lastEl
            n += 1

        # psiFunc with sinh Taylor series
        n = 0
        nTaylor = 0
        y = 6 * x + 1
        psiFunc = y ** (-1)
        # getting nTaylor (amount of Taylor elements) for atan
        while psiFunc > e2:
            nTaylor += 1
            psiFunc = (y ** (-2 * nTaylor - 1) / (2 * nTaylor + 1))
        psiFunc = y ** (-1)
        while n <= nTaylor:
            n += 1
            psiFunc += (-1) ** n * (y ** (-2 * n - 1) / (2 * n + 1))
        psiFunc = math.pi / 2 - psiFunc

        # Heron's formula
        w0 = 1
        w1 = 0.5 * (w0 + fiFunc / w0)
        nHeron = 0
        n = 0
        while abs(w1 - w0) / fiFunc > e3:
            nHeron += 1
            w0 = w1
            w1 = 0.5 * (w0 + fiFunc / w0)
        while n <= nHeron:
            n += 1
            w0 = w1
            w1 = 0.5 * (w0 + fiFunc / w0)

        errors.append(abs(exactFunc(x) - w1 / psiFunc))
        value_list.append([num, x, w1 / psiFunc, exactFunc(x),
                          abs(exactFunc(x) - w1 / psiFunc)])  # final table
        x += step

    # printing results
    print(tabulate(value_list, column_list, tablefmt="grid"))
    print("Max error: ", max(errors))
    print("Average absolute error: ", sum(errors) / len(errors))
    print("Standard deviation: ", sum([x ** 2 for x in errors]) / len(errors))


if __name__ == "__main__":
    main()
