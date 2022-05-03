import numpy as np

N = 15

def f(x):
	return 2 * x[0] ** 2 + (3 + 0.1 * N) * x[1] ** 2 + (4 + 0.1 * N) * x[2] ** 2 + x[0] * x[1] - x[1] * x[2] + x[0] * x[2] + x[0] - 2 * x[1] + 3 * x[2] + N



A = np.array([[4, 1, 1], [1, 6 + 0.2 * N, -1], [1, -1, 8 + 0.2 * N]], dtype=float)
B = np.array([[1], [-2], [3]], dtype=float)

EPS = 1e-6

def grad_descent(A, B):
	x0 = np.array([[1], [0], [0]], dtype=float)
	x1 = x0
	i = 0
	while 1:
		i += 1
		q = np.dot(A, x0) + B

		m = - (np.dot(q.T, q)) / np.dot(q.T, np.dot(A, q))
		x1 = x0 + m[0] * q
		# if abs(f(x1) - f(x0)) < EPS:
		# 	break 
		if np.linalg.norm(x0 - x1) - (1 / m[0]) * np.linalg.norm(np.dot(A, x0) + B) < EPS:
			break
		x0 = x1
	print("Iterations: ", i)
	print("Extremum point: ", *x1)
	print("Value at the extreme point: ", f(x1))

def coord_descent(A, B):
	x0 = np.array([[0], [0], [0]], dtype=float)
	e = np.array([[1], [0], [0]], dtype=float)
	x1 = x0
	i = 0
	while 1:
		m = - np.dot(e.T, np.dot(A, x0) + B) / np.dot(e.T, np.dot(A, e))
		x1 = x0 + m[0] * e
		if np.linalg.norm(x0 - x1) - (1 / m[0]) * np.linalg.norm(np.dot(A, x0) + B) < EPS and i > 2:
			break
		x0 = x1
		e[i % 3] = [0]
		i += 1
		e[i % 3] = [1]
	print("Iterations: ", i)
	print("Extremum point: ", *x1)
	print("Value at the extreme point: ", f(x1))


def main():
	print("****Gradiend descent****")
	grad_descent(A, B)
	print("****Coordinate descent****")
	coord_descent(A, B)


if __name__ == "__main__":
	main()