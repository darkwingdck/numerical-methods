# File "System of linear equations"

import numpy as np
from math import sqrt
from colorama import Fore, Style, Back
from tabulate import tabulate



def upperTriangularAns(mx): # getting answer from upper triangular matrix
	res = 0
	s = 0
	ans = np.ones(len(mx[0]) - 1)
	curXid = len(ans) - 1
	for i in range(len(mx) - 1, -1, -1):
		for j in range(len(mx[i]) - 1):
			if curXid != j:
				s += ans[j] * mx[i][j]
		res =  (mx[i][-1] - s) / mx[i][curXid]
		ans[curXid] = res
		curXid -= 1
		s = 0
	return ans



def lowerTriangularAns(mx): # getting answer from lower triangular matrix
	res = 0
	s = 0
	ans = np.ones(len(mx[0]) - 1)
	curXid = 0
	for i in range(len(mx)):
		for j in range(len(mx[i]) - 1):
			if curXid != j:
				s += ans[j] * mx[i][j]
		res =  (mx[i][-1] - s) / mx[i][curXid]
		ans[curXid] = res
		curXid += 1
		s = 0
	return ans



def triangularMatrix(mx): # getting triangular matrix

	# putting row with not null first element on the first place
	for i in range(len(mx)): 
		if mx[i][0] != 0:
			mx[[0, i]] = mx[[i, 0]]
			break
	# I could make a case with null first element in each row, but I won't
	cur = 0
	for i in range(len(mx)):
		for j in range(i + 1, len(mx)):
			t = mx[j][cur] / mx[i][cur]
			for k in range(cur, len(mx[i])):
				mx[j][k] -= mx[i][k] * t
		cur += 1
	# getting answer
	ans = upperTriangularAns(mx)
	return ans
			


def squareRoot(a, b):
	# https://old.math.tsu.ru/EEResources/cm/text/4_9.htm
	# getting s
	N = len(a)
	s = np.zeros((N, N))
	s[0][0] = sqrt(a[0][0])
	for i in range(1, N):
		s[0][i] = a[0][i] / s[0][0]
	for i in range(1, N):
		tmp = 0
		for j in range(i):
			tmp += (s[j][i] * s[j][i])
		s[i][i] = sqrt(a[i][i] - tmp)
		
		tmp = 0
		for j in range(i + 1, N):
			for k in range(i):
				tmp += s[k][i] * s[k][j]
			s[i][j] = (a[i][j] - tmp) / s[i][i]
	# getting answer
	s = s.transpose()
	y = lowerTriangularAns(np.column_stack((s, b)))
	s = s.transpose()
	ans = upperTriangularAns(np.column_stack((s, y)))
	return ans



def iterations(a, b):
	m = 1/(np.linalg.norm(a) + 10)
	c = np.dot(m, b)
	B = np.eye(len(a)) - np.dot(m, a)
	x0 = c[:]
	x1 = x0[:]
	x2 = np.dot(B, x0) + c
	numberOfIters = 0 # number of iterations
	while np.linalg.norm(x2 - x1) > np.linalg.norm(x2 - x0) * np.linalg.norm(B) / (1 - np.linalg.norm(B)):
		x1 = np.dot(B, x0) + c
		x0 = x1
		x2 = np.dot(B, x0) + c
		numberOfIters += 1
	return x1.transpose(), numberOfIters



def matrices(): # ill-conditioned matrices
	eps = [1e-4, 1e-6]
	MATRICES = []
	for i in eps:
		a = np.array([[1, -1, -1], [0, 1, -1], [0, 0, 1]], dtype=float)
		a += np.array([[i, -i, -i], [i, i, -i], [i, i, i]], dtype=float)
		b = np.array([[-1], [-1], [1]], dtype=float)
		MATRICES.append(np.column_stack((a, b)))
	for i in eps:
		a = np.array([[1, -1, -1, -1], [0, 1, -1, -1], [0, 0, 1, -1], [0, 0, 0, 1]], dtype=float)
		a += np.array([[i, -i, -i, -i], [i, i, -i, -i], [i, i, i, -i], [i, i, i, i]], dtype=float)
		b = np.array([[-1], [-1], [-1], [1]], dtype=float)
		MATRICES.append(np.column_stack((a, b)))
	for i in eps:
		a = np.array([[1, -1, -1, -1, -1], [0, 1, -1, -1, -1], [0, 0, 1, -1, -1], [0, 0, 0, 1, -1], [0, 0, 0, 0, 1]], dtype=float)
		a += np.array([[i, -i, -i, -i, -i], [i, i, -i, -i, -i], [i, i, i, -i, -i], [i, i, i, i, -i], [i, i, i, i, i]], dtype=float)
		b = np.array([[-1], [-1], [-1], [-1], [1]], dtype=float)
		MATRICES.append(np.column_stack((a, b)))
	return MATRICES
	


def main():

	wellCondA = np.array([[7, 1, 1], [1, 9, 1], [1, 1, 11]], dtype=float)
	wellCondB = np.array([9, 11, 13], dtype=float)
	wellCondM = np.column_stack((wellCondA, wellCondB))

	column_list = ["id", "approximate value", "exact value", "error"] # for final table
	value_list = [] # for final table
	MATRICES = matrices()
	print(Fore.MAGENTA + "\n--------------------Gauss--------------------")
	print(Style.RESET_ALL)
	# well conditioned matrix
	print(Fore.CYAN + "---Well conditioned matrix---")
	print(Style.RESET_ALL)
	myGauss = triangularMatrix(wellCondM)
	pyGauss = np.linalg.solve(wellCondA, wellCondB).transpose()
	print("My solution: ", myGauss)
	print("Python solution: ", pyGauss)
	print("Error: ", abs(pyGauss - myGauss), "\n")
	# ill-conditioned matrices
	print(Fore.CYAN + "---Ill-conditioned matrices---")
	print(Style.RESET_ALL)
	for i in range(len(MATRICES)):
		mx = MATRICES[i]
		N = len(mx)
		a = mx[:, :N]
		b = mx[:, -1]
		myGauss = triangularMatrix(mx)
		pyGauss = np.linalg.solve(a, b).transpose()
		value_list.append([i + 1, [round(j, 4) for j in myGauss], [round(j, 4) for j in pyGauss], [j for j in abs(pyGauss - myGauss)]])
	print(tabulate(value_list, column_list, tablefmt="grid"), "\n")
	

	column_list = ["id", "approximate value", "exact value", "error"] # for final table
	value_list = [] # for final table
	print(Fore.MAGENTA + "--------------------Square--------------------")
	print(Style.RESET_ALL)
	# well conditioned matrix
	print(Fore.CYAN + "---Well conditioned matrix---")
	print(Style.RESET_ALL)
	mySquare = squareRoot(wellCondA, wellCondB)
	pySquare = np.linalg.solve(wellCondA, wellCondB).transpose()
	print("My solution: ", mySquare)
	print("Python solution: ", pySquare)
	print("Error: ", abs(mySquare - pySquare), "\n")

	# random positive definite matrces
	print(Fore.CYAN + "---Random positive definite matrices---")
	print(Style.RESET_ALL)
	for i in range(5):
		# creating random positive definite matrix	
		N = 3
		a = np.random.uniform(low=3, high=7, size=(N, N))
		a = np.dot(a, a.transpose())
		a = (a + a.T)/2
		b = np.random.uniform(low=3, high=7, size=(N, 1))
		mx = np.column_stack((a, b))
		
		mySquare = squareRoot(a, b)
		pySquare = np.linalg.solve(a, b).transpose()

		value_list.append([i + 1, [round(j, 4) for j in mySquare], [round(j, 4) for j in pySquare[0]], *[abs(pySquare[0] - mySquare)]])
	print(tabulate(value_list, column_list, tablefmt="grid"))
	

	column_list = ["id", "approximate value", "exact value", "error", "condition number", "number of iterations"] # for final table
	value_list = [] # for final table
	print(Fore.MAGENTA + "-------------Fixed-point iteration------------")
	print(Style.RESET_ALL)
	# well conditioned matrix
	print(Fore.CYAN + "---Well conditioned matrix---")
	print(Style.RESET_ALL)
	myIters = iterations(wellCondA, wellCondB)[0] # function returns answer and number of iterations, so [0]
	pyIters = np.linalg.solve(wellCondA, wellCondB).transpose()
	print("My solution: ", myIters)
	print("Python solution: ", pyIters)
	print("Error: ", abs(pyIters - myIters), "\n")

	# ill-conditioned matrices
	print(Fore.CYAN + "---Ill-conditioned matrces---")
	print(Style.RESET_ALL)
	for i in range(len(MATRICES)):
		mx = MATRICES[i]
		N = len(mx)
		a = mx[:, :N]
		b = mx[:, -1]

		myIters = iterations(a, b)[0]
		pyIters = np.linalg.solve(a, b).transpose()
		numberOfIters = iterations(a, b)[1] # number of iterations for Ill-conditioned matrices

		value_list.append([i + 1, [round(j, 4) for j in myIters], [round(j, 4) for j in pyIters], [round(j, 10) for j in abs(pyIters -  myIters)], np.linalg.cond(a), numberOfIters])
	print(tabulate(value_list, column_list, tablefmt="grid"))
	
	

if __name__ == "__main__":
	main()