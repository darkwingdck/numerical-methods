# File "System of linear equations"

import numpy as np
from math import sqrt
from colorama import Fore, Style

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
			tmp += s[j][i] * s[j][i]
		s[i][i] = sqrt(a[i][i] - tmp)
		
		tmp = 0
		for j in range(i + 1, N):
			for k in range(i):
				tmp += s[k][i] * s[k][j]
			s[i][j] = (a[i][j] - tmp) / s[i][i]
	
	s = s.transpose()
	y = lowerTriangularAns(np.column_stack((s, b)))
	s = s.transpose()
	ans = upperTriangularAns(np.column_stack((s, y)))
	return ans

def iterations(a, b, eps):
	m = 2/(np.linalg.norm(a) + eps)
	c = np.dot(m, b)
	x0 = c
	return x0
	

def main():
	#mx = np.array([[7, 1, 1, 9], [1, 9, 1, 11], [1, 1, 11, 13]], dtype=float)
	#mx = np.array([[1, -1, 2, 0], [2, 1, -3, 0], [3, 0, 2, 0]], dtype=float)
	#mx = np.array([[2, 1, 1, 4], [1, 4, 1, 6], [1, 1, 6, 8]], dtype=float)
	#a = np.array([[2, 1, 1], [1, 4, 1], [1, 1, 6]], dtype=float)
	#b = np.array([[4], [6], [8]], dtype=float)
	#mx = np.column_stack((a, b))

	# creating random matrix
	N = 3
	a = np.random.uniform(low=3, high=7, size=(N, N))
	b = np.random.uniform(low=3, high=7, size=(N, 1))
	mx = np.column_stack((a, b))
	print(Fore.BLUE + "--------------------Gauss--------------------")
	print(Style.RESET_ALL)
	print("Matrix:")
	print(mx)
	print("\nMy solution:")
	myGauss = triangularMatrix(mx)
	print(myGauss)
	print("\nPython solution:")
	pyGauss = np.linalg.solve(a, b).transpose()
	print(pyGauss)
	print("\nError:")
	print(pyGauss -  myGauss)
	
	print(Fore.BLUE + "--------------------Square--------------------")
	print(Style.RESET_ALL)
	# creating random positive definite matrix
	a = np.random.uniform(low=3, high=7, size=(N, N))
	a = np.dot(a, a.transpose())
	a = (a + a.T)/2
	b = np.random.uniform(low=3, high=7, size=(N, 1))
	mx = np.column_stack((a, b))
	print("Matrix:")
	print(mx)
	print("\nMy solution:")
	mySquare = squareRoot(a, b)
	print(mySquare)
	print("\nPython solution:")
	pySquare = np.linalg.solve(a, b).transpose()
	print(pySquare)
	print("\nError:")
	print(pySquare -  mySquare)
	
	print(Fore.BLUE + "-------------Fixed-point iteration------------")
	print(Style.RESET_ALL)

	a = np.array([[1, -1, -1], [0, 1, -1], [0, 0, 1]], dtype=float)
	eps = 1e-3
	a += np.array([[eps, -eps, -eps], [eps, eps, -eps], [eps, eps, eps]], dtype=float)
	b = np.array([[-1], [-1], [1]], dtype=float)
	mx = np.column_stack((a, b))

	print("Matrix:")
	print(mx)
	print("\nMy solution:")
	myIters = iterations(a, b, eps)
	print(myIters)
	print("\nPython solution:")
	pyIters = np.linalg.solve(a, b).transpose()
	print(pyIters)
	print("\nError:")
	print(pyIters -  myIters)
	
	

if __name__ == "__main__":
	main()