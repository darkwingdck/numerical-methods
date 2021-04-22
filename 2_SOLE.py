# File "System of linear equations"

import numpy as np

def gauss(mx):
	# putting row with not null first element on the first place
	for i in range(len(mx)): 
		if mx[i][0] != 0:
			mx[[0, i]] = mx[[i, 0]]
			break
	# I could make a case with null first element in each row, but I won't
   
   	# getting triangular matrix
	cur = 0
	for i in range(len(mx)):
		for j in range(i + 1, len(mx)):
			t = mx[j][cur] / mx[i][cur]
			for k in range(cur, len(mx[i])):
				mx[j][k] -= mx[i][k] * t
		cur += 1

	# getting answer
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
			
			

	
			
			

def square(mx):
	pass

def iteration(mx):
	pass

def main():
	#mx = np.array([[7, 1, 1, 9], [1, 9, 1, 11], [1, 1, 11, 13]], dtype=float)
	#mx = np.array([[1, -1, 2, 0], [2, 1, -3, 0], [3, 0, 2, 0]], dtype=float)
	N = 4
	a = np.random.uniform(low=3, high=7, size=(N, N))
	b = np.random.uniform(low=3, high=7, size=(N, 1))
	mx = np.column_stack((a, b))
	print("--------------------Gauss--------------------")
	print("Matrix:")
	print(mx)
	print("\nMy solution:")
	myGauss = gauss(mx)
	print(myGauss)
	print("\nPython solution:")
	pyGauss = np.linalg.solve(a, b).transpose()
	print(pyGauss)
	print("\nError:")
	print(pyGauss -  myGauss)


if __name__ == "__main__":
	main()