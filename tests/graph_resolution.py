import math

def res( N: int ): return 360 / math.sqrt( math.pi * N )

for i, N in enumerate( [ 12, 42, 162, 642, 2562, 10242, 40962 ] ):
	print( f"{i} ({N}): {res(N):.2f}")

print("\n\n")

M = 365/5.0
for n in range(7):
	print( f"{n}: {M}")
	M = M /2.0