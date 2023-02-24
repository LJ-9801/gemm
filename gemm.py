import time
import os
os.environ['MKL_NUM_THREADS'] = '1'
import numpy as np

if __name__ == "__main__":
    N = 512
    #for i in range(100):
    flop = N*N*(2*N-1)
    A = np.random.rand(N,N)
    B = np.random.rand(N,N)
    #N = N*2

    start = time.monotonic()
    C = A @ B
    end = time .monotonic()

    a = open("A.txt", "w")
    with a as file:
        for line in A:
            np.savetxt(file, line, fmt='%.5f')
    a.close

    b = open("B.txt", "w")
    B = B.transpose()
    with b as file:
        for line in B:
            np.savetxt(file, line, fmt='%.5f')
    b.close

    c = open("C.txt", "w")
    with c as file:
        for line in C:
            np.savetxt(file, line, fmt='%.5f')
    c.close

    print(f"{N : >5}: {flop/(end-start)/10e9:.5f} GFLOPs")