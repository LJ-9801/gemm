python3 gemm.py
clang++ -O2 -ffast-math -mfma gemm.cpp -o gemm && ./gemm
rm gemm A.txt B.txt C.txt
