#include <assert.h>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <vector>
#include <algorithm>
#include <immintrin.h>
#include <ctime>

//https://marek.ai/matrix-multiplication-on-cpu.html


// maybe keep track of the optimal ratio?
// between N and BLOCK
// 1024 --> 8
// 2048 --> 16
#define N 512
#define BLOCK 32 // 1 Gflops

// aligned 32byte needed for _mm256_load_pd
double __attribute__((aligned(32))) A[N*N];
double __attribute__((aligned(32))) B[N*N];
double __attribute__((aligned(32))) C[N*N];

//#define THREADING 1

//read data from python
void readMatrices(std::ifstream& getA, std::ifstream& getB, std::ifstream& getC,
                  std::vector<double>& trueC);

// check accuracy
bool check(const double c[], const std::vector<double>& trueC);

int main(int argc, char* argv[]){
    std::ifstream getA("A.txt");
    std::ifstream getB("B.txt");
    std::ifstream getC("C.txt");
    std::vector<double> trueC(N*N, 0);
    
    readMatrices(getA, getB, getC, trueC);

    #ifdef THREADING
    /*multithreading code here*/
    #else
    /*tiling matrrix multiplication*/
    
    // start timer
    std::clock_t start;
    double duration;

    start = std::clock();
    for(int i=0; i < N; i += BLOCK){
        for (int j =0; j < N; j += BLOCK){
            for(int k = 0; k < N; k += BLOCK){

                const int mini = std::min(i+BLOCK, N);
                const int minj = std::min(j+BLOCK, N);
                const int mink = std::min(k+BLOCK, N);

                for(int it = i; it < mini; it++){
                    for(int jt = j; jt < minj; jt++){
                        __m256d tmp = {0.0, 0.0, 0.0, 0.0}; 
                        for(int kt = k; kt < mink; kt+=4){
                            __m256d va = _mm256_load_pd(&A[it*N+kt]);
                            __m256d vb = _mm256_load_pd(&B[jt*N+kt]);
                            tmp = _mm256_fmadd_pd(va, vb, tmp);
                        }
                        C[it*N+jt] += tmp[0] + tmp[1] + tmp[2] + tmp[3];
                    }
                }
            }
        }
    }

    duration = (std::clock() - start)/(double) CLOCKS_PER_SEC;
    long double flop = (long double)(N*N)*(2*N-1);
    printf("%d: %.5Lf GFLOPS\n", N, flop/(duration)/10E9);
    //stop timer
    #endif

    check(C, trueC);

    return 0;
}


void readMatrices(std::ifstream& getA, std::ifstream& getB, std::ifstream& getC,
                  std::vector<double>& trueC){

    double num = 0.0;
    unsigned int i = 0;
    while(getA >> num){ A[i] = num; i++;}
    getA.close();

    i = 0;
    while(getB >> num){ B[i] = num; i++;}
    getB.close();

    i = 0;
    while(getC >> num){ trueC[i] = num; i++;}
    getC.close();
}

bool check(const double c[], const std::vector<double>& trueC){
    for(unsigned int i = 0; i<trueC.size(); i++){
        if(std::abs(c[i]-trueC[i]) > 1e-3){
            printf("mismatch at %d: %.5f != %.5f\n", i, c[i], trueC[i]);
            return 0;
        }
    }

    printf("match\n");
    return 1;
}