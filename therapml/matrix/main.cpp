#include "matrix_ops.hpp"
#include <cstdlib>
#include <iostream>

using namespace std;

int main(){
    int exp_size = 10;
    long long int max_matrix_size = 1ll << exp_size;

    for(int exp = 1; exp <= exp_size; exp++) {
        int sub_exp_size = 10;
        for(int sub_exp = 1; sub_exp <= sub_exp_size; sub_exp++){
            int rowsA = rand() % (1 << exp) + 1;
            int colsA = rand() % (1 << exp) + 1;
            int rowsB = colsA;
            int colsB = rand() % (1 << exp) + 1;
            Matrix A(rowsA, colsA);
            Matrix B(rowsB, colsB);
            Matrix C1(rowsA, colsB);
            Matrix C2(rowsA, colsB);

            for(int i = 0; i < A.rows; i++) {
                for(int j = 0; j < A.cols; j++) {
                    A(i, j) = rand() / (double)1000;
                }
            }
            for(int i = 0; i < B.rows; i++) {
                for(int j = 0; j < B.cols; j++) {
                    B(i, j) = rand() / (double)1000;
                }
            }
            MatrixOps ops;
            ops.iterative_multiply(A, B, C1);
            ops.multiply(A, B, C2);

            for(int i = 0; i < C1.rows; i++) {
                for(int j = 0; j < C1.cols; j++) {
                    if(abs(C1(i, j) - C2(i, j)) > 1e-6) {
                        cout << "Mismatch at (" << i << ", " << j << "): " << C1(i, j) << " vs " << C2(i, j) << endl;
                        return 1;
                    }
                }
            }
            cout << "No mismatch" << "\n";
        }
    }
}