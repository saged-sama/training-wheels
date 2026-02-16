#include "matrix_ops.hpp"
#include <stdexcept>
#include <algorithm>

using namespace std;

void MatrixOps::multipliability_check(Matrix &A, Matrix &B) {
    if(A.cols != B.rows) {
        throw invalid_argument("Incompatible matrix dimensions for multiplication");
    }
}

void MatrixOps::sumability_check(Matrix &A, Matrix &B) {
    if(A.rows != B.rows || A.cols != B.cols) {
        throw invalid_argument("Incompatible matrix dimensions for summation");
    }
}

void MatrixOps::unrolled_iterative_multiply(Matrix &A, Matrix &B, Matrix &C) {
    multipliability_check(A, B);
    for(int i = 0; i < A.rows; i++) {
        for(int j = 0; j < B.cols; j++) {
            double sum = 0.0;
            int k;
            for(k = 0; k <= A.cols - unroll_factor; k += unroll_factor) {
                sum += A(i, k) * B(k, j);
                sum += A(i, k + 1) * B(k + 1, j);
                sum += A(i, k + 2) * B(k + 2, j);
                sum += A(i, k + 3) * B(k + 3, j);
            }
            for(; k < A.cols; k++) {
                sum += A(i, k) * B(k, j);
            }
            C(i, j) = sum;
        }
    }
}

void MatrixOps::iterative_multiply(Matrix &A, Matrix &B, Matrix &C) {
    multipliability_check(A, B);
    for(int i = 0; i < C.rows; i++) {
        for(int j = 0; j < C.cols; j++) {
            C(i, j) = 0.0;
        }
    }

    for(int i = 0; i < A.rows; i++) {
        for(int k = 0; k < A.cols; k++) {
            double factor = A(i, k);
            for(int j = 0; j < B.cols; j++) {
                C(i, j) += factor * B(k, j);
            }
        }
    }
}

void MatrixOps::add(Matrix &A, Matrix &B, Matrix &C) {
    sumability_check(A, B);
    for(int i = 0; i < A.rows; i++) {
        for(int j = 0; j < A.cols; j++) {
            C(i, j) = A(i, j) + B(i, j);
        }
    }
}

void MatrixOps::split(Matrix &A, Matrix &A11, Matrix &A12, Matrix &A21, Matrix &A22) {
    int mid_row = A.rows / 2;
    int mid_col = A.cols / 2;

    A11 = Matrix(A.data, mid_row, mid_col, A.stride);
    A12 = Matrix(A.data + mid_col, mid_row, mid_col, A.stride);
    A21 = Matrix(A.data + (mid_row * A.stride), mid_row, mid_col, A.stride);
    A22 = Matrix(A.data + (mid_row * A.stride) + mid_col, mid_row, mid_col, A.stride);
}

void MatrixOps::split_vertical(Matrix &A, Matrix &A_left, Matrix &A_right, int cols_left) {
    A_left = Matrix(A.data, A.rows, cols_left, A.stride);
    A_right = Matrix(A.data + cols_left, A.rows, A.cols - cols_left, A.stride);
}

void MatrixOps::split_horizontal(Matrix &A, Matrix &A_top, Matrix &A_bottom, int rows_top) {
    A_top = Matrix(A.data, rows_top, A.cols, A.stride);
    A_bottom = Matrix(A.data + (rows_top * A.stride), A.rows - rows_top, A.cols, A.stride);
}

void MatrixOps::transpose(Matrix &A, Matrix &B) {
    for(int i = 0; i < A.rows; i++) {
        for(int j = 0; j < A.cols; j++) {
            B(j, i) = A(i, j);
        }
    }
}

void MatrixOps::shared_memory_parallel_multiply(Matrix &A, Matrix &B, Matrix &C) {
    multipliability_check(A, B);
    if(A.rows != A.cols || B.rows != B.cols || A.rows != B.rows) {
        throw invalid_argument("Shared memory parallel multiplication requires square matrices of the same size");
    }
    if(A.rows == 1){
        C(0, 0) = A(0, 0) * B(0, 0);
        return;
    }
    Matrix A11, A12, A21, A22;
    Matrix B11, B12, B21, B22;
    Matrix C11, C12, C21, C22;

    split(A, A11, A12, A21, A22);
    split(B, B11, B12, B21, B22);
    split(C, C11, C12, C21, C22);

    multiply(A11, B11, C11);
    {
        Matrix T11(C11.rows, C11.cols);
        multiply(A12, B21, T11);
        add(C11, T11, C11);
    }

    multiply(A11, B12, C12);
    {
        Matrix T12(C12.rows, C12.cols);
        multiply(A12, B22, T12);
        add(C12, T12, C12);
    }

    multiply(A21, B11, C21);
    {
        Matrix T21(C21.rows, C21.cols);
        multiply(A22, B21, T21);
        add(C21, T21, C21);
    }

    multiply(A21, B12, C22);
    {
        Matrix T22(C22.rows, C22.cols);
        multiply(A22, B22, T22);
        add(C22, T22, C22);
    }
}

void MatrixOps::non_square_dc_multiply(Matrix &A, Matrix &B, Matrix &C) {
    multipliability_check(A, B);
    if(max({A.cols, A.rows, B.cols, B.rows}) <= threshold) {
        iterative_multiply(A, B, C);
        return;
    }
    if(max({A.rows, A.cols, B.cols}) == A.rows) {
        int rows_top = A.rows / 2;
        Matrix A_top, A_bottom;
        split_horizontal(A, A_top, A_bottom, rows_top);
        Matrix C_top, C_bottom;
        split_horizontal(C, C_top, C_bottom, rows_top);
        multiply(A_top, B, C_top);
        multiply(A_bottom, B, C_bottom);
    }
    else if(max({A.rows, A.cols, B.cols}) == B.cols){
        int cols_left = B.cols / 2;
        Matrix B_left, B_right;
        split_vertical(B, B_left, B_right, cols_left);
        Matrix C_left, C_right;
        split_vertical(C, C_left, C_right, cols_left);
        multiply(A, B_left, C_left);
        multiply(A, B_right, C_right);
    }
    else {
        int cols_left = A.cols / 2;
        int rows_top = B.rows / 2;
        Matrix A_left, A_right;
        Matrix B_top, B_bottom;
        split_vertical(A, A_left, A_right, cols_left);
        split_horizontal(B, B_top, B_bottom, rows_top);

        Matrix p(A.rows, B.cols), q(A.rows, B.cols);
        multiply(A_left, B_top, p);
        multiply(A_right, B_bottom, q);
        add(p, q, C);
    }
}

void MatrixOps::multiply(Matrix &A, Matrix &B, Matrix &C) {
    multipliability_check(A, B);
    if(max({A.cols, A.rows, B.cols, B.rows}) <= threshold) {
        if(unrolled) {
            unrolled_iterative_multiply(A, B, C);
        } else {
            iterative_multiply(A, B, C);
        }
    } else if(A.rows == A.cols && B.rows == B.cols && A.rows == B.rows) {
        shared_memory_parallel_multiply(A, B, C);
    } else {
        non_square_dc_multiply(A, B, C);
    }
}

double MatrixOps::determinant(Matrix &A) {
    if(!A.isSquare()) {
        throw invalid_argument("Determinant is only defined for square matrices");
    }

    int dim = A.rows;

    double det = 1.0;
    for(int i = 0; i < dim; i++) {
        int pivot = i;
        for(int j = i + 1; j < dim; j++) {
            if(abs(A(j, i)) > abs(A(pivot, i))) {
                pivot = j;
            }
        }
        if(abs(A(pivot, i)) < EPS) {
            return 0.0;
        }
        if(pivot != i) {
            for(int k = 0; k < dim; k++) {
                swap(A(i, k), A(pivot, k));
            }
            det = -det;
        }
        det *= A(i, i);
        for(int j = i + 1; j < dim; j++) {
            double factor = A(j, i) / A(i, i);
            for(int k = i; k < dim; k++) {
                A(j, k) -= factor * A(i, k);
            }
        }
    }
    return det;
}

void MatrixOps::dot(Matrix &A, Matrix &B, Matrix &C) {
    if(A.rows != B.rows || A.cols != B.cols) {
        throw invalid_argument("Dot product requires matrices of compatible shapes");
    }

    Matrix temp(A.rows, A.cols);

    for(int i = 0; i < A.rows; i++) {
        for(int j = 0; j < A.cols; j++) {
            temp(i, j) = A(i, j) * B(i, j);
        }
    }

    for(int j = 0; j < temp.cols; j++){
        C(0, j) = 0.0;
        for(int i = 0; i < temp.rows; i++) {
            C(0, j) += temp(i, j);
        }
    }
}