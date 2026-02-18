#ifndef MATRIX_OPS_HPP
#define MATRIX_OPS_HPP

const double EPS = 1E-9;

class Matrix {
public:
    int rows, cols, stride;
    double* data;
    bool is_view;

    Matrix() : rows(0), cols(0), stride(0), data(nullptr), is_view(true) {}

    Matrix(int r, int c) : rows(r), cols(c), stride(c), is_view(false) {
        data = new double[r * c];
    }

    Matrix(double* d, int r, int c, int s) 
        : data(d), rows(r), cols(c), stride(s), is_view(true) {}

    ~Matrix() {
        if (!is_view && data) delete[] data;
    }

    inline double& operator()(int i, int j) {
        return data[i * stride + j];
    }

    bool isSquare() const {
        return rows == cols;
    }
};

class MatrixOps {
  public:
    int threshold;
    bool unrolled;
    int unroll_factor;
    bool gpu_enabled;

    MatrixOps(int threshold = 128, bool unrolled = true, int unroll_factor = 4, bool gpu_enabled = false) : threshold(threshold), unrolled(unrolled), unroll_factor(unroll_factor), gpu_enabled(gpu_enabled) {}

    void multipliability_check(Matrix &A, Matrix &B);
    void sumability_check(Matrix &A, Matrix &B);

    void multiply(Matrix &A, Matrix &B, Matrix &C);
    void add(Matrix &A, Matrix &B, Matrix &C);
    void transpose(Matrix &A, Matrix &B);
    void iterative_multiply(Matrix &A, Matrix &B, Matrix &C);
    void unrolled_iterative_multiply(Matrix &A, Matrix &B, Matrix &C);
    // void strassen_multiply(Matrix &A, Matrix &B, Matrix &C);
    void shared_memory_parallel_multiply(Matrix &A, Matrix &B, Matrix &C);
    void non_square_dc_multiply(Matrix &A, Matrix &B, Matrix &C);
    
    void split(Matrix &A, Matrix &A11, Matrix &A12, Matrix &A21, Matrix &A22);
    void split_vertical(Matrix &A, Matrix &A_left, Matrix &A_right, int cols_left);
    void split_horizontal(Matrix &A, Matrix &A_top, Matrix &A_bottom, int rows_top);

    double determinant(Matrix &A);
    void dot(Matrix &A, Matrix &B, Matrix &C);
};

#endif // MATRIX_OPS_HPP