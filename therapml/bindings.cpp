#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "matrix/matrix_ops.hpp"

namespace py = pybind11;

// Zero-copy wrapper for Batch Matrix Multiplication
py::array_t<double> run_tensor_multiply(py::array_t<double> input1, py::array_t<double> input2) {
    // request() provides metadata about the NumPy arrays (shape, strides, pointer)
    auto buf1 = input1.request();
    auto buf2 = input2.request();

    if (buf1.ndim != 3 || buf2.ndim != 3)
        throw std::runtime_error("Inputs must be 3D tensors (Batch, Rows, Cols)");

    int batches = buf1.shape[0];
    int A_rows  = buf1.shape[1];
    int A_cols  = buf1.shape[2];
    int B_cols  = buf2.shape[2];

    if (A_cols != buf2.shape[1])
        throw std::runtime_error("Incompatible inner dimensions for multiplication");

    // Allocate the output array directly in NumPy memory
    auto result = py::array_t<double>({batches, A_rows, B_cols});
    auto buf3 = result.request();

    MatrixOps ops;

    for (int b = 0; b < batches; b++) {
        // Calculate pointer offsets for the current batch
        // We assume contiguous memory here (C-style)
        double* ptr1 = static_cast<double*>(buf1.ptr) + (b * A_rows * A_cols);
        double* ptr2 = static_cast<double*>(buf2.ptr) + (b * A_cols * B_cols);
        double* ptr3 = static_cast<double*>(buf3.ptr) + (b * A_rows * B_cols);

        // Create Matrix 'Views'. These do NOT allocate memory.
        // The 'stride' in your Matrix class matches the number of columns 
        // because these are contiguous sub-blocks.
        Matrix A(ptr1, A_rows, A_cols, A_cols);
        Matrix B(ptr2, A_cols, B_cols, B_cols);
        Matrix C(ptr3, A_rows, B_cols, B_cols);

        // Perform the multiplication directly into NumPy's memory
        ops.multiply(A, B, C);
    }

    return result;
}

py::object run_tensor_dot(
    py::array_t<double, py::array::c_style | py::array::forcecast> input1,
    py::array_t<double, py::array::c_style | py::array::forcecast> input2,
    int dim
) {
    auto buf1 = input1.request();
    auto buf2 = input2.request();

    if (buf1.ndim != buf2.ndim) {
        throw std::runtime_error("Input tensors must have the same rank");
    }
    if (dim < 0) {
        throw std::runtime_error("dim must be non-negative");
    }

    MatrixOps ops;

    if (buf1.ndim == 1) {
        if (dim != 0) {
            throw std::runtime_error("1D dot only supports dim=0");
        }
        if (buf1.shape[0] != buf2.shape[0]) {
            throw std::runtime_error("Vector sizes must match for dot");
        }
        int n = static_cast<int>(buf1.shape[0]);
        double* a = static_cast<double*>(buf1.ptr);
        double* b = static_cast<double*>(buf2.ptr);

        Matrix A(a, n, 1, 1);
        Matrix B(b, n, 1, 1);
        Matrix C(1, 1);
        ops.dot(A, B, C);
        return py::float_(C(0, 0));
    }

    if (buf1.ndim == 2) {
        int rows = static_cast<int>(buf1.shape[0]);
        int cols = static_cast<int>(buf1.shape[1]);
        if (rows != buf2.shape[0] || cols != buf2.shape[1]) {
            throw std::runtime_error("2D dot requires same shape for both matrices");
        }
        double* a = static_cast<double*>(buf1.ptr);
        double* b = static_cast<double*>(buf2.ptr);

        if (dim == 0) {
            auto result = py::array_t<double>({cols});
            auto out = result.request();
            double* c = static_cast<double*>(out.ptr);

            Matrix A(a, rows, cols, cols);
            Matrix B(b, rows, cols, cols);
            Matrix C(c, 1, cols, cols);
            ops.dot(A, B, C);
            return result;
        }

        if (dim == 1) {
            auto result = py::array_t<double>({rows});
            auto out = result.request();
            double* c = static_cast<double*>(out.ptr);

            Matrix A(a, rows, cols, cols);
            Matrix B(b, rows, cols, cols);
            Matrix AT(cols, rows);
            Matrix BT(cols, rows);
            ops.transpose(A, AT);
            ops.transpose(B, BT);

            Matrix C(c, 1, rows, rows);
            ops.dot(AT, BT, C);
            return result;
        }

        throw std::runtime_error("2D dot only supports dim=0 or dim=1");
    }

    if (buf1.ndim == 3) {
        int batches = static_cast<int>(buf1.shape[0]);
        int a_rows = static_cast<int>(buf1.shape[1]);
        int a_cols = static_cast<int>(buf1.shape[2]);
        int b_rows = static_cast<int>(buf2.shape[1]);
        int b_cols = static_cast<int>(buf2.shape[2]);
        if (batches != buf2.shape[0]) {
            throw std::runtime_error("3D dot requires matching batch sizes");
        }
        if (a_cols != b_cols) {
            throw std::runtime_error("3D dot requires matching last dimension");
        }
        double* a = static_cast<double*>(buf1.ptr);
        double* b = static_cast<double*>(buf2.ptr);

        if (dim == 0) {
            if (a_rows == b_rows) {
                int flat_cols = a_rows * a_cols;
                auto result = py::array_t<double>({a_rows, a_cols});
                auto out = result.request();
                double* c = static_cast<double*>(out.ptr);

                Matrix A(a, batches, flat_cols, flat_cols);
                Matrix B(b, batches, flat_cols, flat_cols);
                Matrix C(c, 1, flat_cols, flat_cols);
                ops.dot(A, B, C);
                return result;
            }

            auto result = py::array_t<double>({a_rows, b_rows, a_cols});
            auto out = result.request();
            double* c = static_cast<double*>(out.ptr);

            int flat_cols = a_rows * b_rows;
            Matrix Aexp(batches, flat_cols);
            Matrix Bexp(batches, flat_cols);
            Matrix Cexp(1, flat_cols);

            for (int k = 0; k < a_cols; k++) {
                for (int i = 0; i < batches; i++) {
                    for (int j = 0; j < a_rows; j++) {
                        double a_val = a[(static_cast<size_t>(i) * a_rows + j) * a_cols + k];
                        for (int l = 0; l < b_rows; l++) {
                            int col = j * b_rows + l;
                            Aexp(i, col) = a_val;
                            Bexp(i, col) = b[(static_cast<size_t>(i) * b_rows + l) * b_cols + k];
                        }
                    }
                }

                ops.dot(Aexp, Bexp, Cexp);

                for (int j = 0; j < a_rows; j++) {
                    for (int l = 0; l < b_rows; l++) {
                        int col = j * b_rows + l;
                        c[(static_cast<size_t>(j) * b_rows + l) * a_cols + k] = Cexp(0, col);
                    }
                }
            }

            return result;
        }

        if (dim == 1) {
            if (a_rows != b_rows || a_cols != b_cols) {
                throw std::runtime_error("3D dot along dim=1 requires matching shapes");
            }
            auto result = py::array_t<double>({batches, a_cols});
            auto out = result.request();
            double* c = static_cast<double*>(out.ptr);

            for (int i = 0; i < batches; i++) {
                double* a_ptr = a + static_cast<size_t>(i) * a_rows * a_cols;
                double* b_ptr = b + static_cast<size_t>(i) * b_rows * b_cols;
                double* c_ptr = c + static_cast<size_t>(i) * a_cols;

                Matrix A(a_ptr, a_rows, a_cols, a_cols);
                Matrix B(b_ptr, b_rows, b_cols, b_cols);
                Matrix C(c_ptr, 1, a_cols, a_cols);
                ops.dot(A, B, C);
            }
            return result;
        }

        throw std::runtime_error("3D dot only supports dim=0 or dim=1");
    }

    throw std::runtime_error("Unsupported tensor rank for dot");
}

PYBIND11_MODULE(therapml_cpp, m) {
    m.def("run_tensor_multiply", &run_tensor_multiply, "Zero-copy batch matrix multiply");
    m.def("run_tensor_dot", &run_tensor_dot, "Zero-copy tensor dot product");
}