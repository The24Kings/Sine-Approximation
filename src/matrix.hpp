#ifndef _MATRIX_HPP
#define _MATRIX_HPP

#include <iostream>
#include <vector>
#include <cstddef>

template <typename T>
struct Matrix {
    std::vector<std::vector<T>> data;
    size_t col;
    size_t row; 

    Matrix() {
        col = 0;
        row = 0;
        data = std::vector<std::vector<T>>();
    }

    Matrix(T n) {
        col = 1;
        row = 1;
        data = std::vector<std::vector<T>>(row, std::vector<T>(col, n));
    }

    Matrix(Matrix<T> &_m) {
        col = _m.col;
        row = _m.row;
        data = _m.data;
    }

    Matrix(size_t _col, size_t _row) {
        this->col = _col;
        this->row = _row;
        data = std::vector<std::vector<T>>(_row, std::vector<T>(_col));
    }
/*
    Matrix(int _col, int _row) {
        this->col = _col;
        this->row = _row;
        data = std::vector<std::vector<T>>(_row, std::vector<T>(_col));
    }

    Matrix(size_t _col, int _row) {
        this->col = _col;
        this->row = _row;
        data = std::vector<std::vector<T>>(_row, std::vector<T>(_col));
    }

    Matrix(int _col, size_t _row) {
        this->col = _col;
        this->row = _row;
        data = std::vector<std::vector<T>>(_row, std::vector<T>(_col));
    }
*/

    // Matrix multiplication
    Matrix<T> operator*(Matrix<T> _m) {
        Matrix<T> result;
        
        size_t col1 = col; // Left matrix
        size_t row1 = row; // Left matrix
        size_t col2 = _m.col; // Right matrix
        size_t row2 = _m.row; // Right matrix
        
        // Check if the matrix dimensions are valid x of the first matrix is not equal to row of the second matrix
        if (col1 != row2) {
            throw std::invalid_argument("Matrix dimensions do not function with multiplication");
        }

        result = Matrix<T>(col2, row1);

        for (size_t i = 0; i < row1; i++) {
            for (size_t j = 0; j < col2; j++) {
                for (size_t k = 0; k < col1; k++) {
                    result.data[i][j] += data[i][k] * _m.data[k][j]; // Black magic trickery
                }
            }
        }

        return result;
    }

    // Matrix multiplication with a scalar
    Matrix<T> operator*(T _col) {
        Matrix<T> result;

        for (size_t i = 0; i < row; i++) {
            for (size_t j = 0; j < col; j++) {
                result.data[i][j] = data[i][j] * _col;
            }
        }

        return result;
    }

    // Matrix addition
    Matrix<T> operator+(Matrix<T> _m) {
        if (col != _m.col || row != _m.row) {
            throw std::invalid_argument("Matrix dimensions do not match");
        }

        Matrix<T> result = Matrix<T>(col, row);

        for (size_t i = 0; i < row; i++) {
            for (size_t j = 0; j < col; j++) {
                result.data[i][j] = data[i][j] + _m.data[i][j];
            }
        }

        return result;
    }

    // Matrix subtraction
    Matrix<T> operator-(Matrix<T> _m) {
        if (col != _m.col || row != _m.row) {
            throw std::invalid_argument("Matrix dimensions do not match");
        }

        Matrix<T> result = Matrix<T>(col, row);

        for (size_t i = 0; i < row; i++) {
            for (size_t j = 0; j < col; j++) {
                result.data[i][j] = data[i][j] - _m.data[i][j];
            }
        }

        return result;
    }

    // Get the vector at index i
    std::vector<T>& operator[](size_t _i) {
        return data[_i];
    }

    // Set the Matrix to a vector of dimensions 1 x n
    Matrix<T> operator=(std::vector<T>& _v) {
        if (col != 1) {
            throw std::invalid_argument("Matrix dimensions must be 1 x n");
        }

        for (size_t i = 0; i < _v.size(); i++) {
            data[0][i] = _v[i];
        }

        return *this;
    }

    Matrix<T>& operator=(const Matrix<T>& other) {
        if (this != &other) {
            row = other.row;
            col = other.col;
			data = other.data;
        }

        return *this;
    }

    /**
     * @brief Prints a matrix
     * 
     */
    void print_matrix() {
            for (size_t i = 0; i < row; i++) {
                for (size_t j = 0; j < col; j++) {
                    std::cout << data[i][j] << "\t";
                }
                std::cout << std::endl;
            }
    }
};

#endif