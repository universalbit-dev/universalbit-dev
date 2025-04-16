/******************************************************************************
 * File: hip_social_sparse_data.cpp
 * Description:
 * This file demonstrates handling sparse data in the context of web and 
 * social networks using ROCBLAS and ROCSPARSE libraries. Sparse data is 
 * commonly encountered in social networks and web data due to the following:
 *
 * - Large Data Space:
 *   Social networks and web data encompass a massive number of users,
 *   interactions, and potential connections. However, only a small fraction
 *   of these are actively utilized or populated with data (e.g., only a subset
 *   of friends interact frequently).
 *
 * - Sparse Features:
 *   User profiles or behaviors often have a high-dimensional feature space,
 *   but most of these features remain empty or contain zero values for an
 *   individual user.
 *
 * - Graph Sparsity:
 *   Social networks are typically represented as graphs, where nodes represent
 *   users and edges represent connections or interactions. Even in large
 *   networks, the number of edges is relatively small compared to the total
 *   possible edges, making the graph sparse.
 *
 * Implementation:
 * This file uses ROCBLAS for dense linear algebra operations and ROCSPARSE 
 * for sparse matrix computations. Together, these libraries optimize GPU 
 * performance for operations involving sparse data, making it suitable for 
 * applications in social network analysis and recommendation systems.
 *
 * Usage:
 * The example provided demonstrates sparse matrix-vector multiplication, 
 * where a sparse matrix in CSR (Compressed Sparse Row) format is multiplied 
 * with a dense vector. This operation can be extended to more complex 
 * computations involving sparse data.
 *
 * Author: universalbit-dev
 * Date: 2025-04-16
 * License: MIT
 ******************************************************************************/

#include <rocblas.h>
#include <rocsparse.h>
#include <iostream>
#include <vector>

// Example function to perform sparse matrix-vector multiplication
void sparseMatrixVectorMultiplication(const std::vector<float>& csr_values,
                                       const std::vector<int>& csr_row_ptr,
                                       const std::vector<int>& csr_col_ind,
                                       const std::vector<float>& dense_vector,
                                       std::vector<float>& result,
                                       int rows, int cols) {
    // Initialize ROCSPARSE handle
    rocsparse_handle handle;
    rocsparse_create_handle(&handle);

    // Allocate device memory
    float *d_csr_values, *d_dense_vector, *d_result;
    int *d_csr_row_ptr, *d_csr_col_ind;

    hipMalloc(&d_csr_values, csr_values.size() * sizeof(float));
    hipMalloc(&d_csr_row_ptr, csr_row_ptr.size() * sizeof(int));
    hipMalloc(&d_csr_col_ind, csr_col_ind.size() * sizeof(int));
    hipMalloc(&d_dense_vector, dense_vector.size() * sizeof(float));
    hipMalloc(&d_result, result.size() * sizeof(float));

    // Copy data to device
    hipMemcpy(d_csr_values, csr_values.data(), csr_values.size() * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_csr_row_ptr, csr_row_ptr.data(), csr_row_ptr.size() * sizeof(int), hipMemcpyHostToDevice);
    hipMemcpy(d_csr_col_ind, csr_col_ind.data(), csr_col_ind.size() * sizeof(int), hipMemcpyHostToDevice);
    hipMemcpy(d_dense_vector, dense_vector.data(), dense_vector.size() * sizeof(float), hipMemcpyHostToDevice);

    // Perform sparse matrix-vector multiplication
    const float alpha = 1.0f;
    const float beta = 0.0f;

    rocsparse_mat_descr descr;
    rocsparse_create_mat_descr(&descr);

    rocsparse_scsrmv(handle,
                     rocsparse_operation_none,
                     rows,
                     cols,
                     csr_values.size(),
                     &alpha,
                     descr,
                     d_csr_values,
                     d_csr_row_ptr,
                     d_csr_col_ind,
                     d_dense_vector,
                     &beta,
                     d_result);

    // Copy result back to host
    hipMemcpy(result.data(), d_result, result.size() * sizeof(float), hipMemcpyDeviceToHost);

    // Clean up
    rocsparse_destroy_mat_descr(descr);
    rocsparse_destroy_handle(handle);
    hipFree(d_csr_values);
    hipFree(d_csr_row_ptr);
    hipFree(d_csr_col_ind);
    hipFree(d_dense_vector);
    hipFree(d_result);
}

int main() {
    // Example sparse matrix in CSR format
    std::vector<float> csr_values = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<int> csr_row_ptr = {0, 2, 4};
    std::vector<int> csr_col_ind = {0, 1, 0, 1};

    // Example dense vector
    std::vector<float> dense_vector = {1.0f, 2.0f};

    // Result vector
    std::vector<float> result(2, 0.0f);

    // Perform sparse matrix-vector multiplication
    sparseMatrixVectorMultiplication(csr_values, csr_row_ptr, csr_col_ind, dense_vector, result, 2, 2);

    // Print the result
    std::cout << "Result:" << std::endl;
    for (const auto& val : result) {
        std::cout << val << " ";
    }
    std::cout << std::endl;

    return 0;
}
