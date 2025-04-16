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

/******************************************************************************
 * PageRank Algorithm Implementation using ROCSPARSE
 * 
 * This implementation computes the PageRank vector using the power iteration
 * method and leverages ROCSPARSE for sparse matrix-vector multiplication.
 *
 * Key Steps:
 * 1. Represent the web graph as a sparse matrix in CSR format.
 * 2. Normalize the adjacency matrix to form the stochastic matrix.
 * 3. Use the power iteration method to compute the PageRank vector.
 * 4. Convergence is determined when the difference between successive iterations
 *    is below a certain threshold.
 *
 * Example Parameters:
 * - damping_factor: Controls teleportation probability (e.g., typical value is 0.85).
 * - tolerance: Convergence threshold for the PageRank vector.
 * - max_iterations: Maximum number of iterations to run.
 ******************************************************************************/

void calculatePageRank(const std::vector<float>& csr_values,
                       const std::vector<int>& csr_row_ptr,
                       const std::vector<int>& csr_col_ind,
                       std::vector<float>& pagerank,
                       float damping_factor,
                       int rows, int cols, int max_iterations, float tolerance) {
    rocsparse_handle handle;
    rocsparse_create_handle(&handle);

    // Allocate GPU memory for CSR matrix and PageRank vectors
    float *d_csr_values, *d_pagerank, *d_new_pagerank;
    int *d_csr_row_ptr, *d_csr_col_ind;
    hipMalloc(&d_csr_values, csr_values.size() * sizeof(float));
    hipMalloc(&d_csr_row_ptr, csr_row_ptr.size() * sizeof(int));
    hipMalloc(&d_csr_col_ind, csr_col_ind.size() * sizeof(int));
    hipMalloc(&d_pagerank, pagerank.size() * sizeof(float));
    hipMalloc(&d_new_pagerank, pagerank.size() * sizeof(float));

    // Copy data from host to device
    hipMemcpy(d_csr_values, csr_values.data(), csr_values.size() * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_csr_row_ptr, csr_row_ptr.data(), csr_row_ptr.size() * sizeof(int), hipMemcpyHostToDevice);
    hipMemcpy(d_csr_col_ind, csr_col_ind.data(), csr_col_ind.size() * sizeof(int), hipMemcpyHostToDevice);
    hipMemcpy(d_pagerank, pagerank.data(), pagerank.size() * sizeof(float), hipMemcpyHostToDevice);

    // Initialize ROCSPARSE matrix descriptor
    rocsparse_mat_descr descr;
    rocsparse_create_mat_descr(&descr);

    // Constants for the PageRank computation
    const float alpha = damping_factor;
    const float beta = (1.0f - damping_factor) / rows;

    // Iterative PageRank computation
    for (int iter = 0; iter < max_iterations; ++iter) {
        // Perform sparse matrix-vector multiplication: M * pagerank
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
                         d_pagerank,
                         &beta,
                         d_new_pagerank);

        // Copy the result back to host for convergence checking
        std::vector<float> new_pagerank(pagerank.size());
        hipMemcpy(new_pagerank.data(), d_new_pagerank, new_pagerank.size() * sizeof(float), hipMemcpyDeviceToHost);

        // Check for convergence
        float error = 0.0f;
        for (size_t i = 0; i < pagerank.size(); ++i) {
            error += fabs(new_pagerank[i] - pagerank[i]);
        }
        if (error < tolerance) break;

        // Update pagerank vector
        hipMemcpy(d_pagerank, d_new_pagerank, pagerank.size() * sizeof(float), hipMemcpyDeviceToDevice);
        pagerank = new_pagerank;
    }

    // Clean up GPU resources
    rocsparse_destroy_mat_descr(descr);
    rocsparse_destroy_handle(handle);
    hipFree(d_csr_values);
    hipFree(d_csr_row_ptr);
    hipFree(d_csr_col_ind);
    hipFree(d_pagerank);
    hipFree(d_new_pagerank);
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
