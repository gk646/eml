#ifndef EML_TEST_TENSOR_OPS_H
#define EML_TEST_TENSOR_OPS_H

#include <eml/math/Tensor.h>
#include <eml/math/TensorOps.h>

using namespace eml;

// Using https://www.mathsisfun.com/algebra/matrix-multiplying.html

inline int test_matmul_square_a()
{
    float a_data[] = {1, 2, 3, 4};
    Tensor<float> A{2, 2};
    A.allocateCustom(a_data, 4);

    float b_data[] = {2, 0, 1, 2};
    Tensor<float> B{2, 2};
    B.allocateCustom(b_data, 4);

    float c_data[] = {0, 0, 0, 0};
    Tensor<float> C{2, 2};
    C.allocateCustom(c_data, 4);

    Matmul(A, B, C);

    float result[] = {4, 4, 10, 8};
    Tensor<float> R{2, 2};
    R.allocateCustom(result, 4);
    return TensorEquals(C, R);
}

inline int test_matmul_square_b()
{
    float a_data[] = {2, 0, 1, 2};
    Tensor<float> A{2, 2};
    A.allocateCustom(a_data, 4);

    float b_data[] = {1, 2, 3, 4};
    Tensor<float> B{2, 2};
    B.allocateCustom(b_data, 4);

    float c_data[] = {0, 0, 0, 0};
    Tensor<float> C{2, 2};
    C.allocateCustom(c_data, 4);

    Matmul(A, B, C);

    float result[] = {2, 4, 7, 10};
    Tensor<float> R{2, 2};
    R.allocateCustom(result, 4);
    return TensorEquals(C, R);
}

inline int test_matmul_shapes()
{
    float a_data[] = {1, 2, 3, 4, 5, 6};
    Tensor<float> A{2, 3};
    A.allocateCustom(a_data, 6);

    float b_data[] = {7, 8, 9, 10, 11, 12};
    Tensor<float> B{3, 2};
    B.allocateCustom(b_data, 6);

    float c_data[] = {0, 0, 0, 0};
    Tensor<float> C{2, 2};
    C.allocateCustom(c_data, 4);

    Matmul(A, B, C);

    float result[] = {58, 64, 139, 154};
    Tensor<float> R{2, 2};
    R.allocateCustom(result, 4);
    return TensorEquals(C, R);
}


inline void test_math_tensor_ops()
{
    EML_RUN_TEST(test_matmul_shapes);
}


#endif // EML_TEST_TENSOR_OPS_H