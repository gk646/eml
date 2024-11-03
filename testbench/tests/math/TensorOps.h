#ifndef EML_TEST_TENSOR_OPS_H
#define EML_TEST_TENSOR_OPS_H

#include <eml/math/Tensor.h>
#include <eml/math/TensorOps.h>

using namespace eml;

// Using https://www.mathsisfun.com/algebra/matrix-multiplying.html

inline int testMatmulSquare()
{
    float a_data[] = { 1, 2, 3, 4 };
    Tensor<float> A{ 2, 2 };
    A.allocateCustom( a_data, 4 );

    float b_data[] = { 2, 0, 1, 2 };
    Tensor<float> B{ 2, 2 };
    B.allocateCustom( b_data, 4 );

    float c_data[] = { 0, 0, 0, 0 };
    Tensor<float> C{ 2, 2 };
    C.allocateCustom( c_data, 4 );

    ops::Matmul( A, B, C );

    float result[] = { 4, 4, 10, 8 };
    Tensor<float> R{ 2, 2 };
    R.allocateCustom( result, 4 );
    return ops::Equals( C, R );
}

inline int testMatmulSquare2()
{
    float a_data[] = { 2, 0, 1, 2 };
    Tensor<float> A{ 2, 2 };
    A.allocateCustom( a_data, 4 );

    float b_data[] = { 1, 2, 3, 4 };
    Tensor<float> B{ 2, 2 };
    B.allocateCustom( b_data, 4 );

    float c_data[] = { 0, 0, 0, 0 };
    Tensor<float> C{ 2, 2 };
    C.allocateCustom( c_data, 4 );

    ops::Matmul( A, B, C );

    float result[] = { 2, 4, 7, 10 };
    Tensor<float> R{ 2, 2 };
    R.allocateCustom( result, 4 );
    return ops::Equals( C, R );
}

inline int testMatmulShape()
{
    float a_data[] = { 1, 2, 3, 4, 5, 6 };
    Tensor<float> A{ 2, 3 };
    A.allocateCustom( a_data, 6 );

    float b_data[] = { 7, 8, 9, 10, 11, 12 };
    Tensor<float> B{ 3, 2 };
    B.allocateCustom( b_data, 6 );

    float c_data[] = { 0, 0, 0, 0 };
    Tensor<float> C{ 2, 2 };
    C.allocateCustom( c_data, 4 );

    ops::Matmul( A, B, C );

    float result[] = { 58, 64, 139, 154 };
    Tensor<float> R{ 2, 2 };
    R.allocateCustom( result, 4 );
    return ops::Equals( C, R );
}

inline int testEquals()
{
    float aData[] = { 1, 2, 3, 4, 5, 6 };
    Tensor<float> A{ 2, 3 };
    A.allocateCustom( aData, 6 );

    float bData[] = { 7, 8, 9, 10, 11, 12 };
    Tensor<float> B{ 3, 2 };
    B.allocateCustom( bData, 6 );

    const bool first = ops::Equals( A, B ) == false;

    float cData[] = { 7, 8, 9, 10, 11, 12 };
    Tensor<float> C{ 3, 2 };
    C.allocateCustom( cData, 6 );

    return first && ops::Equals( B, C );
}

inline void TestTensorOps()
{
    EML_RUN_TEST( testMatmulSquare );
    EML_RUN_TEST( testMatmulSquare2 );
    EML_RUN_TEST( testMatmulShape );
    EML_RUN_TEST( testEquals );
}

#endif // EML_TEST_TENSOR_OPS_H