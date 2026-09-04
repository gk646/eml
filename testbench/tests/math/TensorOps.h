#ifndef EML_TEST_TENSOR_OPS_H
#define EML_TEST_TENSOR_OPS_H

#include <eml/math/Tensor.h>
#include <eml/math/TensorOps.h>

using namespace eml;

inline int testEquals()
{
    float aData[] = { 1, 2, 3, 4, 5, 6 };
    Tensor<float> A{ 2, 3 };
    A.allocateCustom( aData, 6 );

    float bData[] = { 7, 8, 9, 10, 11, 12 };
    Tensor<float> B{ 3, 2 };
    B.allocateCustom( bData, 6 );

    const bool first = Equals( A, B ) == false;

    float cData[] = { 7, 8, 9, 10, 11, 12 };
    Tensor<float> C{ 3, 2 };
    C.allocateCustom( cData, 6 );

    return first && Equals( B, C );
}

inline void TestTensorOps()
{
    EML_RUN_TEST( testEquals );
}

#endif // EML_TEST_TENSOR_OPS_H