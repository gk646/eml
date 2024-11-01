#ifndef EML_TEST_LAYERS_CONV2D_H
#define EML_TEST_LAYERS_CONV2D_H

#include <eml/nn/layers/Conv2D.h>

inline void TestNNLayersConv2D()
{
    float input[] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
                      // Channel 2
                      17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32,
                      // Channel 3
                      33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48 };

    Tensor<float> A {3,4,4};
    A.allocateCustom(input,48);
    A.print();
    Conv2D<float> layer{ 3, 3, { 2, 2 } };

    auto out = layer.forward(A);
    out.print();

}

#endif // EML_TEST_LAYERS_CONV2D_H