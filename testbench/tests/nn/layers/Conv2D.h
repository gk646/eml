#ifndef EML_TEST_LAYERS_CONV2D_H
#define EML_TEST_LAYERS_CONV2D_H

#include <eml/nn/layers/Conv2D.h>

inline int testConv2DCase1()
{
    float input[] = { 1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  8.0,  9.0,
                      10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0 };
    Tensor<float> A{ 3, 2, 3 };
    A.allocateCustom( input, 18 );
    Conv2D<float> layer{ 3, 1, { 2, 1 }, { 2, 1 }, { 1, 0 }, false, PaddingMode::REFLECT };
    ops::Fill( layer.weights, 1.0F );
    const auto out = layer.forward( A );
    float expected[] = { 51.0, 57.0, 63.0, 51.0, 57.0, 63.0 };
    Tensor R = out.copyDims();
    R.allocateCustom( expected, 6 );
    return ops::Equals( out, R );
}

inline int testConv2DCase2()
{
    float input[] = { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0 };
    Tensor<float> A{ 1, 3, 3 };
    A.allocateCustom( input, 9 );
    Conv2D<float> layer{ 1, 3, { 1, 1 }, { 1, 1 }, { 0, 1 }, true, PaddingMode::REPLICATE };
    ops::Fill( layer.weights, 1.0F );
    ops::Fill( layer.biases, 1.0F );
    const auto out = layer.forward( A );
    float expected[] = { 2.0, 2.0, 3.0, 4.0, 4.0, 5.0, 5.0, 6.0, 7.0, 7.0, 8.0, 8.0, 9.0, 10.0, 10.0,
                         2.0, 2.0, 3.0, 4.0, 4.0, 5.0, 5.0, 6.0, 7.0, 7.0, 8.0, 8.0, 9.0, 10.0, 10.0,
                         2.0, 2.0, 3.0, 4.0, 4.0, 5.0, 5.0, 6.0, 7.0, 7.0, 8.0, 8.0, 9.0, 10.0, 10.0 };
    Tensor R = out.copyDims();
    R.allocateCustom( expected, 45 );
    return ops::Equals( out, R );
}

inline int testConv2DCase3()
{
    float input[] = { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 };
    Tensor<float> A{ 1, 3, 2 };
    A.allocateCustom( input, 6 );
    Conv2D<float> layer{ 1, 1, { 1, 2 }, { 1, 2 }, { 0, 0 }, false, PaddingMode::CIRCULAR };
    ops::Fill( layer.weights, 1.0F );

    const auto out = layer.forward( A );
    float expected[] = { 3.0, 7.0, 11.0 };
    Tensor R = out.copyDims();
    R.allocateCustom( expected, 3 );
    return ops::Equals( out, R );
}

inline int testConv2DCase4()
{
    float input[] = { 1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  8.0,  9.0,
                      10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0 };
    Tensor<float> A{ 3, 2, 3 };
    A.allocateCustom( input, 18 );
    Conv2D<float> layer{ 3, 1, { 2, 1 }, { 2, 1 }, { 1, 0 }, true, PaddingMode::ZEROS };
    ops::Fill( layer.weights, 1.0F );
    ops::Fill( layer.biases, 1.0F );
    const auto out = layer.forward( A );
    float expected[] = { 22.0, 25.0, 28.0, 31.0, 34.0, 37.0 };
    Tensor R = out.copyDims();
    R.allocateCustom( expected, 6 );
    return ops::Equals( out, R );
}

inline int testConv2DCase5()
{
    float input[] = { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0 };
    Tensor<float> A{ 1, 3, 3 };
    A.allocateCustom( input, 9 );
    Conv2D<float> layer{ 1, 3, { 1, 1 }, { 1, 1 }, { 0, 1 }, false, PaddingMode::REFLECT };
    ops::Fill( layer.weights, 1.0F );

    const auto out = layer.forward( A );
    float expected[] = { 2.0, 1.0, 2.0, 3.0, 2.0, 5.0, 4.0, 5.0, 6.0, 5.0, 8.0, 7.0, 8.0, 9.0, 8.0,
                         2.0, 1.0, 2.0, 3.0, 2.0, 5.0, 4.0, 5.0, 6.0, 5.0, 8.0, 7.0, 8.0, 9.0, 8.0,
                         2.0, 1.0, 2.0, 3.0, 2.0, 5.0, 4.0, 5.0, 6.0, 5.0, 8.0, 7.0, 8.0, 9.0, 8.0 };
    Tensor R = out.copyDims();
    R.allocateCustom( expected, 45 );
    return ops::Equals( out, R );
}

inline int testConv2DCase6()
{
    float input[] = { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 };
    Tensor<float> A{ 1, 3, 2 };
    A.allocateCustom( input, 6 );
    Conv2D<float> layer{ 1, 1, { 1, 2 }, { 1, 2 }, { 0, 0 }, true, PaddingMode::REPLICATE };
    ops::Fill( layer.weights, 1.0F );
    ops::Fill( layer.biases, 1.0F );
    const auto out = layer.forward( A );
    float expected[] = { 4.0, 8.0, 12.0 };
    Tensor R = out.copyDims();
    R.allocateCustom( expected, 3 );
    return ops::Equals( out, R );
}

inline int testConv2DCase7()
{
    float input[] = { 1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  8.0,  9.0,
                      10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0 };
    Tensor<float> A{ 3, 2, 3 };
    A.allocateCustom( input, 18 );
    Conv2D<float> layer{ 3, 1, { 2, 1 }, { 2, 1 }, { 1, 0 }, false, PaddingMode::CIRCULAR };
    ops::Fill( layer.weights, 1.0F );

    const auto out = layer.forward( A );
    float expected[] = { 51.0, 57.0, 63.0, 51.0, 57.0, 63.0 };
    Tensor R = out.copyDims();
    R.allocateCustom( expected, 6 );
    return ops::Equals( out, R );
}

inline int testConv2DCase8()
{
    float input[] = { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0 };
    Tensor<float> A{ 1, 3, 3 };
    A.allocateCustom( input, 9 );
    Conv2D<float> layer{ 1, 3, { 1, 1 }, { 1, 1 }, { 0, 1 }, true, PaddingMode::ZEROS };
    ops::Fill( layer.weights, 1.0F );
    ops::Fill( layer.biases, 1.0F );
    const auto out = layer.forward( A );
    out.print();
    float expected[] = { 1.0, 2.0, 3.0, 4.0, 1.0, 1.0, 5.0, 6.0, 7.0, 1.0, 1.0, 8.0, 9.0, 10.0, 1.0,
                         1.0, 2.0, 3.0, 4.0, 1.0, 1.0, 5.0, 6.0, 7.0, 1.0, 1.0, 8.0, 9.0, 10.0, 1.0,
                         1.0, 2.0, 3.0, 4.0, 1.0, 1.0, 5.0, 6.0, 7.0, 1.0, 1.0, 8.0, 9.0, 10.0, 1.0 };
    Tensor R = out.copyDims();
    R.allocateCustom( expected, 45 );
    return ops::Equals( out, R );
}

inline int testConv2DCase9()
{
    float input[] = { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 };
    Tensor<float> A{ 1, 3, 2 };
    A.allocateCustom( input, 6 );
    Conv2D<float> layer{ 1, 1, { 1, 2 }, { 1, 2 }, { 0, 0 }, false, PaddingMode::REFLECT };
    ops::Fill( layer.weights, 1.0F );

    const auto out = layer.forward( A );
    float expected[] = { 3.0, 7.0, 11.0 };
    Tensor R = out.copyDims();
    R.allocateCustom( expected, 3 );
    return ops::Equals( out, R );
}

inline int testConv2DCase10()
{
    float input[] = { 1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  8.0,  9.0,
                      10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0 };
    Tensor<float> A{ 3, 2, 3 };
    A.allocateCustom( input, 18 );
    Conv2D<float> layer{ 3, 1, { 2, 1 }, { 2, 1 }, { 1, 0 }, true, PaddingMode::REPLICATE };
    ops::Fill( layer.weights, 1.0F );
    ops::Fill( layer.biases, 1.0F );
    const auto out = layer.forward( A );
    float expected[] = { 43.0, 49.0, 55.0, 61.0, 67.0, 73.0 };
    Tensor R = out.copyDims();
    R.allocateCustom( expected, 6 );
    return ops::Equals( out, R );
}

inline int testConv2DCase11()
{
    float input[] = { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0 };
    Tensor<float> A{ 1, 3, 3 };
    A.allocateCustom( input, 9 );
    Conv2D<float> layer{ 1, 3, { 1, 1 }, { 1, 1 }, { 0, 1 }, false, PaddingMode::CIRCULAR };
    ops::Fill( layer.weights, 1.0F );

    const auto out = layer.forward( A );
    float expected[] = { 3.0, 1.0, 2.0, 3.0, 1.0, 6.0, 4.0, 5.0, 6.0, 4.0, 9.0, 7.0, 8.0, 9.0, 7.0,
                         3.0, 1.0, 2.0, 3.0, 1.0, 6.0, 4.0, 5.0, 6.0, 4.0, 9.0, 7.0, 8.0, 9.0, 7.0,
                         3.0, 1.0, 2.0, 3.0, 1.0, 6.0, 4.0, 5.0, 6.0, 4.0, 9.0, 7.0, 8.0, 9.0, 7.0 };
    Tensor R = out.copyDims();
    R.allocateCustom( expected, 45 );
    return ops::Equals( out, R );
}

inline int testConv2DCase12()
{
    float input[] = { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 };
    Tensor<float> A{ 1, 3, 2 };
    A.allocateCustom( input, 6 );
    Conv2D<float> layer{ 1, 1, { 1, 2 }, { 1, 2 }, { 0, 0 }, true, PaddingMode::ZEROS };
    ops::Fill( layer.weights, 1.0F );
    ops::Fill( layer.biases, 1.0F );
    const auto out = layer.forward( A );
    float expected[] = { 4.0, 8.0, 12.0 };
    Tensor R = out.copyDims();
    R.allocateCustom( expected, 3 );
    return ops::Equals( out, R );
}

inline int testConv2DCase13()
{
    float input[] = { 1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  8.0,  9.0,
                      10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0 };
    Tensor<float> A{ 3, 2, 3 };
    A.allocateCustom( input, 18 );
    Conv2D<float> layer{ 3, 1, { 2, 1 }, { 2, 1 }, { 1, 0 }, false, PaddingMode::REFLECT };
    ops::Fill( layer.weights, 1.0F );

    const auto out = layer.forward( A );
    float expected[] = { 51.0, 57.0, 63.0, 51.0, 57.0, 63.0 };
    Tensor R = out.copyDims();
    R.allocateCustom( expected, 6 );
    return ops::Equals( out, R );
}

inline int testConv2DCase14()
{
    float input[] = { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0 };
    Tensor<float> A{ 1, 3, 3 };
    A.allocateCustom( input, 9 );
    Conv2D<float> layer{ 1, 3, { 1, 1 }, { 1, 1 }, { 0, 1 }, true, PaddingMode::REPLICATE };
    ops::Fill( layer.weights, 1.0F );
    ops::Fill( layer.biases, 1.0F );
    const auto out = layer.forward( A );
    float expected[] = { 2.0, 2.0, 3.0, 4.0, 4.0, 5.0, 5.0, 6.0, 7.0, 7.0, 8.0, 8.0, 9.0, 10.0, 10.0,
                         2.0, 2.0, 3.0, 4.0, 4.0, 5.0, 5.0, 6.0, 7.0, 7.0, 8.0, 8.0, 9.0, 10.0, 10.0,
                         2.0, 2.0, 3.0, 4.0, 4.0, 5.0, 5.0, 6.0, 7.0, 7.0, 8.0, 8.0, 9.0, 10.0, 10.0 };
    Tensor R = out.copyDims();
    R.allocateCustom( expected, 45 );
    return ops::Equals( out, R );
}

inline int testConv2DCase15()
{
    float input[] = { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 };
    Tensor<float> A{ 1, 3, 2 };
    A.allocateCustom( input, 6 );
    Conv2D<float> layer{ 1, 1, { 1, 2 }, { 1, 2 }, { 0, 0 }, false, PaddingMode::CIRCULAR };
    ops::Fill( layer.weights, 1.0F );

    const auto out = layer.forward( A );
    float expected[] = { 3.0, 7.0, 11.0 };
    Tensor R = out.copyDims();
    R.allocateCustom( expected, 3 );
    return ops::Equals( out, R );
}

inline int testConv2DCase16()
{
    float input[] = { 1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  8.0,  9.0,
                      10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0 };
    Tensor<float> A{ 3, 2, 3 };
    A.allocateCustom( input, 18 );
    Conv2D<float> layer{ 3, 1, { 2, 1 }, { 2, 1 }, { 1, 0 }, true, PaddingMode::ZEROS };
    ops::Fill( layer.weights, 1.0F );
    ops::Fill( layer.biases, 1.0F );
    const auto out = layer.forward( A );
    float expected[] = { 22.0, 25.0, 28.0, 31.0, 34.0, 37.0 };
    Tensor R = out.copyDims();
    R.allocateCustom( expected, 6 );
    return ops::Equals( out, R );
}

inline int testConv2DCase17()
{
    float input[] = { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0 };
    Tensor<float> A{ 1, 3, 3 };
    A.allocateCustom( input, 9 );
    Conv2D<float> layer{ 1, 3, { 1, 1 }, { 1, 1 }, { 0, 1 }, false, PaddingMode::REFLECT };
    ops::Fill( layer.weights, 1.0F );

    const auto out = layer.forward( A );
    float expected[] = { 2.0, 1.0, 2.0, 3.0, 2.0, 5.0, 4.0, 5.0, 6.0, 5.0, 8.0, 7.0, 8.0, 9.0, 8.0,
                         2.0, 1.0, 2.0, 3.0, 2.0, 5.0, 4.0, 5.0, 6.0, 5.0, 8.0, 7.0, 8.0, 9.0, 8.0,
                         2.0, 1.0, 2.0, 3.0, 2.0, 5.0, 4.0, 5.0, 6.0, 5.0, 8.0, 7.0, 8.0, 9.0, 8.0 };
    Tensor R = out.copyDims();
    R.allocateCustom( expected, 45 );
    return ops::Equals( out, R );
}

inline int testConv2DCase18()
{
    float input[] = { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 };
    Tensor<float> A{ 1, 3, 2 };
    A.allocateCustom( input, 6 );
    Conv2D<float> layer{ 1, 1, { 1, 2 }, { 1, 2 }, { 0, 0 }, true, PaddingMode::REPLICATE };
    ops::Fill( layer.weights, 1.0F );
    ops::Fill( layer.biases, 1.0F );
    const auto out = layer.forward( A );
    float expected[] = { 4.0, 8.0, 12.0 };
    Tensor R = out.copyDims();
    R.allocateCustom( expected, 3 );
    return ops::Equals( out, R );
}

inline int testConv2DCase19()
{
    float input[] = { 1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  8.0,  9.0,
                      10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0 };
    Tensor<float> A{ 3, 2, 3 };
    A.allocateCustom( input, 18 );
    Conv2D<float> layer{ 3, 1, { 2, 1 }, { 2, 1 }, { 1, 0 }, false, PaddingMode::CIRCULAR };
    ops::Fill( layer.weights, 1.0F );

    const auto out = layer.forward( A );
    float expected[] = { 51.0, 57.0, 63.0, 51.0, 57.0, 63.0 };
    Tensor R = out.copyDims();
    R.allocateCustom( expected, 6 );
    return ops::Equals( out, R );
}

inline int testConv2DCase20()
{
    float input[] = { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0 };
    Tensor<float> A{ 1, 3, 3 };
    A.allocateCustom( input, 9 );
    Conv2D<float> layer{ 1, 3, { 1, 1 }, { 1, 1 }, { 0, 1 }, true, PaddingMode::ZEROS };
    ops::Fill( layer.weights, 1.0F );
    ops::Fill( layer.biases, 1.0F );
    const auto out = layer.forward( A );
    float expected[] = { 1.0, 2.0, 3.0, 4.0, 1.0, 1.0, 5.0, 6.0, 7.0, 1.0, 1.0, 8.0, 9.0, 10.0, 1.0,
                         1.0, 2.0, 3.0, 4.0, 1.0, 1.0, 5.0, 6.0, 7.0, 1.0, 1.0, 8.0, 9.0, 10.0, 1.0,
                         1.0, 2.0, 3.0, 4.0, 1.0, 1.0, 5.0, 6.0, 7.0, 1.0, 1.0, 8.0, 9.0, 10.0, 1.0 };
    Tensor R = out.copyDims();
    R.allocateCustom( expected, 45 );
    return ops::Equals( out, R );
}

inline int testConv2DCase21()
{
    float input[] = { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 };
    Tensor<float> A{ 1, 3, 2 };
    A.allocateCustom( input, 6 );
    Conv2D<float> layer{ 1, 1, { 1, 2 }, { 1, 2 }, { 0, 0 }, false, PaddingMode::REFLECT };
    ops::Fill( layer.weights, 1.0F );

    const auto out = layer.forward( A );
    float expected[] = { 3.0, 7.0, 11.0 };
    Tensor R = out.copyDims();
    R.allocateCustom( expected, 3 );
    return ops::Equals( out, R );
}

inline int testConv2DCase22()
{
    float input[] = { 1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  8.0,  9.0,
                      10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0 };
    Tensor<float> A{ 3, 2, 3 };
    A.allocateCustom( input, 18 );
    Conv2D<float> layer{ 3, 1, { 2, 1 }, { 2, 1 }, { 1, 0 }, true, PaddingMode::REPLICATE };
    ops::Fill( layer.weights, 1.0F );
    ops::Fill( layer.biases, 1.0F );
    const auto out = layer.forward( A );
    float expected[] = { 43.0, 49.0, 55.0, 61.0, 67.0, 73.0 };
    Tensor R = out.copyDims();
    R.allocateCustom( expected, 6 );
    return ops::Equals( out, R );
}

inline int testConv2DCase23()
{
    float input[] = { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0 };
    Tensor<float> A{ 1, 3, 3 };
    A.allocateCustom( input, 9 );
    Conv2D<float> layer{ 1, 3, { 1, 1 }, { 1, 1 }, { 0, 1 }, false, PaddingMode::CIRCULAR };
    ops::Fill( layer.weights, 1.0F );

    const auto out = layer.forward( A );
    float expected[] = { 3.0, 1.0, 2.0, 3.0, 1.0, 6.0, 4.0, 5.0, 6.0, 4.0, 9.0, 7.0, 8.0, 9.0, 7.0,
                         3.0, 1.0, 2.0, 3.0, 1.0, 6.0, 4.0, 5.0, 6.0, 4.0, 9.0, 7.0, 8.0, 9.0, 7.0,
                         3.0, 1.0, 2.0, 3.0, 1.0, 6.0, 4.0, 5.0, 6.0, 4.0, 9.0, 7.0, 8.0, 9.0, 7.0 };
    Tensor R = out.copyDims();
    R.allocateCustom( expected, 45 );
    return ops::Equals( out, R );
}

inline int testConv2DCase24()
{
    float input[] = { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 };
    Tensor<float> A{ 1, 3, 2 };
    A.allocateCustom( input, 6 );
    Conv2D<float> layer{ 1, 1, { 1, 2 }, { 1, 2 }, { 0, 0 }, true, PaddingMode::ZEROS };
    ops::Fill( layer.weights, 1.0F );
    ops::Fill( layer.biases, 1.0F );
    const auto out = layer.forward( A );
    float expected[] = { 4.0, 8.0, 12.0 };
    Tensor R = out.copyDims();
    R.allocateCustom( expected, 3 );
    return ops::Equals( out, R );
}

inline int testConv2DCase25()
{
    float input[] = { 1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  8.0,  9.0,
                      10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0 };
    Tensor<float> A{ 3, 2, 3 };
    A.allocateCustom( input, 18 );
    Conv2D<float> layer{ 3, 1, { 2, 1 }, { 2, 1 }, { 1, 0 }, false, PaddingMode::REFLECT };
    ops::Fill( layer.weights, 1.0F );

    const auto out = layer.forward( A );
    float expected[] = { 51.0, 57.0, 63.0, 51.0, 57.0, 63.0 };
    Tensor R = out.copyDims();
    R.allocateCustom( expected, 6 );
    return ops::Equals( out, R );
}

inline int testConv2DCase26()
{
    float input[] = { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0 };
    Tensor<float> A{ 1, 3, 3 };
    A.allocateCustom( input, 9 );
    Conv2D<float> layer{ 1, 3, { 1, 1 }, { 1, 1 }, { 0, 1 }, true, PaddingMode::REPLICATE };
    ops::Fill( layer.weights, 1.0F );
    ops::Fill( layer.biases, 1.0F );
    const auto out = layer.forward( A );
    float expected[] = { 2.0, 2.0, 3.0, 4.0, 4.0, 5.0, 5.0, 6.0, 7.0, 7.0, 8.0, 8.0, 9.0, 10.0, 10.0,
                         2.0, 2.0, 3.0, 4.0, 4.0, 5.0, 5.0, 6.0, 7.0, 7.0, 8.0, 8.0, 9.0, 10.0, 10.0,
                         2.0, 2.0, 3.0, 4.0, 4.0, 5.0, 5.0, 6.0, 7.0, 7.0, 8.0, 8.0, 9.0, 10.0, 10.0 };
    Tensor R = out.copyDims();
    R.allocateCustom( expected, 45 );
    return ops::Equals( out, R );
}

inline int testConv2DCase27()
{
    float input[] = { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 };
    Tensor<float> A{ 1, 3, 2 };
    A.allocateCustom( input, 6 );
    Conv2D<float> layer{ 1, 1, { 1, 2 }, { 1, 2 }, { 0, 0 }, false, PaddingMode::CIRCULAR };
    ops::Fill( layer.weights, 1.0F );

    const auto out = layer.forward( A );
    float expected[] = { 3.0, 7.0, 11.0 };
    Tensor R = out.copyDims();
    R.allocateCustom( expected, 3 );
    return ops::Equals( out, R );
}

inline int testConv2DCase28()
{
    float input[] = { 1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  8.0,  9.0,
                      10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0 };
    Tensor<float> A{ 3, 2, 3 };
    A.allocateCustom( input, 18 );
    Conv2D<float> layer{ 3, 1, { 2, 1 }, { 2, 1 }, { 1, 0 }, true, PaddingMode::ZEROS };
    ops::Fill( layer.weights, 1.0F );
    ops::Fill( layer.biases, 1.0F );
    const auto out = layer.forward( A );
    float expected[] = { 22.0, 25.0, 28.0, 31.0, 34.0, 37.0 };
    Tensor R = out.copyDims();
    R.allocateCustom( expected, 6 );
    return ops::Equals( out, R );
}

inline int testConv2DCase29()
{
    float input[] = { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0 };
    Tensor<float> A{ 1, 3, 3 };
    A.allocateCustom( input, 9 );
    Conv2D<float> layer{ 1, 3, { 1, 1 }, { 1, 1 }, { 0, 1 }, false, PaddingMode::REFLECT };
    ops::Fill( layer.weights, 1.0F );

    const auto out = layer.forward( A );
    float expected[] = { 2.0, 1.0, 2.0, 3.0, 2.0, 5.0, 4.0, 5.0, 6.0, 5.0, 8.0, 7.0, 8.0, 9.0, 8.0,
                         2.0, 1.0, 2.0, 3.0, 2.0, 5.0, 4.0, 5.0, 6.0, 5.0, 8.0, 7.0, 8.0, 9.0, 8.0,
                         2.0, 1.0, 2.0, 3.0, 2.0, 5.0, 4.0, 5.0, 6.0, 5.0, 8.0, 7.0, 8.0, 9.0, 8.0 };
    Tensor R = out.copyDims();
    R.allocateCustom( expected, 45 );
    return ops::Equals( out, R );
}

inline int testConv2DCase30()
{
    float input[] = { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 };
    Tensor<float> A{ 1, 3, 2 };
    A.allocateCustom( input, 6 );
    Conv2D<float> layer{ 1, 1, { 1, 2 }, { 1, 2 }, { 0, 0 }, true, PaddingMode::REPLICATE };
    ops::Fill( layer.weights, 1.0F );
    ops::Fill( layer.biases, 1.0F );
    const auto out = layer.forward( A );
    float expected[] = { 4.0, 8.0, 12.0 };
    Tensor R = out.copyDims();
    R.allocateCustom( expected, 3 );
    return ops::Equals( out, R );
}

inline int testConv2DCase31()
{
    float input[] = { 1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  8.0,  9.0,
                      10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0 };
    Tensor<float> A{ 3, 2, 3 };
    A.allocateCustom( input, 18 );
    Conv2D<float> layer{ 3, 1, { 2, 1 }, { 2, 1 }, { 1, 0 }, false, PaddingMode::CIRCULAR };
    ops::Fill( layer.weights, 1.0F );

    const auto out = layer.forward( A );
    float expected[] = { 51.0, 57.0, 63.0, 51.0, 57.0, 63.0 };
    Tensor R = out.copyDims();
    R.allocateCustom( expected, 6 );
    return ops::Equals( out, R );
}

inline int testConv2DCase32()
{
    float input[] = { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0 };
    Tensor<float> A{ 1, 3, 3 };
    A.allocateCustom( input, 9 );
    Conv2D<float> layer{ 1, 3, { 1, 1 }, { 1, 1 }, { 0, 1 }, true, PaddingMode::ZEROS };
    ops::Fill( layer.weights, 1.0F );
    ops::Fill( layer.biases, 1.0F );
    const auto out = layer.forward( A );
    float expected[] = { 1.0, 2.0, 3.0, 4.0, 1.0, 1.0, 5.0, 6.0, 7.0, 1.0, 1.0, 8.0, 9.0, 10.0, 1.0,
                         1.0, 2.0, 3.0, 4.0, 1.0, 1.0, 5.0, 6.0, 7.0, 1.0, 1.0, 8.0, 9.0, 10.0, 1.0,
                         1.0, 2.0, 3.0, 4.0, 1.0, 1.0, 5.0, 6.0, 7.0, 1.0, 1.0, 8.0, 9.0, 10.0, 1.0 };
    Tensor R = out.copyDims();
    R.allocateCustom( expected, 45 );
    return ops::Equals( out, R );
}

inline int testConv2DCase33()
{
    float input[] = { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 };
    Tensor<float> A{ 1, 3, 2 };
    A.allocateCustom( input, 6 );
    Conv2D<float> layer{ 1, 1, { 1, 2 }, { 1, 2 }, { 0, 0 }, false, PaddingMode::REFLECT };
    ops::Fill( layer.weights, 1.0F );

    const auto out = layer.forward( A );
    float expected[] = { 3.0, 7.0, 11.0 };
    Tensor R = out.copyDims();
    R.allocateCustom( expected, 3 );
    return ops::Equals( out, R );
}

inline int testConv2DCase34()
{
    float input[] = { 1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  8.0,  9.0,
                      10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0 };
    Tensor<float> A{ 3, 2, 3 };
    A.allocateCustom( input, 18 );
    Conv2D<float> layer{ 3, 1, { 2, 1 }, { 2, 1 }, { 1, 0 }, true, PaddingMode::REPLICATE };
    ops::Fill( layer.weights, 1.0F );
    ops::Fill( layer.biases, 1.0F );
    const auto out = layer.forward( A );
    float expected[] = { 43.0, 49.0, 55.0, 61.0, 67.0, 73.0 };
    Tensor R = out.copyDims();
    R.allocateCustom( expected, 6 );
    return ops::Equals( out, R );
}

inline int testConv2DCase35()
{
    float input[] = { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0 };
    Tensor<float> A{ 1, 3, 3 };
    A.allocateCustom( input, 9 );
    Conv2D<float> layer{ 1, 3, { 1, 1 }, { 1, 1 }, { 0, 1 }, false, PaddingMode::CIRCULAR };
    ops::Fill( layer.weights, 1.0F );

    const auto out = layer.forward( A );
    float expected[] = { 3.0, 1.0, 2.0, 3.0, 1.0, 6.0, 4.0, 5.0, 6.0, 4.0, 9.0, 7.0, 8.0, 9.0, 7.0,
                         3.0, 1.0, 2.0, 3.0, 1.0, 6.0, 4.0, 5.0, 6.0, 4.0, 9.0, 7.0, 8.0, 9.0, 7.0,
                         3.0, 1.0, 2.0, 3.0, 1.0, 6.0, 4.0, 5.0, 6.0, 4.0, 9.0, 7.0, 8.0, 9.0, 7.0 };
    Tensor R = out.copyDims();
    R.allocateCustom( expected, 45 );
    return ops::Equals( out, R );
}

inline int testConv2DCase36()
{
    float input[] = { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 };
    Tensor<float> A{ 1, 3, 2 };
    A.allocateCustom( input, 6 );
    Conv2D<float> layer{ 1, 1, { 1, 2 }, { 1, 2 }, { 0, 0 }, true, PaddingMode::ZEROS };
    ops::Fill( layer.weights, 1.0F );
    ops::Fill( layer.biases, 1.0F );
    const auto out = layer.forward( A );
    float expected[] = { 4.0, 8.0, 12.0 };
    Tensor R = out.copyDims();
    R.allocateCustom( expected, 3 );
    return ops::Equals( out, R );
}

inline int testConv2DCase37()
{
    float input[] = { 1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  8.0,  9.0,
                      10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0 };
    Tensor<float> A{ 3, 2, 3 };
    A.allocateCustom( input, 18 );
    Conv2D<float> layer{ 3, 1, { 2, 1 }, { 2, 1 }, { 1, 0 }, false, PaddingMode::REFLECT };
    ops::Fill( layer.weights, 1.0F );

    const auto out = layer.forward( A );
    float expected[] = { 51.0, 57.0, 63.0, 51.0, 57.0, 63.0 };
    Tensor R = out.copyDims();
    R.allocateCustom( expected, 6 );
    return ops::Equals( out, R );
}

inline int testConv2DCase38()
{
    float input[] = { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0 };
    Tensor<float> A{ 1, 3, 3 };
    A.allocateCustom( input, 9 );
    Conv2D<float> layer{ 1, 3, { 1, 1 }, { 1, 1 }, { 0, 1 }, true, PaddingMode::REPLICATE };
    ops::Fill( layer.weights, 1.0F );
    ops::Fill( layer.biases, 1.0F );
    const auto out = layer.forward( A );
    float expected[] = { 2.0, 2.0, 3.0, 4.0, 4.0, 5.0, 5.0, 6.0, 7.0, 7.0, 8.0, 8.0, 9.0, 10.0, 10.0,
                         2.0, 2.0, 3.0, 4.0, 4.0, 5.0, 5.0, 6.0, 7.0, 7.0, 8.0, 8.0, 9.0, 10.0, 10.0,
                         2.0, 2.0, 3.0, 4.0, 4.0, 5.0, 5.0, 6.0, 7.0, 7.0, 8.0, 8.0, 9.0, 10.0, 10.0 };
    Tensor R = out.copyDims();
    R.allocateCustom( expected, 45 );
    return ops::Equals( out, R );
}

inline int testConv2DCase39()
{
    float input[] = { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 };
    Tensor<float> A{ 1, 3, 2 };
    A.allocateCustom( input, 6 );
    Conv2D<float> layer{ 1, 1, { 1, 2 }, { 1, 2 }, { 0, 0 }, false, PaddingMode::CIRCULAR };
    ops::Fill( layer.weights, 1.0F );

    const auto out = layer.forward( A );
    float expected[] = { 3.0, 7.0, 11.0 };
    Tensor R = out.copyDims();
    R.allocateCustom( expected, 3 );
    return ops::Equals( out, R );
}

inline int testConv2DCase40()
{
    float input[] = { 1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  8.0,  9.0,
                      10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0 };
    Tensor<float> A{ 3, 2, 3 };
    A.allocateCustom( input, 18 );
    Conv2D<float> layer{ 3, 1, { 2, 1 }, { 2, 1 }, { 1, 0 }, true, PaddingMode::ZEROS };
    ops::Fill( layer.weights, 1.0F );
    ops::Fill( layer.biases, 1.0F );
    const auto out = layer.forward( A );
    float expected[] = { 22.0, 25.0, 28.0, 31.0, 34.0, 37.0 };
    Tensor R = out.copyDims();
    R.allocateCustom( expected, 6 );
    return ops::Equals( out, R );
}

inline int testConv2DCase41()
{
    float input[] = { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0 };
    Tensor<float> A{ 1, 3, 3 };
    A.allocateCustom( input, 9 );
    Conv2D<float> layer{ 1, 3, { 1, 1 }, { 1, 1 }, { 0, 1 }, false, PaddingMode::REFLECT };
    ops::Fill( layer.weights, 1.0F );

    const auto out = layer.forward( A );
    float expected[] = { 2.0, 1.0, 2.0, 3.0, 2.0, 5.0, 4.0, 5.0, 6.0, 5.0, 8.0, 7.0, 8.0, 9.0, 8.0,
                         2.0, 1.0, 2.0, 3.0, 2.0, 5.0, 4.0, 5.0, 6.0, 5.0, 8.0, 7.0, 8.0, 9.0, 8.0,
                         2.0, 1.0, 2.0, 3.0, 2.0, 5.0, 4.0, 5.0, 6.0, 5.0, 8.0, 7.0, 8.0, 9.0, 8.0 };
    Tensor R = out.copyDims();
    R.allocateCustom( expected, 45 );
    return ops::Equals( out, R );
}

inline int testConv2DCase42()
{
    float input[] = { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 };
    Tensor<float> A{ 1, 3, 2 };
    A.allocateCustom( input, 6 );
    Conv2D<float> layer{ 1, 1, { 1, 2 }, { 1, 2 }, { 0, 0 }, true, PaddingMode::REPLICATE };
    ops::Fill( layer.weights, 1.0F );
    ops::Fill( layer.biases, 1.0F );
    const auto out = layer.forward( A );
    float expected[] = { 4.0, 8.0, 12.0 };
    Tensor R = out.copyDims();
    R.allocateCustom( expected, 3 );
    return ops::Equals( out, R );
}

inline int testConv2DCase43()
{
    float input[] = { 1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  8.0,  9.0,
                      10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0 };
    Tensor<float> A{ 3, 2, 3 };
    A.allocateCustom( input, 18 );
    Conv2D<float> layer{ 3, 1, { 2, 1 }, { 2, 1 }, { 1, 0 }, false, PaddingMode::CIRCULAR };
    ops::Fill( layer.weights, 1.0F );

    const auto out = layer.forward( A );
    float expected[] = { 51.0, 57.0, 63.0, 51.0, 57.0, 63.0 };
    Tensor R = out.copyDims();
    R.allocateCustom( expected, 6 );
    return ops::Equals( out, R );
}

inline int testConv2DCase44()
{
    float input[] = { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0 };
    Tensor<float> A{ 1, 3, 3 };
    A.allocateCustom( input, 9 );
    Conv2D<float> layer{ 1, 3, { 1, 1 }, { 1, 1 }, { 0, 1 }, true, PaddingMode::ZEROS };
    ops::Fill( layer.weights, 1.0F );
    ops::Fill( layer.biases, 1.0F );
    const auto out = layer.forward( A );
    float expected[] = { 1.0, 2.0, 3.0, 4.0, 1.0, 1.0, 5.0, 6.0, 7.0, 1.0, 1.0, 8.0, 9.0, 10.0, 1.0,
                         1.0, 2.0, 3.0, 4.0, 1.0, 1.0, 5.0, 6.0, 7.0, 1.0, 1.0, 8.0, 9.0, 10.0, 1.0,
                         1.0, 2.0, 3.0, 4.0, 1.0, 1.0, 5.0, 6.0, 7.0, 1.0, 1.0, 8.0, 9.0, 10.0, 1.0 };
    Tensor R = out.copyDims();
    R.allocateCustom( expected, 45 );
    return ops::Equals( out, R );
}

inline int testConv2DCase45()
{
    float input[] = { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 };
    Tensor<float> A{ 1, 3, 2 };
    A.allocateCustom( input, 6 );
    Conv2D<float> layer{ 1, 1, { 1, 2 }, { 1, 2 }, { 0, 0 }, false, PaddingMode::REFLECT };
    ops::Fill( layer.weights, 1.0F );

    const auto out = layer.forward( A );
    float expected[] = { 3.0, 7.0, 11.0 };
    Tensor R = out.copyDims();
    R.allocateCustom( expected, 3 );
    return ops::Equals( out, R );
}

inline int testConv2DCase46()
{
    float input[] = { 1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  8.0,  9.0,
                      10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0 };
    Tensor<float> A{ 3, 2, 3 };
    A.allocateCustom( input, 18 );
    Conv2D<float> layer{ 3, 1, { 2, 1 }, { 2, 1 }, { 1, 0 }, true, PaddingMode::REPLICATE };
    ops::Fill( layer.weights, 1.0F );
    ops::Fill( layer.biases, 1.0F );
    const auto out = layer.forward( A );
    float expected[] = { 43.0, 49.0, 55.0, 61.0, 67.0, 73.0 };
    Tensor R = out.copyDims();
    R.allocateCustom( expected, 6 );
    return ops::Equals( out, R );
}

inline int testConv2DCase47()
{
    float input[] = { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0 };
    Tensor<float> A{ 1, 3, 3 };
    A.allocateCustom( input, 9 );
    Conv2D<float> layer{ 1, 3, { 1, 1 }, { 1, 1 }, { 0, 1 }, false, PaddingMode::CIRCULAR };
    ops::Fill( layer.weights, 1.0F );

    const auto out = layer.forward( A );
    float expected[] = { 3.0, 1.0, 2.0, 3.0, 1.0, 6.0, 4.0, 5.0, 6.0, 4.0, 9.0, 7.0, 8.0, 9.0, 7.0,
                         3.0, 1.0, 2.0, 3.0, 1.0, 6.0, 4.0, 5.0, 6.0, 4.0, 9.0, 7.0, 8.0, 9.0, 7.0,
                         3.0, 1.0, 2.0, 3.0, 1.0, 6.0, 4.0, 5.0, 6.0, 4.0, 9.0, 7.0, 8.0, 9.0, 7.0 };
    Tensor R = out.copyDims();
    R.allocateCustom( expected, 45 );
    return ops::Equals( out, R );
}

inline int testConv2DCase48()
{
    float input[] = { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 };
    Tensor<float> A{ 1, 3, 2 };
    A.allocateCustom( input, 6 );
    Conv2D<float> layer{ 1, 1, { 1, 2 }, { 1, 2 }, { 0, 0 }, true, PaddingMode::ZEROS };
    ops::Fill( layer.weights, 1.0F );
    ops::Fill( layer.biases, 1.0F );
    const auto out = layer.forward( A );
    float expected[] = { 4.0, 8.0, 12.0 };
    Tensor R = out.copyDims();
    R.allocateCustom( expected, 3 );
    return ops::Equals( out, R );
}

inline int testConv2DCase49()
{
    float input[] = { 1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  8.0,  9.0,
                      10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0 };
    Tensor<float> A{ 3, 2, 3 };
    A.allocateCustom( input, 18 );
    Conv2D<float> layer{ 3, 1, { 2, 1 }, { 2, 1 }, { 1, 0 }, false, PaddingMode::REFLECT };
    ops::Fill( layer.weights, 1.0F );

    const auto out = layer.forward( A );
    float expected[] = { 51.0, 57.0, 63.0, 51.0, 57.0, 63.0 };
    Tensor R = out.copyDims();
    R.allocateCustom( expected, 6 );
    return ops::Equals( out, R );
}

inline int testConv2DCase50()
{
    float input[] = { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0 };
    Tensor<float> A{ 1, 3, 3 };
    A.allocateCustom( input, 9 );
    Conv2D<float> layer{ 1, 3, { 1, 1 }, { 1, 1 }, { 0, 1 }, true, PaddingMode::REPLICATE };
    ops::Fill( layer.weights, 1.0F );
    ops::Fill( layer.biases, 1.0F );
    const auto out = layer.forward( A );
    float expected[] = { 2.0, 2.0, 3.0, 4.0, 4.0, 5.0, 5.0, 6.0, 7.0, 7.0, 8.0, 8.0, 9.0, 10.0, 10.0,
                         2.0, 2.0, 3.0, 4.0, 4.0, 5.0, 5.0, 6.0, 7.0, 7.0, 8.0, 8.0, 9.0, 10.0, 10.0,
                         2.0, 2.0, 3.0, 4.0, 4.0, 5.0, 5.0, 6.0, 7.0, 7.0, 8.0, 8.0, 9.0, 10.0, 10.0 };
    Tensor R = out.copyDims();
    R.allocateCustom( expected, 45 );
    return ops::Equals( out, R );
}

inline int testConv2DCase51()
{
    float input[] = { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 };
    Tensor<float> A{ 1, 3, 2 };
    A.allocateCustom( input, 6 );
    Conv2D<float> layer{ 1, 1, { 1, 2 }, { 1, 2 }, { 0, 0 }, false, PaddingMode::CIRCULAR };
    ops::Fill( layer.weights, 1.0F );

    const auto out = layer.forward( A );
    float expected[] = { 3.0, 7.0, 11.0 };
    Tensor R = out.copyDims();
    R.allocateCustom( expected, 3 );
    return ops::Equals( out, R );
}

inline int testConv2DCase52()
{
    float input[] = { 1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  8.0,  9.0,
                      10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0 };
    Tensor<float> A{ 3, 2, 3 };
    A.allocateCustom( input, 18 );
    Conv2D<float> layer{ 3, 1, { 2, 1 }, { 2, 1 }, { 1, 0 }, true, PaddingMode::ZEROS };
    ops::Fill( layer.weights, 1.0F );
    ops::Fill( layer.biases, 1.0F );
    const auto out = layer.forward( A );
    float expected[] = { 22.0, 25.0, 28.0, 31.0, 34.0, 37.0 };
    Tensor R = out.copyDims();
    R.allocateCustom( expected, 6 );
    return ops::Equals( out, R );
}

inline int testConv2DCase53()
{
    float input[] = { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0 };
    Tensor<float> A{ 1, 3, 3 };
    A.allocateCustom( input, 9 );
    Conv2D<float> layer{ 1, 3, { 1, 1 }, { 1, 1 }, { 0, 1 }, false, PaddingMode::REFLECT };
    ops::Fill( layer.weights, 1.0F );

    const auto out = layer.forward( A );
    float expected[] = { 2.0, 1.0, 2.0, 3.0, 2.0, 5.0, 4.0, 5.0, 6.0, 5.0, 8.0, 7.0, 8.0, 9.0, 8.0,
                         2.0, 1.0, 2.0, 3.0, 2.0, 5.0, 4.0, 5.0, 6.0, 5.0, 8.0, 7.0, 8.0, 9.0, 8.0,
                         2.0, 1.0, 2.0, 3.0, 2.0, 5.0, 4.0, 5.0, 6.0, 5.0, 8.0, 7.0, 8.0, 9.0, 8.0 };
    Tensor R = out.copyDims();
    R.allocateCustom( expected, 45 );
    return ops::Equals( out, R );
}

inline int testConv2DCase54()
{
    float input[] = { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 };
    Tensor<float> A{ 1, 3, 2 };
    A.allocateCustom( input, 6 );
    Conv2D<float> layer{ 1, 1, { 1, 2 }, { 1, 2 }, { 0, 0 }, true, PaddingMode::REPLICATE };
    ops::Fill( layer.weights, 1.0F );
    ops::Fill( layer.biases, 1.0F );
    const auto out = layer.forward( A );
    float expected[] = { 4.0, 8.0, 12.0 };
    Tensor R = out.copyDims();
    R.allocateCustom( expected, 3 );
    return ops::Equals( out, R );
}

inline int testConv2DCase55()
{
    float input[] = { 1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  8.0,  9.0,
                      10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0 };
    Tensor<float> A{ 3, 2, 3 };
    A.allocateCustom( input, 18 );
    Conv2D<float> layer{ 3, 1, { 2, 1 }, { 2, 1 }, { 1, 0 }, false, PaddingMode::CIRCULAR };
    ops::Fill( layer.weights, 1.0F );

    const auto out = layer.forward( A );
    float expected[] = { 51.0, 57.0, 63.0, 51.0, 57.0, 63.0 };
    Tensor R = out.copyDims();
    R.allocateCustom( expected, 6 );
    return ops::Equals( out, R );
}

inline int testConv2DCase56()
{
    float input[] = { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0 };
    Tensor<float> A{ 1, 3, 3 };
    A.allocateCustom( input, 9 );
    Conv2D<float> layer{ 1, 3, { 1, 1 }, { 1, 1 }, { 0, 1 }, true, PaddingMode::ZEROS };
    ops::Fill( layer.weights, 1.0F );
    ops::Fill( layer.biases, 1.0F );
    const auto out = layer.forward( A );
    float expected[] = { 1.0, 2.0, 3.0, 4.0, 1.0, 1.0, 5.0, 6.0, 7.0, 1.0, 1.0, 8.0, 9.0, 10.0, 1.0,
                         1.0, 2.0, 3.0, 4.0, 1.0, 1.0, 5.0, 6.0, 7.0, 1.0, 1.0, 8.0, 9.0, 10.0, 1.0,
                         1.0, 2.0, 3.0, 4.0, 1.0, 1.0, 5.0, 6.0, 7.0, 1.0, 1.0, 8.0, 9.0, 10.0, 1.0 };
    Tensor R = out.copyDims();
    R.allocateCustom( expected, 45 );
    return ops::Equals( out, R );
}

inline int testConv2DCase57()
{
    float input[] = { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 };
    Tensor<float> A{ 1, 3, 2 };
    A.allocateCustom( input, 6 );
    Conv2D<float> layer{ 1, 1, { 1, 2 }, { 1, 2 }, { 0, 0 }, false, PaddingMode::REFLECT };
    ops::Fill( layer.weights, 1.0F );

    const auto out = layer.forward( A );
    float expected[] = { 3.0, 7.0, 11.0 };
    Tensor R = out.copyDims();
    R.allocateCustom( expected, 3 );
    return ops::Equals( out, R );
}

inline int testConv2DCase58()
{
    float input[] = { 1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  8.0,  9.0,
                      10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0 };
    Tensor<float> A{ 3, 2, 3 };
    A.allocateCustom( input, 18 );
    Conv2D<float> layer{ 3, 1, { 2, 1 }, { 2, 1 }, { 1, 0 }, true, PaddingMode::REPLICATE };
    ops::Fill( layer.weights, 1.0F );
    ops::Fill( layer.biases, 1.0F );
    const auto out = layer.forward( A );
    float expected[] = { 43.0, 49.0, 55.0, 61.0, 67.0, 73.0 };
    Tensor R = out.copyDims();
    R.allocateCustom( expected, 6 );
    return ops::Equals( out, R );
}

inline int testConv2DCase59()
{
    float input[] = { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0 };
    Tensor<float> A{ 1, 3, 3 };
    A.allocateCustom( input, 9 );
    Conv2D<float> layer{ 1, 3, { 1, 1 }, { 1, 1 }, { 0, 1 }, false, PaddingMode::CIRCULAR };
    ops::Fill( layer.weights, 1.0F );

    const auto out = layer.forward( A );
    float expected[] = { 3.0, 1.0, 2.0, 3.0, 1.0, 6.0, 4.0, 5.0, 6.0, 4.0, 9.0, 7.0, 8.0, 9.0, 7.0,
                         3.0, 1.0, 2.0, 3.0, 1.0, 6.0, 4.0, 5.0, 6.0, 4.0, 9.0, 7.0, 8.0, 9.0, 7.0,
                         3.0, 1.0, 2.0, 3.0, 1.0, 6.0, 4.0, 5.0, 6.0, 4.0, 9.0, 7.0, 8.0, 9.0, 7.0 };
    Tensor R = out.copyDims();
    R.allocateCustom( expected, 45 );
    return ops::Equals( out, R );
}

inline int testConv2DCase60()
{
    float input[] = { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 };
    Tensor<float> A{ 1, 3, 2 };
    A.allocateCustom( input, 6 );
    Conv2D<float> layer{ 1, 1, { 1, 2 }, { 1, 2 }, { 0, 0 }, true, PaddingMode::ZEROS };
    ops::Fill( layer.weights, 1.0F );
    ops::Fill( layer.biases, 1.0F );
    const auto out = layer.forward( A );
    float expected[] = { 4.0, 8.0, 12.0 };
    Tensor R = out.copyDims();
    R.allocateCustom( expected, 3 );
    return ops::Equals( out, R );
}

inline int testConv2DCase61()
{
    float input[] = { 1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  8.0,  9.0,
                      10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0 };
    Tensor<float> A{ 3, 2, 3 };
    A.allocateCustom( input, 18 );
    Conv2D<float> layer{ 3, 1, { 2, 1 }, { 2, 1 }, { 1, 0 }, false, PaddingMode::REFLECT };
    ops::Fill( layer.weights, 1.0F );

    const auto out = layer.forward( A );
    float expected[] = { 51.0, 57.0, 63.0, 51.0, 57.0, 63.0 };
    Tensor R = out.copyDims();
    R.allocateCustom( expected, 6 );
    return ops::Equals( out, R );
}

inline int testConv2DCase62()
{
    float input[] = { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0 };
    Tensor<float> A{ 1, 3, 3 };
    A.allocateCustom( input, 9 );
    Conv2D<float> layer{ 1, 3, { 1, 1 }, { 1, 1 }, { 0, 1 }, true, PaddingMode::REPLICATE };
    ops::Fill( layer.weights, 1.0F );
    ops::Fill( layer.biases, 1.0F );
    const auto out = layer.forward( A );
    float expected[] = { 2.0, 2.0, 3.0, 4.0, 4.0, 5.0, 5.0, 6.0, 7.0, 7.0, 8.0, 8.0, 9.0, 10.0, 10.0,
                         2.0, 2.0, 3.0, 4.0, 4.0, 5.0, 5.0, 6.0, 7.0, 7.0, 8.0, 8.0, 9.0, 10.0, 10.0,
                         2.0, 2.0, 3.0, 4.0, 4.0, 5.0, 5.0, 6.0, 7.0, 7.0, 8.0, 8.0, 9.0, 10.0, 10.0 };
    Tensor R = out.copyDims();
    R.allocateCustom( expected, 45 );
    return ops::Equals( out, R );
}

inline int testConv2DCase63()
{
    float input[] = { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 };
    Tensor<float> A{ 1, 3, 2 };
    A.allocateCustom( input, 6 );
    Conv2D<float> layer{ 1, 1, { 1, 2 }, { 1, 2 }, { 0, 0 }, false, PaddingMode::CIRCULAR };
    ops::Fill( layer.weights, 1.0F );

    const auto out = layer.forward( A );
    float expected[] = { 3.0, 7.0, 11.0 };
    Tensor R = out.copyDims();
    R.allocateCustom( expected, 3 );
    return ops::Equals( out, R );
}

inline int testConv2DCase64()
{
    float input[] = { 1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  8.0,  9.0,
                      10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0 };
    Tensor<float> A{ 3, 2, 3 };
    A.allocateCustom( input, 18 );
    Conv2D<float> layer{ 3, 1, { 2, 1 }, { 2, 1 }, { 1, 0 }, true, PaddingMode::ZEROS };
    ops::Fill( layer.weights, 1.0F );
    ops::Fill( layer.biases, 1.0F );
    const auto out = layer.forward( A );
    float expected[] = { 22.0, 25.0, 28.0, 31.0, 34.0, 37.0 };
    Tensor R = out.copyDims();
    R.allocateCustom( expected, 6 );
    return ops::Equals( out, R );
}

inline int testConv2DCase65()
{
    float input[] = { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0 };
    Tensor<float> A{ 1, 3, 3 };
    A.allocateCustom( input, 9 );
    Conv2D<float> layer{ 1, 3, { 1, 1 }, { 1, 1 }, { 0, 1 }, false, PaddingMode::REFLECT };
    ops::Fill( layer.weights, 1.0F );

    const auto out = layer.forward( A );
    float expected[] = { 2.0, 1.0, 2.0, 3.0, 2.0, 5.0, 4.0, 5.0, 6.0, 5.0, 8.0, 7.0, 8.0, 9.0, 8.0,
                         2.0, 1.0, 2.0, 3.0, 2.0, 5.0, 4.0, 5.0, 6.0, 5.0, 8.0, 7.0, 8.0, 9.0, 8.0,
                         2.0, 1.0, 2.0, 3.0, 2.0, 5.0, 4.0, 5.0, 6.0, 5.0, 8.0, 7.0, 8.0, 9.0, 8.0 };
    Tensor R = out.copyDims();
    R.allocateCustom( expected, 45 );
    return ops::Equals( out, R );
}

inline int testConv2DCase66()
{
    float input[] = { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 };
    Tensor<float> A{ 1, 3, 2 };
    A.allocateCustom( input, 6 );
    Conv2D<float> layer{ 1, 1, { 1, 2 }, { 1, 2 }, { 0, 0 }, true, PaddingMode::REPLICATE };
    ops::Fill( layer.weights, 1.0F );
    ops::Fill( layer.biases, 1.0F );
    const auto out = layer.forward( A );
    float expected[] = { 4.0, 8.0, 12.0 };
    Tensor R = out.copyDims();
    R.allocateCustom( expected, 3 );
    return ops::Equals( out, R );
}

inline int testConv2DCase67()
{
    float input[] = { 1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  8.0,  9.0,
                      10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0 };
    Tensor<float> A{ 3, 2, 3 };
    A.allocateCustom( input, 18 );
    Conv2D<float> layer{ 3, 1, { 2, 1 }, { 2, 1 }, { 1, 0 }, false, PaddingMode::CIRCULAR };
    ops::Fill( layer.weights, 1.0F );

    const auto out = layer.forward( A );
    float expected[] = { 51.0, 57.0, 63.0, 51.0, 57.0, 63.0 };
    Tensor R = out.copyDims();
    R.allocateCustom( expected, 6 );
    return ops::Equals( out, R );
}

inline int testConv2DCase68()
{
    float input[] = { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0 };
    Tensor<float> A{ 1, 3, 3 };
    A.allocateCustom( input, 9 );
    Conv2D<float> layer{ 1, 3, { 1, 1 }, { 1, 1 }, { 0, 1 }, true, PaddingMode::ZEROS };
    ops::Fill( layer.weights, 1.0F );
    ops::Fill( layer.biases, 1.0F );
    const auto out = layer.forward( A );
    float expected[] = { 1.0, 2.0, 3.0, 4.0, 1.0, 1.0, 5.0, 6.0, 7.0, 1.0, 1.0, 8.0, 9.0, 10.0, 1.0,
                         1.0, 2.0, 3.0, 4.0, 1.0, 1.0, 5.0, 6.0, 7.0, 1.0, 1.0, 8.0, 9.0, 10.0, 1.0,
                         1.0, 2.0, 3.0, 4.0, 1.0, 1.0, 5.0, 6.0, 7.0, 1.0, 1.0, 8.0, 9.0, 10.0, 1.0 };
    Tensor R = out.copyDims();
    R.allocateCustom( expected, 45 );
    return ops::Equals( out, R );
}

inline int testConv2DCase69()
{
    float input[] = { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 };
    Tensor<float> A{ 1, 3, 2 };
    A.allocateCustom( input, 6 );
    Conv2D<float> layer{ 1, 1, { 1, 2 }, { 1, 2 }, { 0, 0 }, false, PaddingMode::REFLECT };
    ops::Fill( layer.weights, 1.0F );

    const auto out = layer.forward( A );
    float expected[] = { 3.0, 7.0, 11.0 };
    Tensor R = out.copyDims();
    R.allocateCustom( expected, 3 );
    return ops::Equals( out, R );
}

inline int testConv2DCase70()
{
    float input[] = { 1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  8.0,  9.0,
                      10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0 };
    Tensor<float> A{ 3, 2, 3 };
    A.allocateCustom( input, 18 );
    Conv2D<float> layer{ 3, 1, { 2, 1 }, { 2, 1 }, { 1, 0 }, true, PaddingMode::REPLICATE };
    ops::Fill( layer.weights, 1.0F );
    ops::Fill( layer.biases, 1.0F );
    const auto out = layer.forward( A );
    float expected[] = { 43.0, 49.0, 55.0, 61.0, 67.0, 73.0 };
    Tensor R = out.copyDims();
    R.allocateCustom( expected, 6 );
    return ops::Equals( out, R );
}

inline int testConv2DCase71()
{
    float input[] = { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0 };
    Tensor<float> A{ 1, 3, 3 };
    A.allocateCustom( input, 9 );
    Conv2D<float> layer{ 1, 3, { 1, 1 }, { 1, 1 }, { 0, 1 }, false, PaddingMode::CIRCULAR };
    ops::Fill( layer.weights, 1.0F );

    const auto out = layer.forward( A );
    float expected[] = { 3.0, 1.0, 2.0, 3.0, 1.0, 6.0, 4.0, 5.0, 6.0, 4.0, 9.0, 7.0, 8.0, 9.0, 7.0,
                         3.0, 1.0, 2.0, 3.0, 1.0, 6.0, 4.0, 5.0, 6.0, 4.0, 9.0, 7.0, 8.0, 9.0, 7.0,
                         3.0, 1.0, 2.0, 3.0, 1.0, 6.0, 4.0, 5.0, 6.0, 4.0, 9.0, 7.0, 8.0, 9.0, 7.0 };
    Tensor R = out.copyDims();
    R.allocateCustom( expected, 45 );
    return ops::Equals( out, R );
}

inline int testConv2DCase72()
{
    float input[] = { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 };
    Tensor<float> A{ 1, 3, 2 };
    A.allocateCustom( input, 6 );
    Conv2D<float> layer{ 1, 1, { 1, 2 }, { 1, 2 }, { 0, 0 }, true, PaddingMode::ZEROS };
    ops::Fill( layer.weights, 1.0F );
    ops::Fill( layer.biases, 1.0F );
    const auto out = layer.forward( A );
    float expected[] = { 4.0, 8.0, 12.0 };
    Tensor R = out.copyDims();
    R.allocateCustom( expected, 3 );
    return ops::Equals( out, R );
}

inline int testConv2DCase73()
{
    float input[] = { 1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  8.0,  9.0,
                      10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0 };
    Tensor<float> A{ 3, 2, 3 };
    A.allocateCustom( input, 18 );
    Conv2D<float> layer{ 3, 1, { 2, 1 }, { 2, 1 }, { 1, 0 }, false, PaddingMode::REFLECT };
    ops::Fill( layer.weights, 1.0F );

    const auto out = layer.forward( A );
    float expected[] = { 51.0, 57.0, 63.0, 51.0, 57.0, 63.0 };
    Tensor R = out.copyDims();
    R.allocateCustom( expected, 6 );
    return ops::Equals( out, R );
}

inline int testConv2DCase74()
{
    float input[] = { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0 };
    Tensor<float> A{ 1, 3, 3 };
    A.allocateCustom( input, 9 );
    Conv2D<float> layer{ 1, 3, { 1, 1 }, { 1, 1 }, { 0, 1 }, true, PaddingMode::REPLICATE };
    ops::Fill( layer.weights, 1.0F );
    ops::Fill( layer.biases, 1.0F );
    const auto out = layer.forward( A );
    float expected[] = { 2.0, 2.0, 3.0, 4.0, 4.0, 5.0, 5.0, 6.0, 7.0, 7.0, 8.0, 8.0, 9.0, 10.0, 10.0,
                         2.0, 2.0, 3.0, 4.0, 4.0, 5.0, 5.0, 6.0, 7.0, 7.0, 8.0, 8.0, 9.0, 10.0, 10.0,
                         2.0, 2.0, 3.0, 4.0, 4.0, 5.0, 5.0, 6.0, 7.0, 7.0, 8.0, 8.0, 9.0, 10.0, 10.0 };
    Tensor R = out.copyDims();
    R.allocateCustom( expected, 45 );
    return ops::Equals( out, R );
}

inline int testConv2DCase75()
{
    float input[] = { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 };
    Tensor<float> A{ 1, 3, 2 };
    A.allocateCustom( input, 6 );
    Conv2D<float> layer{ 1, 1, { 1, 2 }, { 1, 2 }, { 0, 0 }, false, PaddingMode::CIRCULAR };
    ops::Fill( layer.weights, 1.0F );

    const auto out = layer.forward( A );
    float expected[] = { 3.0, 7.0, 11.0 };
    Tensor R = out.copyDims();
    R.allocateCustom( expected, 3 );
    return ops::Equals( out, R );
}

inline void TestNNLayersConv2D()
{
    EML_RUN_TEST( testConv2DCase1 );
    EML_RUN_TEST( testConv2DCase2 );
    EML_RUN_TEST( testConv2DCase3 );
    EML_RUN_TEST( testConv2DCase4 );
    EML_RUN_TEST( testConv2DCase5 );
    EML_RUN_TEST( testConv2DCase6 );
    EML_RUN_TEST( testConv2DCase7 );
    EML_RUN_TEST( testConv2DCase8 );
    EML_RUN_TEST( testConv2DCase9 );
    EML_RUN_TEST( testConv2DCase10 );
    EML_RUN_TEST( testConv2DCase11 );
    EML_RUN_TEST( testConv2DCase12 );
    EML_RUN_TEST( testConv2DCase13 );
    EML_RUN_TEST( testConv2DCase14 );
    EML_RUN_TEST( testConv2DCase15 );
    EML_RUN_TEST( testConv2DCase16 );
    EML_RUN_TEST( testConv2DCase17 );
    EML_RUN_TEST( testConv2DCase18 );
    EML_RUN_TEST( testConv2DCase19 );
    EML_RUN_TEST( testConv2DCase20 );
    EML_RUN_TEST( testConv2DCase21 );
    EML_RUN_TEST( testConv2DCase22 );
    EML_RUN_TEST( testConv2DCase23 );
    EML_RUN_TEST( testConv2DCase24 );
    EML_RUN_TEST( testConv2DCase25 );
    EML_RUN_TEST( testConv2DCase26 );
    EML_RUN_TEST( testConv2DCase27 );
    EML_RUN_TEST( testConv2DCase28 );
    EML_RUN_TEST( testConv2DCase29 );
    EML_RUN_TEST( testConv2DCase30 );
    EML_RUN_TEST( testConv2DCase31 );
    EML_RUN_TEST( testConv2DCase32 );
    EML_RUN_TEST( testConv2DCase33 );
    EML_RUN_TEST( testConv2DCase34 );
    EML_RUN_TEST( testConv2DCase35 );
    EML_RUN_TEST( testConv2DCase36 );
    EML_RUN_TEST( testConv2DCase37 );
    EML_RUN_TEST( testConv2DCase38 );
    EML_RUN_TEST( testConv2DCase39 );
    EML_RUN_TEST( testConv2DCase40 );
    EML_RUN_TEST( testConv2DCase41 );
    EML_RUN_TEST( testConv2DCase42 );
    EML_RUN_TEST( testConv2DCase43 );
    EML_RUN_TEST( testConv2DCase44 );
    EML_RUN_TEST( testConv2DCase45 );
    EML_RUN_TEST( testConv2DCase46 );
    EML_RUN_TEST( testConv2DCase47 );
    EML_RUN_TEST( testConv2DCase48 );
    EML_RUN_TEST( testConv2DCase49 );
    EML_RUN_TEST( testConv2DCase50 );
    EML_RUN_TEST( testConv2DCase51 );
    EML_RUN_TEST( testConv2DCase52 );
    EML_RUN_TEST( testConv2DCase53 );
    EML_RUN_TEST( testConv2DCase54 );
    EML_RUN_TEST( testConv2DCase55 );
    EML_RUN_TEST( testConv2DCase56 );
    EML_RUN_TEST( testConv2DCase57 );
    EML_RUN_TEST( testConv2DCase58 );
    EML_RUN_TEST( testConv2DCase59 );
    EML_RUN_TEST( testConv2DCase60 );
    EML_RUN_TEST( testConv2DCase61 );
    EML_RUN_TEST( testConv2DCase62 );
    EML_RUN_TEST( testConv2DCase63 );
    EML_RUN_TEST( testConv2DCase64 );
    EML_RUN_TEST( testConv2DCase65 );
    EML_RUN_TEST( testConv2DCase66 );
    EML_RUN_TEST( testConv2DCase67 );
    EML_RUN_TEST( testConv2DCase68 );
    EML_RUN_TEST( testConv2DCase69 );
    EML_RUN_TEST( testConv2DCase70 );
    EML_RUN_TEST( testConv2DCase71 );
    EML_RUN_TEST( testConv2DCase72 );
    EML_RUN_TEST( testConv2DCase73 );
    EML_RUN_TEST( testConv2DCase74 );
    EML_RUN_TEST( testConv2DCase75 );
}

#endif // EML_TEST_LAYERS_CONV2D_H