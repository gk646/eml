
#ifndef EML_TEST_NN_LAYERS_CONV2D_H
#define EML_TEST_NN_LAYERS_CONV2D_H

inline int testConv2DCase0()
{
    qint32_t input[] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36 };
    Tensor<qint32_t> A{ 3, 3, 4 };
A.allocateCustom(input, 36);
    Conv2D<qint32_t> layer{ 3, 1, { 1, 2 }, { 2, 1 }, { 0, 1 }, true, PaddingMode::CIRCULAR };
    ops::Fill(layer.weights, 1);
    ops::Fill(layer.biases, 1);
    const auto out = layer.forward(A);
    qint32_t expected[] = { 88, 82, 88, 94, 88, 136, 130, 136, 142, 136 };
    Tensor<qint32_t> R{out.shape()};
    R.allocateCustom(expected, 10);
    return ops::Equals(out, R);

}
inline int testConv2DCase1()
{
    qint32_t input[] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 };
    Tensor<qint32_t> A{ 1, 4, 4 };
A.allocateCustom(input, 16);
    Conv2D<qint32_t> layer{ 1, 3, { 1, 1 }, { 1, 2 }, { 2, 2 }, false, PaddingMode::REPLICATE };
    ops::Fill(layer.weights, 1);
    
    const auto out = layer.forward(A);
    qint32_t expected[] = { 1, 1, 3, 4, 1, 1, 3, 4, 1, 1, 3, 4, 5, 5, 7, 8, 9, 9, 11, 12, 13, 13, 15, 16, 13, 13, 15, 16, 13, 13, 15, 16, 1, 1, 3, 4, 1, 1, 3, 4, 1, 1, 3, 4, 5, 5, 7, 8, 9, 9, 11, 12, 13, 13, 15, 16, 13, 13, 15, 16, 13, 13, 15, 16, 1, 1, 3, 4, 1, 1, 3, 4, 1, 1, 3, 4, 5, 5, 7, 8, 9, 9, 11, 12, 13, 13, 15, 16, 13, 13, 15, 16, 13, 13, 15, 16 };
    Tensor<qint32_t> R{out.shape()};
    R.allocateCustom(expected, 96);
    return ops::Equals(out, R);

}
inline int testConv2DCase2()
{
    qint32_t input[] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 };
    Tensor<qint32_t> A{ 1, 4, 4 };
A.allocateCustom(input, 16);
    Conv2D<qint32_t> layer{ 1, 3, { 1, 1 }, { 1, 2 }, { 0, 0 }, true, PaddingMode::ZEROS };
    ops::Fill(layer.weights, 1);
    ops::Fill(layer.biases, 1);
    const auto out = layer.forward(A);
    qint32_t expected[] = { 2, 4, 6, 8, 10, 12, 14, 16, 2, 4, 6, 8, 10, 12, 14, 16, 2, 4, 6, 8, 10, 12, 14, 16 };
    Tensor<qint32_t> R{out.shape()};
    R.allocateCustom(expected, 24);
    return ops::Equals(out, R);

}
inline int testConv2DCase3()
{
    qint32_t input[] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36 };
    Tensor<qint32_t> A{ 3, 3, 4 };
A.allocateCustom(input, 36);
    Conv2D<qint32_t> layer{ 3, 1, { 1, 2 }, { 2, 1 }, { 1, 0 }, false, PaddingMode::REFLECT };
    ops::Fill(layer.weights, 1);
    
    const auto out = layer.forward(A);
    qint32_t expected[] = { 105, 111, 117, 105, 111, 117, 105, 111, 117 };
    Tensor<qint32_t> R{out.shape()};
    R.allocateCustom(expected, 9);
    return ops::Equals(out, R);

}
inline int testConv2DCase4()
{
    qint32_t input[] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 };
    Tensor<qint32_t> A{ 1, 4, 3 };
A.allocateCustom(input, 12);
    Conv2D<qint32_t> layer{ 1, 1, { 2, 1 }, { 1, 1 }, { 0, 1 }, true, PaddingMode::CIRCULAR };
    ops::Fill(layer.weights, 1);
    ops::Fill(layer.biases, 1);
    const auto out = layer.forward(A);
    qint32_t expected[] = { 10, 6, 8, 10, 6, 16, 12, 14, 16, 12, 22, 18, 20, 22, 18 };
    Tensor<qint32_t> R{out.shape()};
    R.allocateCustom(expected, 15);
    return ops::Equals(out, R);

}
inline int testConv2DCase5()
{
    qint32_t input[] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 };
    Tensor<qint32_t> A{ 1, 4, 3 };
A.allocateCustom(input, 12);
    Conv2D<qint32_t> layer{ 1, 1, { 2, 1 }, { 1, 1 }, { 2, 2 }, false, PaddingMode::REPLICATE };
    ops::Fill(layer.weights, 1);
    
    const auto out = layer.forward(A);
    qint32_t expected[] = { 2, 2, 2, 4, 6, 6, 6, 2, 2, 2, 4, 6, 6, 6, 5, 5, 5, 7, 9, 9, 9, 11, 11, 11, 13, 15, 15, 15, 17, 17, 17, 19, 21, 21, 21, 20, 20, 20, 22, 24, 24, 24, 20, 20, 20, 22, 24, 24, 24 };
    Tensor<qint32_t> R{out.shape()};
    R.allocateCustom(expected, 49);
    return ops::Equals(out, R);

}
inline int testConv2DCase6()
{
    qint32_t input[] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36 };
    Tensor<qint32_t> A{ 3, 3, 4 };
A.allocateCustom(input, 36);
    Conv2D<qint32_t> layer{ 3, 1, { 1, 2 }, { 2, 1 }, { 0, 0 }, true, PaddingMode::ZEROS };
    ops::Fill(layer.weights, 1);
    ops::Fill(layer.biases, 1);
    const auto out = layer.forward(A);
    qint32_t expected[] = { 82, 88, 94, 130, 136, 142 };
    Tensor<qint32_t> R{out.shape()};
    R.allocateCustom(expected, 6);
    return ops::Equals(out, R);

}
inline int testConv2DCase7()
{
    qint32_t input[] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 };
    Tensor<qint32_t> A{ 1, 4, 4 };
A.allocateCustom(input, 16);
    Conv2D<qint32_t> layer{ 1, 3, { 1, 1 }, { 1, 2 }, { 1, 0 }, false, PaddingMode::REFLECT };
    ops::Fill(layer.weights, 1);
    
    const auto out = layer.forward(A);
    qint32_t expected[] = { 5, 7, 1, 3, 5, 7, 9, 11, 13, 15, 9, 11, 5, 7, 1, 3, 5, 7, 9, 11, 13, 15, 9, 11, 5, 7, 1, 3, 5, 7, 9, 11, 13, 15, 9, 11 };
    Tensor<qint32_t> R{out.shape()};
    R.allocateCustom(expected, 36);
    return ops::Equals(out, R);

}
inline int testConv2DCase8()
{
    qint32_t input[] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 };
    Tensor<qint32_t> A{ 1, 4, 4 };
A.allocateCustom(input, 16);
    Conv2D<qint32_t> layer{ 1, 3, { 1, 1 }, { 1, 2 }, { 0, 1 }, true, PaddingMode::CIRCULAR };
    ops::Fill(layer.weights, 1);
    ops::Fill(layer.biases, 1);
    const auto out = layer.forward(A);
    qint32_t expected[] = { 5, 3, 5, 9, 7, 9, 13, 11, 13, 17, 15, 17, 5, 3, 5, 9, 7, 9, 13, 11, 13, 17, 15, 17, 5, 3, 5, 9, 7, 9, 13, 11, 13, 17, 15, 17 };
    Tensor<qint32_t> R{out.shape()};
    R.allocateCustom(expected, 36);
    return ops::Equals(out, R);

}
inline int testConv2DCase9()
{
    qint32_t input[] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36 };
    Tensor<qint32_t> A{ 3, 3, 4 };
A.allocateCustom(input, 36);
    Conv2D<qint32_t> layer{ 3, 1, { 1, 2 }, { 2, 1 }, { 2, 2 }, false, PaddingMode::REPLICATE };
    ops::Fill(layer.weights, 1);
    
    const auto out = layer.forward(A);
    qint32_t expected[] = { 78, 78, 81, 87, 93, 96, 96, 78, 78, 81, 87, 93, 96, 96, 126, 126, 129, 135, 141, 144, 144, 126, 126, 129, 135, 141, 144, 144 };
    Tensor<qint32_t> R{out.shape()};
    R.allocateCustom(expected, 28);
    return ops::Equals(out, R);

}
inline int testConv2DCase10()
{
    qint32_t input[] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 };
    Tensor<qint32_t> A{ 1, 4, 3 };
A.allocateCustom(input, 12);
    Conv2D<qint32_t> layer{ 1, 1, { 2, 1 }, { 1, 1 }, { 0, 0 }, true, PaddingMode::ZEROS };
    ops::Fill(layer.weights, 1);
    ops::Fill(layer.biases, 1);
    const auto out = layer.forward(A);
    qint32_t expected[] = { 6, 8, 10, 12, 14, 16, 18, 20, 22 };
    Tensor<qint32_t> R{out.shape()};
    R.allocateCustom(expected, 9);
    return ops::Equals(out, R);

}
inline int testConv2DCase11()
{
    qint32_t input[] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 };
    Tensor<qint32_t> A{ 1, 4, 3 };
A.allocateCustom(input, 12);
    Conv2D<qint32_t> layer{ 1, 1, { 2, 1 }, { 1, 1 }, { 1, 0 }, false, PaddingMode::REFLECT };
    ops::Fill(layer.weights, 1);
    
    const auto out = layer.forward(A);
    qint32_t expected[] = { 5, 7, 9, 5, 7, 9, 11, 13, 15, 17, 19, 21, 17, 19, 21 };
    Tensor<qint32_t> R{out.shape()};
    R.allocateCustom(expected, 15);
    return ops::Equals(out, R);

}
inline int testConv2DCase12()
{
    qint32_t input[] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36 };
    Tensor<qint32_t> A{ 3, 3, 4 };
A.allocateCustom(input, 36);
    Conv2D<qint32_t> layer{ 3, 1, { 1, 2 }, { 2, 1 }, { 0, 1 }, true, PaddingMode::CIRCULAR };
    ops::Fill(layer.weights, 1);
    ops::Fill(layer.biases, 1);
    const auto out = layer.forward(A);
    qint32_t expected[] = { 88, 82, 88, 94, 88, 136, 130, 136, 142, 136 };
    Tensor<qint32_t> R{out.shape()};
    R.allocateCustom(expected, 10);
    return ops::Equals(out, R);

}
inline int testConv2DCase13()
{
    qint32_t input[] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 };
    Tensor<qint32_t> A{ 1, 4, 4 };
A.allocateCustom(input, 16);
    Conv2D<qint32_t> layer{ 1, 3, { 1, 1 }, { 1, 2 }, { 2, 2 }, false, PaddingMode::REPLICATE };
    ops::Fill(layer.weights, 1);
    
    const auto out = layer.forward(A);
    qint32_t expected[] = { 1, 1, 3, 4, 1, 1, 3, 4, 1, 1, 3, 4, 5, 5, 7, 8, 9, 9, 11, 12, 13, 13, 15, 16, 13, 13, 15, 16, 13, 13, 15, 16, 1, 1, 3, 4, 1, 1, 3, 4, 1, 1, 3, 4, 5, 5, 7, 8, 9, 9, 11, 12, 13, 13, 15, 16, 13, 13, 15, 16, 13, 13, 15, 16, 1, 1, 3, 4, 1, 1, 3, 4, 1, 1, 3, 4, 5, 5, 7, 8, 9, 9, 11, 12, 13, 13, 15, 16, 13, 13, 15, 16, 13, 13, 15, 16 };
    Tensor<qint32_t> R{out.shape()};
    R.allocateCustom(expected, 96);
    return ops::Equals(out, R);

}
inline int testConv2DCase14()
{
    qint32_t input[] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 };
    Tensor<qint32_t> A{ 1, 4, 4 };
A.allocateCustom(input, 16);
    Conv2D<qint32_t> layer{ 1, 3, { 1, 1 }, { 1, 2 }, { 0, 0 }, true, PaddingMode::ZEROS };
    ops::Fill(layer.weights, 1);
    ops::Fill(layer.biases, 1);
    const auto out = layer.forward(A);
    qint32_t expected[] = { 2, 4, 6, 8, 10, 12, 14, 16, 2, 4, 6, 8, 10, 12, 14, 16, 2, 4, 6, 8, 10, 12, 14, 16 };
    Tensor<qint32_t> R{out.shape()};
    R.allocateCustom(expected, 24);
    return ops::Equals(out, R);

}
inline int testConv2DCase15()
{
    qint32_t input[] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36 };
    Tensor<qint32_t> A{ 3, 3, 4 };
A.allocateCustom(input, 36);
    Conv2D<qint32_t> layer{ 3, 1, { 1, 2 }, { 2, 1 }, { 1, 0 }, false, PaddingMode::REFLECT };
    ops::Fill(layer.weights, 1);
    
    const auto out = layer.forward(A);
    qint32_t expected[] = { 105, 111, 117, 105, 111, 117, 105, 111, 117 };
    Tensor<qint32_t> R{out.shape()};
    R.allocateCustom(expected, 9);
    return ops::Equals(out, R);

}
inline int testConv2DCase16()
{
    qint32_t input[] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 };
    Tensor<qint32_t> A{ 1, 4, 3 };
A.allocateCustom(input, 12);
    Conv2D<qint32_t> layer{ 1, 1, { 2, 1 }, { 1, 1 }, { 0, 1 }, true, PaddingMode::CIRCULAR };
    ops::Fill(layer.weights, 1);
    ops::Fill(layer.biases, 1);
    const auto out = layer.forward(A);
    qint32_t expected[] = { 10, 6, 8, 10, 6, 16, 12, 14, 16, 12, 22, 18, 20, 22, 18 };
    Tensor<qint32_t> R{out.shape()};
    R.allocateCustom(expected, 15);
    return ops::Equals(out, R);

}
inline int testConv2DCase17()
{
    qint32_t input[] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 };
    Tensor<qint32_t> A{ 1, 4, 3 };
A.allocateCustom(input, 12);
    Conv2D<qint32_t> layer{ 1, 1, { 2, 1 }, { 1, 1 }, { 2, 2 }, false, PaddingMode::REPLICATE };
    ops::Fill(layer.weights, 1);
    
    const auto out = layer.forward(A);
    qint32_t expected[] = { 2, 2, 2, 4, 6, 6, 6, 2, 2, 2, 4, 6, 6, 6, 5, 5, 5, 7, 9, 9, 9, 11, 11, 11, 13, 15, 15, 15, 17, 17, 17, 19, 21, 21, 21, 20, 20, 20, 22, 24, 24, 24, 20, 20, 20, 22, 24, 24, 24 };
    Tensor<qint32_t> R{out.shape()};
    R.allocateCustom(expected, 49);
    return ops::Equals(out, R);

}
inline int testConv2DCase18()
{
    qint32_t input[] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36 };
    Tensor<qint32_t> A{ 3, 3, 4 };
A.allocateCustom(input, 36);
    Conv2D<qint32_t> layer{ 3, 1, { 1, 2 }, { 2, 1 }, { 0, 0 }, true, PaddingMode::ZEROS };
    ops::Fill(layer.weights, 1);
    ops::Fill(layer.biases, 1);
    const auto out = layer.forward(A);
    qint32_t expected[] = { 82, 88, 94, 130, 136, 142 };
    Tensor<qint32_t> R{out.shape()};
    R.allocateCustom(expected, 6);
    return ops::Equals(out, R);

}
inline int testConv2DCase19()
{
    qint32_t input[] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 };
    Tensor<qint32_t> A{ 1, 4, 4 };
A.allocateCustom(input, 16);
    Conv2D<qint32_t> layer{ 1, 3, { 1, 1 }, { 1, 2 }, { 1, 0 }, false, PaddingMode::REFLECT };
    ops::Fill(layer.weights, 1);
    
    const auto out = layer.forward(A);
    qint32_t expected[] = { 5, 7, 1, 3, 5, 7, 9, 11, 13, 15, 9, 11, 5, 7, 1, 3, 5, 7, 9, 11, 13, 15, 9, 11, 5, 7, 1, 3, 5, 7, 9, 11, 13, 15, 9, 11 };
    Tensor<qint32_t> R{out.shape()};
    R.allocateCustom(expected, 36);
    return ops::Equals(out, R);

}
inline int testConv2DCase20()
{
    qint32_t input[] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 };
    Tensor<qint32_t> A{ 1, 4, 4 };
A.allocateCustom(input, 16);
    Conv2D<qint32_t> layer{ 1, 3, { 1, 1 }, { 1, 2 }, { 0, 1 }, true, PaddingMode::CIRCULAR };
    ops::Fill(layer.weights, 1);
    ops::Fill(layer.biases, 1);
    const auto out = layer.forward(A);
    qint32_t expected[] = { 5, 3, 5, 9, 7, 9, 13, 11, 13, 17, 15, 17, 5, 3, 5, 9, 7, 9, 13, 11, 13, 17, 15, 17, 5, 3, 5, 9, 7, 9, 13, 11, 13, 17, 15, 17 };
    Tensor<qint32_t> R{out.shape()};
    R.allocateCustom(expected, 36);
    return ops::Equals(out, R);

}
inline int testConv2DCase21()
{
    qint32_t input[] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36 };
    Tensor<qint32_t> A{ 3, 3, 4 };
A.allocateCustom(input, 36);
    Conv2D<qint32_t> layer{ 3, 1, { 1, 2 }, { 2, 1 }, { 2, 2 }, false, PaddingMode::REPLICATE };
    ops::Fill(layer.weights, 1);
    
    const auto out = layer.forward(A);
    qint32_t expected[] = { 78, 78, 81, 87, 93, 96, 96, 78, 78, 81, 87, 93, 96, 96, 126, 126, 129, 135, 141, 144, 144, 126, 126, 129, 135, 141, 144, 144 };
    Tensor<qint32_t> R{out.shape()};
    R.allocateCustom(expected, 28);
    return ops::Equals(out, R);

}
inline int testConv2DCase22()
{
    qint32_t input[] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 };
    Tensor<qint32_t> A{ 1, 4, 3 };
A.allocateCustom(input, 12);
    Conv2D<qint32_t> layer{ 1, 1, { 2, 1 }, { 1, 1 }, { 0, 0 }, true, PaddingMode::ZEROS };
    ops::Fill(layer.weights, 1);
    ops::Fill(layer.biases, 1);
    const auto out = layer.forward(A);
    qint32_t expected[] = { 6, 8, 10, 12, 14, 16, 18, 20, 22 };
    Tensor<qint32_t> R{out.shape()};
    R.allocateCustom(expected, 9);
    return ops::Equals(out, R);

}
inline int testConv2DCase23()
{
    qint32_t input[] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 };
    Tensor<qint32_t> A{ 1, 4, 3 };
A.allocateCustom(input, 12);
    Conv2D<qint32_t> layer{ 1, 1, { 2, 1 }, { 1, 1 }, { 1, 0 }, false, PaddingMode::REFLECT };
    ops::Fill(layer.weights, 1);
    
    const auto out = layer.forward(A);
    qint32_t expected[] = { 5, 7, 9, 5, 7, 9, 11, 13, 15, 17, 19, 21, 17, 19, 21 };
    Tensor<qint32_t> R{out.shape()};
    R.allocateCustom(expected, 15);
    return ops::Equals(out, R);

}
inline int testConv2DCase24()
{
    qint32_t input[] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36 };
    Tensor<qint32_t> A{ 3, 3, 4 };
A.allocateCustom(input, 36);
    Conv2D<qint32_t> layer{ 3, 1, { 1, 2 }, { 2, 1 }, { 0, 1 }, true, PaddingMode::CIRCULAR };
    ops::Fill(layer.weights, 1);
    ops::Fill(layer.biases, 1);
    const auto out = layer.forward(A);
    qint32_t expected[] = { 88, 82, 88, 94, 88, 136, 130, 136, 142, 136 };
    Tensor<qint32_t> R{out.shape()};
    R.allocateCustom(expected, 10);
    return ops::Equals(out, R);

}
inline int testConv2DCase25()
{
    qint32_t input[] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 };
    Tensor<qint32_t> A{ 1, 4, 4 };
A.allocateCustom(input, 16);
    Conv2D<qint32_t> layer{ 1, 3, { 1, 1 }, { 1, 2 }, { 2, 2 }, false, PaddingMode::REPLICATE };
    ops::Fill(layer.weights, 1);
    
    const auto out = layer.forward(A);
    qint32_t expected[] = { 1, 1, 3, 4, 1, 1, 3, 4, 1, 1, 3, 4, 5, 5, 7, 8, 9, 9, 11, 12, 13, 13, 15, 16, 13, 13, 15, 16, 13, 13, 15, 16, 1, 1, 3, 4, 1, 1, 3, 4, 1, 1, 3, 4, 5, 5, 7, 8, 9, 9, 11, 12, 13, 13, 15, 16, 13, 13, 15, 16, 13, 13, 15, 16, 1, 1, 3, 4, 1, 1, 3, 4, 1, 1, 3, 4, 5, 5, 7, 8, 9, 9, 11, 12, 13, 13, 15, 16, 13, 13, 15, 16, 13, 13, 15, 16 };
    Tensor<qint32_t> R{out.shape()};
    R.allocateCustom(expected, 96);
    return ops::Equals(out, R);

}
inline int testConv2DCase26()
{
    qint32_t input[] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 };
    Tensor<qint32_t> A{ 1, 4, 4 };
A.allocateCustom(input, 16);
    Conv2D<qint32_t> layer{ 1, 3, { 1, 1 }, { 1, 2 }, { 0, 0 }, true, PaddingMode::ZEROS };
    ops::Fill(layer.weights, 1);
    ops::Fill(layer.biases, 1);
    const auto out = layer.forward(A);
    qint32_t expected[] = { 2, 4, 6, 8, 10, 12, 14, 16, 2, 4, 6, 8, 10, 12, 14, 16, 2, 4, 6, 8, 10, 12, 14, 16 };
    Tensor<qint32_t> R{out.shape()};
    R.allocateCustom(expected, 24);
    return ops::Equals(out, R);

}
inline int testConv2DCase27()
{
    qint32_t input[] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36 };
    Tensor<qint32_t> A{ 3, 3, 4 };
A.allocateCustom(input, 36);
    Conv2D<qint32_t> layer{ 3, 1, { 1, 2 }, { 2, 1 }, { 1, 0 }, false, PaddingMode::REFLECT };
    ops::Fill(layer.weights, 1);
    
    const auto out = layer.forward(A);
    qint32_t expected[] = { 105, 111, 117, 105, 111, 117, 105, 111, 117 };
    Tensor<qint32_t> R{out.shape()};
    R.allocateCustom(expected, 9);
    return ops::Equals(out, R);

}
inline int testConv2DCase28()
{
    qint32_t input[] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 };
    Tensor<qint32_t> A{ 1, 4, 3 };
A.allocateCustom(input, 12);
    Conv2D<qint32_t> layer{ 1, 1, { 2, 1 }, { 1, 1 }, { 0, 1 }, true, PaddingMode::CIRCULAR };
    ops::Fill(layer.weights, 1);
    ops::Fill(layer.biases, 1);
    const auto out = layer.forward(A);
    qint32_t expected[] = { 10, 6, 8, 10, 6, 16, 12, 14, 16, 12, 22, 18, 20, 22, 18 };
    Tensor<qint32_t> R{out.shape()};
    R.allocateCustom(expected, 15);
    return ops::Equals(out, R);

}
inline int testConv2DCase29()
{
    qint32_t input[] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 };
    Tensor<qint32_t> A{ 1, 4, 3 };
A.allocateCustom(input, 12);
    Conv2D<qint32_t> layer{ 1, 1, { 2, 1 }, { 1, 1 }, { 2, 2 }, false, PaddingMode::REPLICATE };
    ops::Fill(layer.weights, 1);
    
    const auto out = layer.forward(A);
    qint32_t expected[] = { 2, 2, 2, 4, 6, 6, 6, 2, 2, 2, 4, 6, 6, 6, 5, 5, 5, 7, 9, 9, 9, 11, 11, 11, 13, 15, 15, 15, 17, 17, 17, 19, 21, 21, 21, 20, 20, 20, 22, 24, 24, 24, 20, 20, 20, 22, 24, 24, 24 };
    Tensor<qint32_t> R{out.shape()};
    R.allocateCustom(expected, 49);
    return ops::Equals(out, R);

}
inline int testConv2DCase30()
{
    qint32_t input[] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36 };
    Tensor<qint32_t> A{ 3, 3, 4 };
A.allocateCustom(input, 36);
    Conv2D<qint32_t> layer{ 3, 1, { 1, 2 }, { 2, 1 }, { 0, 0 }, true, PaddingMode::ZEROS };
    ops::Fill(layer.weights, 1);
    ops::Fill(layer.biases, 1);
    const auto out = layer.forward(A);
    qint32_t expected[] = { 82, 88, 94, 130, 136, 142 };
    Tensor<qint32_t> R{out.shape()};
    R.allocateCustom(expected, 6);
    return ops::Equals(out, R);

}
inline int testConv2DCase31()
{
    qint32_t input[] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 };
    Tensor<qint32_t> A{ 1, 4, 4 };
A.allocateCustom(input, 16);
    Conv2D<qint32_t> layer{ 1, 3, { 1, 1 }, { 1, 2 }, { 1, 0 }, false, PaddingMode::REFLECT };
    ops::Fill(layer.weights, 1);
    
    const auto out = layer.forward(A);
    qint32_t expected[] = { 5, 7, 1, 3, 5, 7, 9, 11, 13, 15, 9, 11, 5, 7, 1, 3, 5, 7, 9, 11, 13, 15, 9, 11, 5, 7, 1, 3, 5, 7, 9, 11, 13, 15, 9, 11 };
    Tensor<qint32_t> R{out.shape()};
    R.allocateCustom(expected, 36);
    return ops::Equals(out, R);

}
inline int testConv2DCase32()
{
    qint32_t input[] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 };
    Tensor<qint32_t> A{ 1, 4, 4 };
A.allocateCustom(input, 16);
    Conv2D<qint32_t> layer{ 1, 3, { 1, 1 }, { 1, 2 }, { 0, 1 }, true, PaddingMode::CIRCULAR };
    ops::Fill(layer.weights, 1);
    ops::Fill(layer.biases, 1);
    const auto out = layer.forward(A);
    qint32_t expected[] = { 5, 3, 5, 9, 7, 9, 13, 11, 13, 17, 15, 17, 5, 3, 5, 9, 7, 9, 13, 11, 13, 17, 15, 17, 5, 3, 5, 9, 7, 9, 13, 11, 13, 17, 15, 17 };
    Tensor<qint32_t> R{out.shape()};
    R.allocateCustom(expected, 36);
    return ops::Equals(out, R);

}
inline int testConv2DCase33()
{
    qint32_t input[] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36 };
    Tensor<qint32_t> A{ 3, 3, 4 };
A.allocateCustom(input, 36);
    Conv2D<qint32_t> layer{ 3, 1, { 1, 2 }, { 2, 1 }, { 2, 2 }, false, PaddingMode::REPLICATE };
    ops::Fill(layer.weights, 1);
    
    const auto out = layer.forward(A);
    qint32_t expected[] = { 78, 78, 81, 87, 93, 96, 96, 78, 78, 81, 87, 93, 96, 96, 126, 126, 129, 135, 141, 144, 144, 126, 126, 129, 135, 141, 144, 144 };
    Tensor<qint32_t> R{out.shape()};
    R.allocateCustom(expected, 28);
    return ops::Equals(out, R);

}
inline int testConv2DCase34()
{
    qint32_t input[] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 };
    Tensor<qint32_t> A{ 1, 4, 3 };
A.allocateCustom(input, 12);
    Conv2D<qint32_t> layer{ 1, 1, { 2, 1 }, { 1, 1 }, { 0, 0 }, true, PaddingMode::ZEROS };
    ops::Fill(layer.weights, 1);
    ops::Fill(layer.biases, 1);
    const auto out = layer.forward(A);
    qint32_t expected[] = { 6, 8, 10, 12, 14, 16, 18, 20, 22 };
    Tensor<qint32_t> R{out.shape()};
    R.allocateCustom(expected, 9);
    return ops::Equals(out, R);

}
inline int testConv2DCase35()
{
    qint32_t input[] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 };
    Tensor<qint32_t> A{ 1, 4, 3 };
A.allocateCustom(input, 12);
    Conv2D<qint32_t> layer{ 1, 1, { 2, 1 }, { 1, 1 }, { 1, 0 }, false, PaddingMode::REFLECT };
    ops::Fill(layer.weights, 1);
    
    const auto out = layer.forward(A);
    qint32_t expected[] = { 5, 7, 9, 5, 7, 9, 11, 13, 15, 17, 19, 21, 17, 19, 21 };
    Tensor<qint32_t> R{out.shape()};
    R.allocateCustom(expected, 15);
    return ops::Equals(out, R);

}
inline int testConv2DCase36()
{
    qint32_t input[] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36 };
    Tensor<qint32_t> A{ 3, 3, 4 };
A.allocateCustom(input, 36);
    Conv2D<qint32_t> layer{ 3, 1, { 1, 2 }, { 2, 1 }, { 0, 1 }, true, PaddingMode::CIRCULAR };
    ops::Fill(layer.weights, 1);
    ops::Fill(layer.biases, 1);
    const auto out = layer.forward(A);
    qint32_t expected[] = { 88, 82, 88, 94, 88, 136, 130, 136, 142, 136 };
    Tensor<qint32_t> R{out.shape()};
    R.allocateCustom(expected, 10);
    return ops::Equals(out, R);

}
inline int testConv2DCase37()
{
    qint32_t input[] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 };
    Tensor<qint32_t> A{ 1, 4, 4 };
A.allocateCustom(input, 16);
    Conv2D<qint32_t> layer{ 1, 3, { 1, 1 }, { 1, 2 }, { 2, 2 }, false, PaddingMode::REPLICATE };
    ops::Fill(layer.weights, 1);
    
    const auto out = layer.forward(A);
    qint32_t expected[] = { 1, 1, 3, 4, 1, 1, 3, 4, 1, 1, 3, 4, 5, 5, 7, 8, 9, 9, 11, 12, 13, 13, 15, 16, 13, 13, 15, 16, 13, 13, 15, 16, 1, 1, 3, 4, 1, 1, 3, 4, 1, 1, 3, 4, 5, 5, 7, 8, 9, 9, 11, 12, 13, 13, 15, 16, 13, 13, 15, 16, 13, 13, 15, 16, 1, 1, 3, 4, 1, 1, 3, 4, 1, 1, 3, 4, 5, 5, 7, 8, 9, 9, 11, 12, 13, 13, 15, 16, 13, 13, 15, 16, 13, 13, 15, 16 };
    Tensor<qint32_t> R{out.shape()};
    R.allocateCustom(expected, 96);
    return ops::Equals(out, R);

}
inline int testConv2DCase38()
{
    qint32_t input[] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 };
    Tensor<qint32_t> A{ 1, 4, 4 };
A.allocateCustom(input, 16);
    Conv2D<qint32_t> layer{ 1, 3, { 1, 1 }, { 1, 2 }, { 0, 0 }, true, PaddingMode::ZEROS };
    ops::Fill(layer.weights, 1);
    ops::Fill(layer.biases, 1);
    const auto out = layer.forward(A);
    qint32_t expected[] = { 2, 4, 6, 8, 10, 12, 14, 16, 2, 4, 6, 8, 10, 12, 14, 16, 2, 4, 6, 8, 10, 12, 14, 16 };
    Tensor<qint32_t> R{out.shape()};
    R.allocateCustom(expected, 24);
    return ops::Equals(out, R);

}
inline int testConv2DCase39()
{
    qint32_t input[] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36 };
    Tensor<qint32_t> A{ 3, 3, 4 };
A.allocateCustom(input, 36);
    Conv2D<qint32_t> layer{ 3, 1, { 1, 2 }, { 2, 1 }, { 1, 0 }, false, PaddingMode::REFLECT };
    ops::Fill(layer.weights, 1);
    
    const auto out = layer.forward(A);
    qint32_t expected[] = { 105, 111, 117, 105, 111, 117, 105, 111, 117 };
    Tensor<qint32_t> R{out.shape()};
    R.allocateCustom(expected, 9);
    return ops::Equals(out, R);

}
inline int testConv2DCase40()
{
    qint32_t input[] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 };
    Tensor<qint32_t> A{ 1, 4, 3 };
A.allocateCustom(input, 12);
    Conv2D<qint32_t> layer{ 1, 1, { 2, 1 }, { 1, 1 }, { 0, 1 }, true, PaddingMode::CIRCULAR };
    ops::Fill(layer.weights, 1);
    ops::Fill(layer.biases, 1);
    const auto out = layer.forward(A);
    qint32_t expected[] = { 10, 6, 8, 10, 6, 16, 12, 14, 16, 12, 22, 18, 20, 22, 18 };
    Tensor<qint32_t> R{out.shape()};
    R.allocateCustom(expected, 15);
    return ops::Equals(out, R);

}
inline int testConv2DCase41()
{
    qint32_t input[] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 };
    Tensor<qint32_t> A{ 1, 4, 3 };
A.allocateCustom(input, 12);
    Conv2D<qint32_t> layer{ 1, 1, { 2, 1 }, { 1, 1 }, { 2, 2 }, false, PaddingMode::REPLICATE };
    ops::Fill(layer.weights, 1);
    
    const auto out = layer.forward(A);
    qint32_t expected[] = { 2, 2, 2, 4, 6, 6, 6, 2, 2, 2, 4, 6, 6, 6, 5, 5, 5, 7, 9, 9, 9, 11, 11, 11, 13, 15, 15, 15, 17, 17, 17, 19, 21, 21, 21, 20, 20, 20, 22, 24, 24, 24, 20, 20, 20, 22, 24, 24, 24 };
    Tensor<qint32_t> R{out.shape()};
    R.allocateCustom(expected, 49);
    return ops::Equals(out, R);

}
inline int testConv2DCase42()
{
    qint32_t input[] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36 };
    Tensor<qint32_t> A{ 3, 3, 4 };
A.allocateCustom(input, 36);
    Conv2D<qint32_t> layer{ 3, 1, { 1, 2 }, { 2, 1 }, { 0, 0 }, true, PaddingMode::ZEROS };
    ops::Fill(layer.weights, 1);
    ops::Fill(layer.biases, 1);
    const auto out = layer.forward(A);
    qint32_t expected[] = { 82, 88, 94, 130, 136, 142 };
    Tensor<qint32_t> R{out.shape()};
    R.allocateCustom(expected, 6);
    return ops::Equals(out, R);

}
inline int testConv2DCase43()
{
    qint32_t input[] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 };
    Tensor<qint32_t> A{ 1, 4, 4 };
A.allocateCustom(input, 16);
    Conv2D<qint32_t> layer{ 1, 3, { 1, 1 }, { 1, 2 }, { 1, 0 }, false, PaddingMode::REFLECT };
    ops::Fill(layer.weights, 1);
    
    const auto out = layer.forward(A);
    qint32_t expected[] = { 5, 7, 1, 3, 5, 7, 9, 11, 13, 15, 9, 11, 5, 7, 1, 3, 5, 7, 9, 11, 13, 15, 9, 11, 5, 7, 1, 3, 5, 7, 9, 11, 13, 15, 9, 11 };
    Tensor<qint32_t> R{out.shape()};
    R.allocateCustom(expected, 36);
    return ops::Equals(out, R);

}
inline int testConv2DCase44()
{
    qint32_t input[] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 };
    Tensor<qint32_t> A{ 1, 4, 4 };
A.allocateCustom(input, 16);
    Conv2D<qint32_t> layer{ 1, 3, { 1, 1 }, { 1, 2 }, { 0, 1 }, true, PaddingMode::CIRCULAR };
    ops::Fill(layer.weights, 1);
    ops::Fill(layer.biases, 1);
    const auto out = layer.forward(A);
    qint32_t expected[] = { 5, 3, 5, 9, 7, 9, 13, 11, 13, 17, 15, 17, 5, 3, 5, 9, 7, 9, 13, 11, 13, 17, 15, 17, 5, 3, 5, 9, 7, 9, 13, 11, 13, 17, 15, 17 };
    Tensor<qint32_t> R{out.shape()};
    R.allocateCustom(expected, 36);
    return ops::Equals(out, R);

}
inline int testConv2DCase45()
{
    qint32_t input[] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36 };
    Tensor<qint32_t> A{ 3, 3, 4 };
A.allocateCustom(input, 36);
    Conv2D<qint32_t> layer{ 3, 1, { 1, 2 }, { 2, 1 }, { 2, 2 }, false, PaddingMode::REPLICATE };
    ops::Fill(layer.weights, 1);
    
    const auto out = layer.forward(A);
    qint32_t expected[] = { 78, 78, 81, 87, 93, 96, 96, 78, 78, 81, 87, 93, 96, 96, 126, 126, 129, 135, 141, 144, 144, 126, 126, 129, 135, 141, 144, 144 };
    Tensor<qint32_t> R{out.shape()};
    R.allocateCustom(expected, 28);
    return ops::Equals(out, R);

}
inline int testConv2DCase46()
{
    qint32_t input[] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 };
    Tensor<qint32_t> A{ 1, 4, 3 };
A.allocateCustom(input, 12);
    Conv2D<qint32_t> layer{ 1, 1, { 2, 1 }, { 1, 1 }, { 0, 0 }, true, PaddingMode::ZEROS };
    ops::Fill(layer.weights, 1);
    ops::Fill(layer.biases, 1);
    const auto out = layer.forward(A);
    qint32_t expected[] = { 6, 8, 10, 12, 14, 16, 18, 20, 22 };
    Tensor<qint32_t> R{out.shape()};
    R.allocateCustom(expected, 9);
    return ops::Equals(out, R);

}
inline int testConv2DCase47()
{
    qint32_t input[] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 };
    Tensor<qint32_t> A{ 1, 4, 3 };
A.allocateCustom(input, 12);
    Conv2D<qint32_t> layer{ 1, 1, { 2, 1 }, { 1, 1 }, { 1, 0 }, false, PaddingMode::REFLECT };
    ops::Fill(layer.weights, 1);
    
    const auto out = layer.forward(A);
    qint32_t expected[] = { 5, 7, 9, 5, 7, 9, 11, 13, 15, 17, 19, 21, 17, 19, 21 };
    Tensor<qint32_t> R{out.shape()};
    R.allocateCustom(expected, 15);
    return ops::Equals(out, R);

}
inline int testConv2DCase48()
{
    qint32_t input[] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36 };
    Tensor<qint32_t> A{ 3, 3, 4 };
A.allocateCustom(input, 36);
    Conv2D<qint32_t> layer{ 3, 1, { 1, 2 }, { 2, 1 }, { 0, 1 }, true, PaddingMode::CIRCULAR };
    ops::Fill(layer.weights, 1);
    ops::Fill(layer.biases, 1);
    const auto out = layer.forward(A);
    qint32_t expected[] = { 88, 82, 88, 94, 88, 136, 130, 136, 142, 136 };
    Tensor<qint32_t> R{out.shape()};
    R.allocateCustom(expected, 10);
    return ops::Equals(out, R);

}
inline int testConv2DCase49()
{
    qint32_t input[] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 };
    Tensor<qint32_t> A{ 1, 4, 4 };
A.allocateCustom(input, 16);
    Conv2D<qint32_t> layer{ 1, 3, { 1, 1 }, { 1, 2 }, { 2, 2 }, false, PaddingMode::REPLICATE };
    ops::Fill(layer.weights, 1);
    
    const auto out = layer.forward(A);
    qint32_t expected[] = { 1, 1, 3, 4, 1, 1, 3, 4, 1, 1, 3, 4, 5, 5, 7, 8, 9, 9, 11, 12, 13, 13, 15, 16, 13, 13, 15, 16, 13, 13, 15, 16, 1, 1, 3, 4, 1, 1, 3, 4, 1, 1, 3, 4, 5, 5, 7, 8, 9, 9, 11, 12, 13, 13, 15, 16, 13, 13, 15, 16, 13, 13, 15, 16, 1, 1, 3, 4, 1, 1, 3, 4, 1, 1, 3, 4, 5, 5, 7, 8, 9, 9, 11, 12, 13, 13, 15, 16, 13, 13, 15, 16, 13, 13, 15, 16 };
    Tensor<qint32_t> R{out.shape()};
    R.allocateCustom(expected, 96);
    return ops::Equals(out, R);

}
inline int testConv2DCase50()
{
    qint32_t input[] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 };
    Tensor<qint32_t> A{ 1, 4, 4 };
A.allocateCustom(input, 16);
    Conv2D<qint32_t> layer{ 1, 3, { 1, 1 }, { 1, 2 }, { 0, 0 }, true, PaddingMode::ZEROS };
    ops::Fill(layer.weights, 1);
    ops::Fill(layer.biases, 1);
    const auto out = layer.forward(A);
    qint32_t expected[] = { 2, 4, 6, 8, 10, 12, 14, 16, 2, 4, 6, 8, 10, 12, 14, 16, 2, 4, 6, 8, 10, 12, 14, 16 };
    Tensor<qint32_t> R{out.shape()};
    R.allocateCustom(expected, 24);
    return ops::Equals(out, R);

}
inline int testConv2DCase51()
{
    qint32_t input[] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36 };
    Tensor<qint32_t> A{ 3, 3, 4 };
A.allocateCustom(input, 36);
    Conv2D<qint32_t> layer{ 3, 1, { 1, 2 }, { 2, 1 }, { 1, 0 }, false, PaddingMode::REFLECT };
    ops::Fill(layer.weights, 1);
    
    const auto out = layer.forward(A);
    qint32_t expected[] = { 105, 111, 117, 105, 111, 117, 105, 111, 117 };
    Tensor<qint32_t> R{out.shape()};
    R.allocateCustom(expected, 9);
    return ops::Equals(out, R);

}
inline int testConv2DCase52()
{
    qint32_t input[] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 };
    Tensor<qint32_t> A{ 1, 4, 3 };
A.allocateCustom(input, 12);
    Conv2D<qint32_t> layer{ 1, 1, { 2, 1 }, { 1, 1 }, { 0, 1 }, true, PaddingMode::CIRCULAR };
    ops::Fill(layer.weights, 1);
    ops::Fill(layer.biases, 1);
    const auto out = layer.forward(A);
    qint32_t expected[] = { 10, 6, 8, 10, 6, 16, 12, 14, 16, 12, 22, 18, 20, 22, 18 };
    Tensor<qint32_t> R{out.shape()};
    R.allocateCustom(expected, 15);
    return ops::Equals(out, R);

}
inline int testConv2DCase53()
{
    qint32_t input[] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 };
    Tensor<qint32_t> A{ 1, 4, 3 };
A.allocateCustom(input, 12);
    Conv2D<qint32_t> layer{ 1, 1, { 2, 1 }, { 1, 1 }, { 2, 2 }, false, PaddingMode::REPLICATE };
    ops::Fill(layer.weights, 1);
    
    const auto out = layer.forward(A);
    qint32_t expected[] = { 2, 2, 2, 4, 6, 6, 6, 2, 2, 2, 4, 6, 6, 6, 5, 5, 5, 7, 9, 9, 9, 11, 11, 11, 13, 15, 15, 15, 17, 17, 17, 19, 21, 21, 21, 20, 20, 20, 22, 24, 24, 24, 20, 20, 20, 22, 24, 24, 24 };
    Tensor<qint32_t> R{out.shape()};
    R.allocateCustom(expected, 49);
    return ops::Equals(out, R);

}
inline int testConv2DCase54()
{
    qint32_t input[] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36 };
    Tensor<qint32_t> A{ 3, 3, 4 };
A.allocateCustom(input, 36);
    Conv2D<qint32_t> layer{ 3, 1, { 1, 2 }, { 2, 1 }, { 0, 0 }, true, PaddingMode::ZEROS };
    ops::Fill(layer.weights, 1);
    ops::Fill(layer.biases, 1);
    const auto out = layer.forward(A);
    qint32_t expected[] = { 82, 88, 94, 130, 136, 142 };
    Tensor<qint32_t> R{out.shape()};
    R.allocateCustom(expected, 6);
    return ops::Equals(out, R);

}
inline int testConv2DCase55()
{
    qint32_t input[] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 };
    Tensor<qint32_t> A{ 1, 4, 4 };
A.allocateCustom(input, 16);
    Conv2D<qint32_t> layer{ 1, 3, { 1, 1 }, { 1, 2 }, { 1, 0 }, false, PaddingMode::REFLECT };
    ops::Fill(layer.weights, 1);
    
    const auto out = layer.forward(A);
    qint32_t expected[] = { 5, 7, 1, 3, 5, 7, 9, 11, 13, 15, 9, 11, 5, 7, 1, 3, 5, 7, 9, 11, 13, 15, 9, 11, 5, 7, 1, 3, 5, 7, 9, 11, 13, 15, 9, 11 };
    Tensor<qint32_t> R{out.shape()};
    R.allocateCustom(expected, 36);
    return ops::Equals(out, R);

}
inline int testConv2DCase56()
{
    qint32_t input[] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 };
    Tensor<qint32_t> A{ 1, 4, 4 };
A.allocateCustom(input, 16);
    Conv2D<qint32_t> layer{ 1, 3, { 1, 1 }, { 1, 2 }, { 0, 1 }, true, PaddingMode::CIRCULAR };
    ops::Fill(layer.weights, 1);
    ops::Fill(layer.biases, 1);
    const auto out = layer.forward(A);
    qint32_t expected[] = { 5, 3, 5, 9, 7, 9, 13, 11, 13, 17, 15, 17, 5, 3, 5, 9, 7, 9, 13, 11, 13, 17, 15, 17, 5, 3, 5, 9, 7, 9, 13, 11, 13, 17, 15, 17 };
    Tensor<qint32_t> R{out.shape()};
    R.allocateCustom(expected, 36);
    return ops::Equals(out, R);

}
inline int testConv2DCase57()
{
    qint32_t input[] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36 };
    Tensor<qint32_t> A{ 3, 3, 4 };
A.allocateCustom(input, 36);
    Conv2D<qint32_t> layer{ 3, 1, { 1, 2 }, { 2, 1 }, { 2, 2 }, false, PaddingMode::REPLICATE };
    ops::Fill(layer.weights, 1);
    
    const auto out = layer.forward(A);
    qint32_t expected[] = { 78, 78, 81, 87, 93, 96, 96, 78, 78, 81, 87, 93, 96, 96, 126, 126, 129, 135, 141, 144, 144, 126, 126, 129, 135, 141, 144, 144 };
    Tensor<qint32_t> R{out.shape()};
    R.allocateCustom(expected, 28);
    return ops::Equals(out, R);

}
inline int testConv2DCase58()
{
    qint32_t input[] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 };
    Tensor<qint32_t> A{ 1, 4, 3 };
A.allocateCustom(input, 12);
    Conv2D<qint32_t> layer{ 1, 1, { 2, 1 }, { 1, 1 }, { 0, 0 }, true, PaddingMode::ZEROS };
    ops::Fill(layer.weights, 1);
    ops::Fill(layer.biases, 1);
    const auto out = layer.forward(A);
    qint32_t expected[] = { 6, 8, 10, 12, 14, 16, 18, 20, 22 };
    Tensor<qint32_t> R{out.shape()};
    R.allocateCustom(expected, 9);
    return ops::Equals(out, R);

}
inline int testConv2DCase59()
{
    qint32_t input[] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 };
    Tensor<qint32_t> A{ 1, 4, 3 };
A.allocateCustom(input, 12);
    Conv2D<qint32_t> layer{ 1, 1, { 2, 1 }, { 1, 1 }, { 1, 0 }, false, PaddingMode::REFLECT };
    ops::Fill(layer.weights, 1);
    
    const auto out = layer.forward(A);
    qint32_t expected[] = { 5, 7, 9, 5, 7, 9, 11, 13, 15, 17, 19, 21, 17, 19, 21 };
    Tensor<qint32_t> R{out.shape()};
    R.allocateCustom(expected, 15);
    return ops::Equals(out, R);

}
inline int testConv2DCase60()
{
    qint32_t input[] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36 };
    Tensor<qint32_t> A{ 3, 3, 4 };
A.allocateCustom(input, 36);
    Conv2D<qint32_t> layer{ 3, 1, { 1, 2 }, { 2, 1 }, { 0, 1 }, true, PaddingMode::CIRCULAR };
    ops::Fill(layer.weights, 1);
    ops::Fill(layer.biases, 1);
    const auto out = layer.forward(A);
    qint32_t expected[] = { 88, 82, 88, 94, 88, 136, 130, 136, 142, 136 };
    Tensor<qint32_t> R{out.shape()};
    R.allocateCustom(expected, 10);
    return ops::Equals(out, R);

}
inline int testConv2DCase61()
{
    qint32_t input[] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 };
    Tensor<qint32_t> A{ 1, 4, 4 };
A.allocateCustom(input, 16);
    Conv2D<qint32_t> layer{ 1, 3, { 1, 1 }, { 1, 2 }, { 2, 2 }, false, PaddingMode::REPLICATE };
    ops::Fill(layer.weights, 1);
    
    const auto out = layer.forward(A);
    qint32_t expected[] = { 1, 1, 3, 4, 1, 1, 3, 4, 1, 1, 3, 4, 5, 5, 7, 8, 9, 9, 11, 12, 13, 13, 15, 16, 13, 13, 15, 16, 13, 13, 15, 16, 1, 1, 3, 4, 1, 1, 3, 4, 1, 1, 3, 4, 5, 5, 7, 8, 9, 9, 11, 12, 13, 13, 15, 16, 13, 13, 15, 16, 13, 13, 15, 16, 1, 1, 3, 4, 1, 1, 3, 4, 1, 1, 3, 4, 5, 5, 7, 8, 9, 9, 11, 12, 13, 13, 15, 16, 13, 13, 15, 16, 13, 13, 15, 16 };
    Tensor<qint32_t> R{out.shape()};
    R.allocateCustom(expected, 96);
    return ops::Equals(out, R);

}
inline int testConv2DCase62()
{
    qint32_t input[] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 };
    Tensor<qint32_t> A{ 1, 4, 4 };
A.allocateCustom(input, 16);
    Conv2D<qint32_t> layer{ 1, 3, { 1, 1 }, { 1, 2 }, { 0, 0 }, true, PaddingMode::ZEROS };
    ops::Fill(layer.weights, 1);
    ops::Fill(layer.biases, 1);
    const auto out = layer.forward(A);
    qint32_t expected[] = { 2, 4, 6, 8, 10, 12, 14, 16, 2, 4, 6, 8, 10, 12, 14, 16, 2, 4, 6, 8, 10, 12, 14, 16 };
    Tensor<qint32_t> R{out.shape()};
    R.allocateCustom(expected, 24);
    return ops::Equals(out, R);

}
inline int testConv2DCase63()
{
    qint32_t input[] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36 };
    Tensor<qint32_t> A{ 3, 3, 4 };
A.allocateCustom(input, 36);
    Conv2D<qint32_t> layer{ 3, 1, { 1, 2 }, { 2, 1 }, { 1, 0 }, false, PaddingMode::REFLECT };
    ops::Fill(layer.weights, 1);
    
    const auto out = layer.forward(A);
    qint32_t expected[] = { 105, 111, 117, 105, 111, 117, 105, 111, 117 };
    Tensor<qint32_t> R{out.shape()};
    R.allocateCustom(expected, 9);
    return ops::Equals(out, R);

}
inline int testConv2DCase64()
{
    qint32_t input[] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 };
    Tensor<qint32_t> A{ 1, 4, 3 };
A.allocateCustom(input, 12);
    Conv2D<qint32_t> layer{ 1, 1, { 2, 1 }, { 1, 1 }, { 0, 1 }, true, PaddingMode::CIRCULAR };
    ops::Fill(layer.weights, 1);
    ops::Fill(layer.biases, 1);
    const auto out = layer.forward(A);
    qint32_t expected[] = { 10, 6, 8, 10, 6, 16, 12, 14, 16, 12, 22, 18, 20, 22, 18 };
    Tensor<qint32_t> R{out.shape()};
    R.allocateCustom(expected, 15);
    return ops::Equals(out, R);

}
inline int testConv2DCase65()
{
    qint32_t input[] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 };
    Tensor<qint32_t> A{ 1, 4, 3 };
A.allocateCustom(input, 12);
    Conv2D<qint32_t> layer{ 1, 1, { 2, 1 }, { 1, 1 }, { 2, 2 }, false, PaddingMode::REPLICATE };
    ops::Fill(layer.weights, 1);
    
    const auto out = layer.forward(A);
    qint32_t expected[] = { 2, 2, 2, 4, 6, 6, 6, 2, 2, 2, 4, 6, 6, 6, 5, 5, 5, 7, 9, 9, 9, 11, 11, 11, 13, 15, 15, 15, 17, 17, 17, 19, 21, 21, 21, 20, 20, 20, 22, 24, 24, 24, 20, 20, 20, 22, 24, 24, 24 };
    Tensor<qint32_t> R{out.shape()};
    R.allocateCustom(expected, 49);
    return ops::Equals(out, R);

}
inline int testConv2DCase66()
{
    qint32_t input[] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36 };
    Tensor<qint32_t> A{ 3, 3, 4 };
A.allocateCustom(input, 36);
    Conv2D<qint32_t> layer{ 3, 1, { 1, 2 }, { 2, 1 }, { 0, 0 }, true, PaddingMode::ZEROS };
    ops::Fill(layer.weights, 1);
    ops::Fill(layer.biases, 1);
    const auto out = layer.forward(A);
    qint32_t expected[] = { 82, 88, 94, 130, 136, 142 };
    Tensor<qint32_t> R{out.shape()};
    R.allocateCustom(expected, 6);
    return ops::Equals(out, R);

}
inline int testConv2DCase67()
{
    qint32_t input[] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 };
    Tensor<qint32_t> A{ 1, 4, 4 };
A.allocateCustom(input, 16);
    Conv2D<qint32_t> layer{ 1, 3, { 1, 1 }, { 1, 2 }, { 1, 0 }, false, PaddingMode::REFLECT };
    ops::Fill(layer.weights, 1);
    
    const auto out = layer.forward(A);
    qint32_t expected[] = { 5, 7, 1, 3, 5, 7, 9, 11, 13, 15, 9, 11, 5, 7, 1, 3, 5, 7, 9, 11, 13, 15, 9, 11, 5, 7, 1, 3, 5, 7, 9, 11, 13, 15, 9, 11 };
    Tensor<qint32_t> R{out.shape()};
    R.allocateCustom(expected, 36);
    return ops::Equals(out, R);

}
inline int testConv2DCase68()
{
    qint32_t input[] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 };
    Tensor<qint32_t> A{ 1, 4, 4 };
A.allocateCustom(input, 16);
    Conv2D<qint32_t> layer{ 1, 3, { 1, 1 }, { 1, 2 }, { 0, 1 }, true, PaddingMode::CIRCULAR };
    ops::Fill(layer.weights, 1);
    ops::Fill(layer.biases, 1);
    const auto out = layer.forward(A);
    qint32_t expected[] = { 5, 3, 5, 9, 7, 9, 13, 11, 13, 17, 15, 17, 5, 3, 5, 9, 7, 9, 13, 11, 13, 17, 15, 17, 5, 3, 5, 9, 7, 9, 13, 11, 13, 17, 15, 17 };
    Tensor<qint32_t> R{out.shape()};
    R.allocateCustom(expected, 36);
    return ops::Equals(out, R);

}
inline int testConv2DCase69()
{
    qint32_t input[] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36 };
    Tensor<qint32_t> A{ 3, 3, 4 };
A.allocateCustom(input, 36);
    Conv2D<qint32_t> layer{ 3, 1, { 1, 2 }, { 2, 1 }, { 2, 2 }, false, PaddingMode::REPLICATE };
    ops::Fill(layer.weights, 1);
    
    const auto out = layer.forward(A);
    qint32_t expected[] = { 78, 78, 81, 87, 93, 96, 96, 78, 78, 81, 87, 93, 96, 96, 126, 126, 129, 135, 141, 144, 144, 126, 126, 129, 135, 141, 144, 144 };
    Tensor<qint32_t> R{out.shape()};
    R.allocateCustom(expected, 28);
    return ops::Equals(out, R);

}
inline int testConv2DCase70()
{
    qint32_t input[] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 };
    Tensor<qint32_t> A{ 1, 4, 3 };
A.allocateCustom(input, 12);
    Conv2D<qint32_t> layer{ 1, 1, { 2, 1 }, { 1, 1 }, { 0, 0 }, true, PaddingMode::ZEROS };
    ops::Fill(layer.weights, 1);
    ops::Fill(layer.biases, 1);
    const auto out = layer.forward(A);
    qint32_t expected[] = { 6, 8, 10, 12, 14, 16, 18, 20, 22 };
    Tensor<qint32_t> R{out.shape()};
    R.allocateCustom(expected, 9);
    return ops::Equals(out, R);

}
inline int testConv2DCase71()
{
    qint32_t input[] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 };
    Tensor<qint32_t> A{ 1, 4, 3 };
A.allocateCustom(input, 12);
    Conv2D<qint32_t> layer{ 1, 1, { 2, 1 }, { 1, 1 }, { 1, 0 }, false, PaddingMode::REFLECT };
    ops::Fill(layer.weights, 1);
    
    const auto out = layer.forward(A);
    qint32_t expected[] = { 5, 7, 9, 5, 7, 9, 11, 13, 15, 17, 19, 21, 17, 19, 21 };
    Tensor<qint32_t> R{out.shape()};
    R.allocateCustom(expected, 15);
    return ops::Equals(out, R);

}
inline int testConv2DCase72()
{
    qint32_t input[] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36 };
    Tensor<qint32_t> A{ 3, 3, 4 };
A.allocateCustom(input, 36);
    Conv2D<qint32_t> layer{ 3, 1, { 1, 2 }, { 2, 1 }, { 0, 1 }, true, PaddingMode::CIRCULAR };
    ops::Fill(layer.weights, 1);
    ops::Fill(layer.biases, 1);
    const auto out = layer.forward(A);
    qint32_t expected[] = { 88, 82, 88, 94, 88, 136, 130, 136, 142, 136 };
    Tensor<qint32_t> R{out.shape()};
    R.allocateCustom(expected, 10);
    return ops::Equals(out, R);

}
inline int testConv2DCase73()
{
    qint32_t input[] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 };
    Tensor<qint32_t> A{ 1, 4, 4 };
A.allocateCustom(input, 16);
    Conv2D<qint32_t> layer{ 1, 3, { 1, 1 }, { 1, 2 }, { 2, 2 }, false, PaddingMode::REPLICATE };
    ops::Fill(layer.weights, 1);
    
    const auto out = layer.forward(A);
    qint32_t expected[] = { 1, 1, 3, 4, 1, 1, 3, 4, 1, 1, 3, 4, 5, 5, 7, 8, 9, 9, 11, 12, 13, 13, 15, 16, 13, 13, 15, 16, 13, 13, 15, 16, 1, 1, 3, 4, 1, 1, 3, 4, 1, 1, 3, 4, 5, 5, 7, 8, 9, 9, 11, 12, 13, 13, 15, 16, 13, 13, 15, 16, 13, 13, 15, 16, 1, 1, 3, 4, 1, 1, 3, 4, 1, 1, 3, 4, 5, 5, 7, 8, 9, 9, 11, 12, 13, 13, 15, 16, 13, 13, 15, 16, 13, 13, 15, 16 };
    Tensor<qint32_t> R{out.shape()};
    R.allocateCustom(expected, 96);
    return ops::Equals(out, R);

}
inline int testConv2DCase74()
{
    qint32_t input[] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 };
    Tensor<qint32_t> A{ 1, 4, 4 };
A.allocateCustom(input, 16);
    Conv2D<qint32_t> layer{ 1, 3, { 1, 1 }, { 1, 2 }, { 0, 0 }, true, PaddingMode::ZEROS };
    ops::Fill(layer.weights, 1);
    ops::Fill(layer.biases, 1);
    const auto out = layer.forward(A);
    qint32_t expected[] = { 2, 4, 6, 8, 10, 12, 14, 16, 2, 4, 6, 8, 10, 12, 14, 16, 2, 4, 6, 8, 10, 12, 14, 16 };
    Tensor<qint32_t> R{out.shape()};
    R.allocateCustom(expected, 24);
    return ops::Equals(out, R);

}

inline void TestNNConv2D()
{	EML_RUN_TEST( testConv2DCase0 );
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
#ifndef EML_TINY_TESTBENCH
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
#endif
}

#endif // EML_TEST_NN_LAYERS_CONV2D_H
    