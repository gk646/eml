
#ifndef EML_TEST_NN_LAYERS_LINEAR_H
#define EML_TEST_NN_LAYERS_LINEAR_H

inline int testLinearCase0()
{
    qint32_t input[] = { 1, 2, 3, 4 };
    Tensor<qint32_t> A{ 1, 4 };
A.allocateCustom(input, 4);
    Linear<qint32_t> layer{ 4, 2, true };
    ops::Fill(layer.weights, 1);
    ops::Fill(layer.biases, 1);
    const auto out = layer.forward(A);
    qint32_t expected[] = { 11, 11 };
    Tensor<qint32_t> R{out.shape()};
    R.allocateCustom(expected, 2);
    return ops::Equals(out, R);

}
inline int testLinearCase1()
{
    qint32_t input[] = { 1, 2, 3, 4, 5, 6, 7, 8 };
    Tensor<qint32_t> A{ 1, 8 };
A.allocateCustom(input, 8);
    Linear<qint32_t> layer{ 8, 8, false };
    ops::Fill(layer.weights, 1);
    
    const auto out = layer.forward(A);
    qint32_t expected[] = { 36, 36, 36, 36, 36, 36, 36, 36 };
    Tensor<qint32_t> R{out.shape()};
    R.allocateCustom(expected, 8);
    return ops::Equals(out, R);

}
inline int testLinearCase2()
{
    qint32_t input[] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 };
    Tensor<qint32_t> A{ 1, 16 };
A.allocateCustom(input, 16);
    Linear<qint32_t> layer{ 16, 20, true };
    ops::Fill(layer.weights, 1);
    ops::Fill(layer.biases, 1);
    const auto out = layer.forward(A);
    qint32_t expected[] = { 137, 137, 137, 137, 137, 137, 137, 137, 137, 137, 137, 137, 137, 137, 137, 137, 137, 137, 137, 137 };
    Tensor<qint32_t> R{out.shape()};
    R.allocateCustom(expected, 20);
    return ops::Equals(out, R);

}
inline int testLinearCase3()
{
    qint32_t input[] = { 1, 2, 3, 4 };
    Tensor<qint32_t> A{ 1, 4 };
A.allocateCustom(input, 4);
    Linear<qint32_t> layer{ 4, 2, false };
    ops::Fill(layer.weights, 1);
    
    const auto out = layer.forward(A);
    qint32_t expected[] = { 10, 10 };
    Tensor<qint32_t> R{out.shape()};
    R.allocateCustom(expected, 2);
    return ops::Equals(out, R);

}
inline int testLinearCase4()
{
    qint32_t input[] = { 1, 2, 3, 4, 5, 6, 7, 8 };
    Tensor<qint32_t> A{ 1, 8 };
A.allocateCustom(input, 8);
    Linear<qint32_t> layer{ 8, 8, true };
    ops::Fill(layer.weights, 1);
    ops::Fill(layer.biases, 1);
    const auto out = layer.forward(A);
    qint32_t expected[] = { 37, 37, 37, 37, 37, 37, 37, 37 };
    Tensor<qint32_t> R{out.shape()};
    R.allocateCustom(expected, 8);
    return ops::Equals(out, R);

}
inline int testLinearCase5()
{
    qint32_t input[] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 };
    Tensor<qint32_t> A{ 1, 16 };
A.allocateCustom(input, 16);
    Linear<qint32_t> layer{ 16, 20, false };
    ops::Fill(layer.weights, 1);
    
    const auto out = layer.forward(A);
    qint32_t expected[] = { 136, 136, 136, 136, 136, 136, 136, 136, 136, 136, 136, 136, 136, 136, 136, 136, 136, 136, 136, 136 };
    Tensor<qint32_t> R{out.shape()};
    R.allocateCustom(expected, 20);
    return ops::Equals(out, R);

}
inline int testLinearCase6()
{
    qint32_t input[] = { 1, 2, 3, 4 };
    Tensor<qint32_t> A{ 1, 4 };
A.allocateCustom(input, 4);
    Linear<qint32_t> layer{ 4, 2, true };
    ops::Fill(layer.weights, 1);
    ops::Fill(layer.biases, 1);
    const auto out = layer.forward(A);
    qint32_t expected[] = { 11, 11 };
    Tensor<qint32_t> R{out.shape()};
    R.allocateCustom(expected, 2);
    return ops::Equals(out, R);

}

inline void TestNNLinear()
{	EML_RUN_TEST( testLinearCase0 );
	EML_RUN_TEST( testLinearCase1 );
	EML_RUN_TEST( testLinearCase2 );
	EML_RUN_TEST( testLinearCase3 );
	EML_RUN_TEST( testLinearCase4 );
	EML_RUN_TEST( testLinearCase5 );
	EML_RUN_TEST( testLinearCase6 );

}

#endif // EML_TEST_NN_LAYERS_LINEAR_H
    