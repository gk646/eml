
#ifndef EML_TEST_NN_LAYERS_LINEAR_H
#define EML_TEST_NN_LAYERS_LINEAR_H

inline int testLinearCase0()
{
    qint32_t A_data[ 4 ]{};
    Tensor<qint32_t> A{ 1, 4 };
    A.allocateCustom( A_data, 4 );
    arange( A, 1 );

    Linear<qint32_t> layer{ 4, 2, true };
    fill( layer.weights, 1 );
    fill( layer.biases, 1 );
    const auto out = layer.forward( A );

    qint32_t R_data[] = { 11, 11 };
    Tensor<qint32_t> R{ 2 };
    R.allocateCustom( R_data, 2 );

    EML_ASSERT_TENSOR_EQUALS( R, out );
}
inline int testLinearCase1()
{
    qint32_t A_data[ 8 ]{};
    Tensor<qint32_t> A{ 1, 8 };
    A.allocateCustom( A_data, 8 );
    arange( A, 1 );

    Linear<qint32_t> layer{ 8, 20, false };
    fill( layer.weights, 1 );

    const auto out = layer.forward( A );

    qint32_t R_data[] = { 36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36 };
    Tensor<qint32_t> R{ 20 };
    R.allocateCustom( R_data, 20 );

    EML_ASSERT_TENSOR_EQUALS( R, out );
}
inline int testLinearCase2()
{
    qint32_t A_data[ 16 ]{};
    Tensor<qint32_t> A{ 1, 16 };
    A.allocateCustom( A_data, 16 );
    arange( A, 1 );

    Linear<qint32_t> layer{ 16, 2, true };
    fill( layer.weights, 1 );
    fill( layer.biases, 1 );
    const auto out = layer.forward( A );

    qint32_t R_data[] = { 137, 137 };
    Tensor<qint32_t> R{ 2 };
    R.allocateCustom( R_data, 2 );

    EML_ASSERT_TENSOR_EQUALS( R, out );
}
inline int testLinearCase3()
{
    qint32_t A_data[ 8 ]{};
    Tensor<qint32_t> A{ 1, 8 };
    A.allocateCustom( A_data, 8 );
    arange( A, 1 );

    Linear<qint32_t> layer{ 8, 20, false };
    fill( layer.weights, 1 );

    const auto out = layer.forward( A );

    qint32_t R_data[] = { 36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36 };
    Tensor<qint32_t> R{ 20 };
    R.allocateCustom( R_data, 20 );

    EML_ASSERT_TENSOR_EQUALS( R, out );
}
inline int testLinearCase4()
{
    qint32_t A_data[ 4 ]{};
    Tensor<qint32_t> A{ 1, 4 };
    A.allocateCustom( A_data, 4 );
    arange( A, 1 );

    Linear<qint32_t> layer{ 4, 20, true };
    fill( layer.weights, 1 );
    fill( layer.biases, 1 );
    const auto out = layer.forward( A );

    qint32_t R_data[] = { 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11 };
    Tensor<qint32_t> R{ 20 };
    R.allocateCustom( R_data, 20 );

    EML_ASSERT_TENSOR_EQUALS( R, out );
}
inline int testLinearCase5()
{
    qint32_t A_data[ 4 ]{};
    Tensor<qint32_t> A{ 1, 4 };
    A.allocateCustom( A_data, 4 );
    arange( A, 1 );

    Linear<qint32_t> layer{ 4, 2, false };
    fill( layer.weights, 1 );

    const auto out = layer.forward( A );

    qint32_t R_data[] = { 10, 10 };
    Tensor<qint32_t> R{ 2 };
    R.allocateCustom( R_data, 2 );

    EML_ASSERT_TENSOR_EQUALS( R, out );
}
inline int testLinearCase6()
{
    qint32_t A_data[ 8 ]{};
    Tensor<qint32_t> A{ 1, 8 };
    A.allocateCustom( A_data, 8 );
    arange( A, 1 );

    Linear<qint32_t> layer{ 8, 8, true };
    fill( layer.weights, 1 );
    fill( layer.biases, 1 );
    const auto out = layer.forward( A );

    qint32_t R_data[] = { 37, 37, 37, 37, 37, 37, 37, 37 };
    Tensor<qint32_t> R{ 8 };
    R.allocateCustom( R_data, 8 );

    EML_ASSERT_TENSOR_EQUALS( R, out );
}
inline int testLinearCase7()
{
    qint32_t A_data[ 16 ]{};
    Tensor<qint32_t> A{ 1, 16 };
    A.allocateCustom( A_data, 16 );
    arange( A, 1 );

    Linear<qint32_t> layer{ 16, 20, false };
    fill( layer.weights, 1 );

    const auto out = layer.forward( A );

    qint32_t R_data[] = { 136, 136, 136, 136, 136, 136, 136, 136, 136, 136,
                          136, 136, 136, 136, 136, 136, 136, 136, 136, 136 };
    Tensor<qint32_t> R{ 20 };
    R.allocateCustom( R_data, 20 );

    EML_ASSERT_TENSOR_EQUALS( R, out );
}
inline int testLinearCase8()
{
    qint32_t A_data[ 16 ]{};
    Tensor<qint32_t> A{ 1, 16 };
    A.allocateCustom( A_data, 16 );
    arange( A, 1 );

    Linear<qint32_t> layer{ 16, 20, true };
    fill( layer.weights, 1 );
    fill( layer.biases, 1 );
    const auto out = layer.forward( A );

    qint32_t R_data[] = { 137, 137, 137, 137, 137, 137, 137, 137, 137, 137,
                          137, 137, 137, 137, 137, 137, 137, 137, 137, 137 };
    Tensor<qint32_t> R{ 20 };
    R.allocateCustom( R_data, 20 );

    EML_ASSERT_TENSOR_EQUALS( R, out );
}
inline int testLinearCase9()
{
    qint32_t A_data[ 8 ]{};
    Tensor<qint32_t> A{ 1, 8 };
    A.allocateCustom( A_data, 8 );
    arange( A, 1 );

    Linear<qint32_t> layer{ 8, 20, false };
    fill( layer.weights, 1 );

    const auto out = layer.forward( A );

    qint32_t R_data[] = { 36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36 };
    Tensor<qint32_t> R{ 20 };
    R.allocateCustom( R_data, 20 );

    EML_ASSERT_TENSOR_EQUALS( R, out );
}
inline int testLinearCase10()
{
    qint32_t A_data[ 8 ]{};
    Tensor<qint32_t> A{ 1, 8 };
    A.allocateCustom( A_data, 8 );
    arange( A, 1 );

    Linear<qint32_t> layer{ 8, 20, true };
    fill( layer.weights, 1 );
    fill( layer.biases, 1 );
    const auto out = layer.forward( A );

    qint32_t R_data[] = { 37, 37, 37, 37, 37, 37, 37, 37, 37, 37, 37, 37, 37, 37, 37, 37, 37, 37, 37, 37 };
    Tensor<qint32_t> R{ 20 };
    R.allocateCustom( R_data, 20 );

    EML_ASSERT_TENSOR_EQUALS( R, out );
}
inline int testLinearCase11()
{
    qint32_t A_data[ 8 ]{};
    Tensor<qint32_t> A{ 1, 8 };
    A.allocateCustom( A_data, 8 );
    arange( A, 1 );

    Linear<qint32_t> layer{ 8, 2, false };
    fill( layer.weights, 1 );

    const auto out = layer.forward( A );

    qint32_t R_data[] = { 36, 36 };
    Tensor<qint32_t> R{ 2 };
    R.allocateCustom( R_data, 2 );

    EML_ASSERT_TENSOR_EQUALS( R, out );
}
inline int testLinearCase12()
{
    qint32_t A_data[ 16 ]{};
    Tensor<qint32_t> A{ 1, 16 };
    A.allocateCustom( A_data, 16 );
    arange( A, 1 );

    Linear<qint32_t> layer{ 16, 20, true };
    fill( layer.weights, 1 );
    fill( layer.biases, 1 );
    const auto out = layer.forward( A );

    qint32_t R_data[] = { 137, 137, 137, 137, 137, 137, 137, 137, 137, 137,
                          137, 137, 137, 137, 137, 137, 137, 137, 137, 137 };
    Tensor<qint32_t> R{ 20 };
    R.allocateCustom( R_data, 20 );

    EML_ASSERT_TENSOR_EQUALS( R, out );
}
inline int testLinearCase13()
{
    qint32_t A_data[ 16 ]{};
    Tensor<qint32_t> A{ 1, 16 };
    A.allocateCustom( A_data, 16 );
    arange( A, 1 );

    Linear<qint32_t> layer{ 16, 8, false };
    fill( layer.weights, 1 );

    const auto out = layer.forward( A );

    qint32_t R_data[] = { 136, 136, 136, 136, 136, 136, 136, 136 };
    Tensor<qint32_t> R{ 8 };
    R.allocateCustom( R_data, 8 );

    EML_ASSERT_TENSOR_EQUALS( R, out );
}
inline int testLinearCase14()
{
    qint32_t A_data[ 4 ]{};
    Tensor<qint32_t> A{ 1, 4 };
    A.allocateCustom( A_data, 4 );
    arange( A, 1 );

    Linear<qint32_t> layer{ 4, 8, true };
    fill( layer.weights, 1 );
    fill( layer.biases, 1 );
    const auto out = layer.forward( A );

    qint32_t R_data[] = { 11, 11, 11, 11, 11, 11, 11, 11 };
    Tensor<qint32_t> R{ 8 };
    R.allocateCustom( R_data, 8 );

    EML_ASSERT_TENSOR_EQUALS( R, out );
}
inline int testLinearCase15()
{
    qint32_t A_data[ 16 ]{};
    Tensor<qint32_t> A{ 1, 16 };
    A.allocateCustom( A_data, 16 );
    arange( A, 1 );

    Linear<qint32_t> layer{ 16, 8, false };
    fill( layer.weights, 1 );

    const auto out = layer.forward( A );

    qint32_t R_data[] = { 136, 136, 136, 136, 136, 136, 136, 136 };
    Tensor<qint32_t> R{ 8 };
    R.allocateCustom( R_data, 8 );

    EML_ASSERT_TENSOR_EQUALS( R, out );
}
inline int testLinearCase16()
{
    qint32_t A_data[ 4 ]{};
    Tensor<qint32_t> A{ 1, 4 };
    A.allocateCustom( A_data, 4 );
    arange( A, 1 );

    Linear<qint32_t> layer{ 4, 8, true };
    fill( layer.weights, 1 );
    fill( layer.biases, 1 );
    const auto out = layer.forward( A );

    qint32_t R_data[] = { 11, 11, 11, 11, 11, 11, 11, 11 };
    Tensor<qint32_t> R{ 8 };
    R.allocateCustom( R_data, 8 );

    EML_ASSERT_TENSOR_EQUALS( R, out );
}
inline int testLinearCase17()
{
    qint32_t A_data[ 4 ]{};
    Tensor<qint32_t> A{ 1, 4 };
    A.allocateCustom( A_data, 4 );
    arange( A, 1 );

    Linear<qint32_t> layer{ 4, 8, false };
    fill( layer.weights, 1 );

    const auto out = layer.forward( A );

    qint32_t R_data[] = { 10, 10, 10, 10, 10, 10, 10, 10 };
    Tensor<qint32_t> R{ 8 };
    R.allocateCustom( R_data, 8 );

    EML_ASSERT_TENSOR_EQUALS( R, out );
}
inline int testLinearCase18()
{
    qint32_t A_data[ 8 ]{};
    Tensor<qint32_t> A{ 1, 8 };
    A.allocateCustom( A_data, 8 );
    arange( A, 1 );

    Linear<qint32_t> layer{ 8, 8, true };
    fill( layer.weights, 1 );
    fill( layer.biases, 1 );
    const auto out = layer.forward( A );

    qint32_t R_data[] = { 37, 37, 37, 37, 37, 37, 37, 37 };
    Tensor<qint32_t> R{ 8 };
    R.allocateCustom( R_data, 8 );

    EML_ASSERT_TENSOR_EQUALS( R, out );
}
inline int testLinearCase19()
{
    qint32_t A_data[ 16 ]{};
    Tensor<qint32_t> A{ 1, 16 };
    A.allocateCustom( A_data, 16 );
    arange( A, 1 );

    Linear<qint32_t> layer{ 16, 2, false };
    fill( layer.weights, 1 );

    const auto out = layer.forward( A );

    qint32_t R_data[] = { 136, 136 };
    Tensor<qint32_t> R{ 2 };
    R.allocateCustom( R_data, 2 );

    EML_ASSERT_TENSOR_EQUALS( R, out );
}
inline int testLinearCase20()
{
    qint32_t A_data[ 8 ]{};
    Tensor<qint32_t> A{ 1, 8 };
    A.allocateCustom( A_data, 8 );
    arange( A, 1 );

    Linear<qint32_t> layer{ 8, 2, true };
    fill( layer.weights, 1 );
    fill( layer.biases, 1 );
    const auto out = layer.forward( A );

    qint32_t R_data[] = { 37, 37 };
    Tensor<qint32_t> R{ 2 };
    R.allocateCustom( R_data, 2 );

    EML_ASSERT_TENSOR_EQUALS( R, out );
}
inline int testLinearCase21()
{
    qint32_t A_data[ 16 ]{};
    Tensor<qint32_t> A{ 1, 16 };
    A.allocateCustom( A_data, 16 );
    arange( A, 1 );

    Linear<qint32_t> layer{ 16, 8, false };
    fill( layer.weights, 1 );

    const auto out = layer.forward( A );

    qint32_t R_data[] = { 136, 136, 136, 136, 136, 136, 136, 136 };
    Tensor<qint32_t> R{ 8 };
    R.allocateCustom( R_data, 8 );

    EML_ASSERT_TENSOR_EQUALS( R, out );
}
inline int testLinearCase22()
{
    qint32_t A_data[ 8 ]{};
    Tensor<qint32_t> A{ 1, 8 };
    A.allocateCustom( A_data, 8 );
    arange( A, 1 );

    Linear<qint32_t> layer{ 8, 2, true };
    fill( layer.weights, 1 );
    fill( layer.biases, 1 );
    const auto out = layer.forward( A );

    qint32_t R_data[] = { 37, 37 };
    Tensor<qint32_t> R{ 2 };
    R.allocateCustom( R_data, 2 );

    EML_ASSERT_TENSOR_EQUALS( R, out );
}
inline int testLinearCase23()
{
    qint32_t A_data[ 4 ]{};
    Tensor<qint32_t> A{ 1, 4 };
    A.allocateCustom( A_data, 4 );
    arange( A, 1 );

    Linear<qint32_t> layer{ 4, 2, false };
    fill( layer.weights, 1 );

    const auto out = layer.forward( A );

    qint32_t R_data[] = { 10, 10 };
    Tensor<qint32_t> R{ 2 };
    R.allocateCustom( R_data, 2 );

    EML_ASSERT_TENSOR_EQUALS( R, out );
}
inline int testLinearCase24()
{
    qint32_t A_data[ 16 ]{};
    Tensor<qint32_t> A{ 1, 16 };
    A.allocateCustom( A_data, 16 );
    arange( A, 1 );

    Linear<qint32_t> layer{ 16, 8, true };
    fill( layer.weights, 1 );
    fill( layer.biases, 1 );
    const auto out = layer.forward( A );

    qint32_t R_data[] = { 137, 137, 137, 137, 137, 137, 137, 137 };
    Tensor<qint32_t> R{ 8 };
    R.allocateCustom( R_data, 8 );

    EML_ASSERT_TENSOR_EQUALS( R, out );
}
inline int testLinearCase25()
{
    qint32_t A_data[ 4 ]{};
    Tensor<qint32_t> A{ 1, 4 };
    A.allocateCustom( A_data, 4 );
    arange( A, 1 );

    Linear<qint32_t> layer{ 4, 20, false };
    fill( layer.weights, 1 );

    const auto out = layer.forward( A );

    qint32_t R_data[] = { 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10 };
    Tensor<qint32_t> R{ 20 };
    R.allocateCustom( R_data, 20 );

    EML_ASSERT_TENSOR_EQUALS( R, out );
}

inline void TestNNLinear()
{
    EML_RUN_TEST( testLinearCase0 );
    EML_RUN_TEST( testLinearCase1 );
    EML_RUN_TEST( testLinearCase2 );
    EML_RUN_TEST( testLinearCase3 );
    EML_RUN_TEST( testLinearCase4 );
    EML_RUN_TEST( testLinearCase5 );
    EML_RUN_TEST( testLinearCase6 );
    EML_RUN_TEST( testLinearCase7 );
    EML_RUN_TEST( testLinearCase8 );
    EML_RUN_TEST( testLinearCase9 );
    EML_RUN_TEST( testLinearCase10 );
    EML_RUN_TEST( testLinearCase11 );
    EML_RUN_TEST( testLinearCase12 );
    EML_RUN_TEST( testLinearCase13 );
    EML_RUN_TEST( testLinearCase14 );
    EML_RUN_TEST( testLinearCase15 );
    EML_RUN_TEST( testLinearCase16 );
    EML_RUN_TEST( testLinearCase17 );
    EML_RUN_TEST( testLinearCase18 );
    EML_RUN_TEST( testLinearCase19 );
    EML_RUN_TEST( testLinearCase20 );
    EML_RUN_TEST( testLinearCase21 );
    EML_RUN_TEST( testLinearCase22 );
    EML_RUN_TEST( testLinearCase23 );
    EML_RUN_TEST( testLinearCase24 );
    EML_RUN_TEST( testLinearCase25 );
#ifndef EML_TINY_TESTBENCH
#endif
}

#endif // EML_TEST_NN_LAYERS_LINEAR_H
