
#ifndef EML_TEST_NN_LAYERS_CONV1D_H
#define EML_TEST_NN_LAYERS_CONV1D_H

inline int testConv1DCase0()
{
    qint32_t A_data[4]{} ;
	Tensor<qint32_t> A{ 1, 1, 4 };
	A.allocateCustom(A_data, 4);
	arange(A,  1);

    Conv1D<qint32_t> layer{ 1, 1, 1, 3, 0, true, PaddingMode::REFLECT };
    fill(layer.weights, 1);
    fill(layer.biases, 1);
    const auto out = layer.forward(A);

    qint32_t R_data[] = { 2, 5 };
	Tensor<qint32_t> R{ 1, 1, 2 };
	R.allocateCustom(R_data, 2);

    EML_ASSERT_TENSOR_EQUALS(R, out);
}
inline int testConv1DCase1()
{
    qint32_t A_data[15]{} ;
	Tensor<qint32_t> A{ 3, 1, 5 };
	A.allocateCustom(A_data, 15);
	arange(A,  1);

    Conv1D<qint32_t> layer{ 3, 1, 1, 1, 0, false, PaddingMode::CIRCULAR };
    fill(layer.weights, 1);
    
    const auto out = layer.forward(A);

    qint32_t R_data[] = { 18, 21, 24, 27, 30 };
	Tensor<qint32_t> R{ 1, 1, 5 };
	R.allocateCustom(R_data, 5);

    EML_ASSERT_TENSOR_EQUALS(R, out);
}
inline int testConv1DCase2()
{
    qint32_t A_data[5]{} ;
	Tensor<qint32_t> A{ 1, 1, 5 };
	A.allocateCustom(A_data, 5);
	arange(A,  1);

    Conv1D<qint32_t> layer{ 1, 3, 2, 3, 0, true, PaddingMode::REPLICATE };
    fill(layer.weights, 1);
    fill(layer.biases, 1);
    const auto out = layer.forward(A);

    qint32_t R_data[] = { 4, 10, 4, 10, 4, 10 };
	Tensor<qint32_t> R{ 3, 1, 2 };
	R.allocateCustom(R_data, 6);

    EML_ASSERT_TENSOR_EQUALS(R, out);
}
inline int testConv1DCase3()
{
    qint32_t A_data[4]{} ;
	Tensor<qint32_t> A{ 1, 1, 4 };
	A.allocateCustom(A_data, 4);
	arange(A,  1);

    Conv1D<qint32_t> layer{ 1, 3, 1, 2, 0, false, PaddingMode::REFLECT };
    fill(layer.weights, 1);
    
    const auto out = layer.forward(A);

    qint32_t R_data[] = { 1, 3, 1, 3, 1, 3 };
	Tensor<qint32_t> R{ 3, 1, 2 };
	R.allocateCustom(R_data, 6);

    EML_ASSERT_TENSOR_EQUALS(R, out);
}
inline int testConv1DCase4()
{
    qint32_t A_data[15]{} ;
	Tensor<qint32_t> A{ 3, 1, 5 };
	A.allocateCustom(A_data, 15);
	arange(A,  1);

    Conv1D<qint32_t> layer{ 3, 1, 3, 2, 0, true, PaddingMode::CIRCULAR };
    fill(layer.weights, 1);
    fill(layer.biases, 1);
    const auto out = layer.forward(A);

    qint32_t R_data[] = { 64, 82 };
	Tensor<qint32_t> R{ 1, 1, 2 };
	R.allocateCustom(R_data, 2);

    EML_ASSERT_TENSOR_EQUALS(R, out);
}
inline int testConv1DCase5()
{
    qint32_t A_data[18]{} ;
	Tensor<qint32_t> A{ 3, 1, 6 };
	A.allocateCustom(A_data, 18);
	arange(A,  1);

    Conv1D<qint32_t> layer{ 3, 1, 2, 3, 0, false, PaddingMode::REFLECT };
    fill(layer.weights, 1);
    
    const auto out = layer.forward(A);

    qint32_t R_data[] = { 45, 63 };
	Tensor<qint32_t> R{ 1, 1, 2 };
	R.allocateCustom(R_data, 2);

    EML_ASSERT_TENSOR_EQUALS(R, out);
}
inline int testConv1DCase6()
{
    qint32_t A_data[5]{} ;
	Tensor<qint32_t> A{ 1, 1, 5 };
	A.allocateCustom(A_data, 5);
	arange(A,  1);

    Conv1D<qint32_t> layer{ 1, 3, 1, 2, 0, true, PaddingMode::CIRCULAR };
    fill(layer.weights, 1);
    fill(layer.biases, 1);
    const auto out = layer.forward(A);

    qint32_t R_data[] = { 2, 4, 6, 2, 4, 6, 2, 4, 6 };
	Tensor<qint32_t> R{ 3, 1, 3 };
	R.allocateCustom(R_data, 9);

    EML_ASSERT_TENSOR_EQUALS(R, out);
}
inline int testConv1DCase7()
{
    qint32_t A_data[6]{} ;
	Tensor<qint32_t> A{ 1, 1, 6 };
	A.allocateCustom(A_data, 6);
	arange(A,  1);

    Conv1D<qint32_t> layer{ 1, 3, 2, 1, 0, false, PaddingMode::REPLICATE };
    fill(layer.weights, 1);
    
    const auto out = layer.forward(A);

    qint32_t R_data[] = { 3, 5, 7, 9, 11, 3, 5, 7, 9, 11, 3, 5, 7, 9, 11 };
	Tensor<qint32_t> R{ 3, 1, 5 };
	R.allocateCustom(R_data, 15);

    EML_ASSERT_TENSOR_EQUALS(R, out);
}
inline int testConv1DCase8()
{
    qint32_t A_data[18]{} ;
	Tensor<qint32_t> A{ 3, 1, 6 };
	A.allocateCustom(A_data, 18);
	arange(A,  1);

    Conv1D<qint32_t> layer{ 3, 1, 3, 1, 0, true, PaddingMode::ZEROS };
    fill(layer.weights, 1);
    fill(layer.biases, 1);
    const auto out = layer.forward(A);

    qint32_t R_data[] = { 73, 82, 91, 100 };
	Tensor<qint32_t> R{ 1, 1, 4 };
	R.allocateCustom(R_data, 4);

    EML_ASSERT_TENSOR_EQUALS(R, out);
}
inline int testConv1DCase9()
{
    qint32_t A_data[4]{} ;
	Tensor<qint32_t> A{ 1, 1, 4 };
	A.allocateCustom(A_data, 4);
	arange(A,  1);

    Conv1D<qint32_t> layer{ 1, 1, 1, 2, 0, false, PaddingMode::CIRCULAR };
    fill(layer.weights, 1);
    
    const auto out = layer.forward(A);

    qint32_t R_data[] = { 1, 3 };
	Tensor<qint32_t> R{ 1, 1, 2 };
	R.allocateCustom(R_data, 2);

    EML_ASSERT_TENSOR_EQUALS(R, out);
}
inline int testConv1DCase10()
{
    qint32_t A_data[12]{} ;
	Tensor<qint32_t> A{ 3, 1, 4 };
	A.allocateCustom(A_data, 12);
	arange(A,  1);

    Conv1D<qint32_t> layer{ 3, 1, 3, 2, 0, true, PaddingMode::REFLECT };
    fill(layer.weights, 1);
    fill(layer.biases, 1);
    const auto out = layer.forward(A);

    qint32_t R_data[] = { 55 };
	Tensor<qint32_t> R{ 1, 1, 1 };
	R.allocateCustom(R_data, 1);

    EML_ASSERT_TENSOR_EQUALS(R, out);
}
inline int testConv1DCase11()
{
    qint32_t A_data[12]{} ;
	Tensor<qint32_t> A{ 3, 1, 4 };
	A.allocateCustom(A_data, 12);
	arange(A,  1);

    Conv1D<qint32_t> layer{ 3, 1, 3, 2, 0, false, PaddingMode::CIRCULAR };
    fill(layer.weights, 1);
    
    const auto out = layer.forward(A);

    qint32_t R_data[] = { 54 };
	Tensor<qint32_t> R{ 1, 1, 1 };
	R.allocateCustom(R_data, 1);

    EML_ASSERT_TENSOR_EQUALS(R, out);
}
inline int testConv1DCase12()
{
    qint32_t A_data[5]{} ;
	Tensor<qint32_t> A{ 1, 1, 5 };
	A.allocateCustom(A_data, 5);
	arange(A,  1);

    Conv1D<qint32_t> layer{ 1, 1, 3, 2, 0, true, PaddingMode::REFLECT };
    fill(layer.weights, 1);
    fill(layer.biases, 1);
    const auto out = layer.forward(A);

    qint32_t R_data[] = { 7, 13 };
	Tensor<qint32_t> R{ 1, 1, 2 };
	R.allocateCustom(R_data, 2);

    EML_ASSERT_TENSOR_EQUALS(R, out);
}
inline int testConv1DCase13()
{
    qint32_t A_data[4]{} ;
	Tensor<qint32_t> A{ 1, 1, 4 };
	A.allocateCustom(A_data, 4);
	arange(A,  1);

    Conv1D<qint32_t> layer{ 1, 3, 3, 2, 0, false, PaddingMode::REPLICATE };
    fill(layer.weights, 1);
    
    const auto out = layer.forward(A);

    qint32_t R_data[] = { 6, 6, 6 };
	Tensor<qint32_t> R{ 3, 1, 1 };
	R.allocateCustom(R_data, 3);

    EML_ASSERT_TENSOR_EQUALS(R, out);
}
inline int testConv1DCase14()
{
    qint32_t A_data[18]{} ;
	Tensor<qint32_t> A{ 3, 1, 6 };
	A.allocateCustom(A_data, 18);
	arange(A,  1);

    Conv1D<qint32_t> layer{ 3, 1, 1, 2, 0, true, PaddingMode::REPLICATE };
    fill(layer.weights, 1);
    fill(layer.biases, 1);
    const auto out = layer.forward(A);

    qint32_t R_data[] = { 22, 28, 34 };
	Tensor<qint32_t> R{ 1, 1, 3 };
	R.allocateCustom(R_data, 3);

    EML_ASSERT_TENSOR_EQUALS(R, out);
}
inline int testConv1DCase15()
{
    qint32_t A_data[6]{} ;
	Tensor<qint32_t> A{ 1, 1, 6 };
	A.allocateCustom(A_data, 6);
	arange(A,  1);

    Conv1D<qint32_t> layer{ 1, 1, 3, 1, 0, false, PaddingMode::ZEROS };
    fill(layer.weights, 1);
    
    const auto out = layer.forward(A);

    qint32_t R_data[] = { 6, 9, 12, 15 };
	Tensor<qint32_t> R{ 1, 1, 4 };
	R.allocateCustom(R_data, 4);

    EML_ASSERT_TENSOR_EQUALS(R, out);
}
inline int testConv1DCase16()
{
    qint32_t A_data[5]{} ;
	Tensor<qint32_t> A{ 1, 1, 5 };
	A.allocateCustom(A_data, 5);
	arange(A,  1);

    Conv1D<qint32_t> layer{ 1, 3, 2, 3, 0, true, PaddingMode::REFLECT };
    fill(layer.weights, 1);
    fill(layer.biases, 1);
    const auto out = layer.forward(A);

    qint32_t R_data[] = { 4, 10, 4, 10, 4, 10 };
	Tensor<qint32_t> R{ 3, 1, 2 };
	R.allocateCustom(R_data, 6);

    EML_ASSERT_TENSOR_EQUALS(R, out);
}
inline int testConv1DCase17()
{
    qint32_t A_data[6]{} ;
	Tensor<qint32_t> A{ 1, 1, 6 };
	A.allocateCustom(A_data, 6);
	arange(A,  1);

    Conv1D<qint32_t> layer{ 1, 1, 2, 3, 0, false, PaddingMode::ZEROS };
    fill(layer.weights, 1);
    
    const auto out = layer.forward(A);

    qint32_t R_data[] = { 3, 9 };
	Tensor<qint32_t> R{ 1, 1, 2 };
	R.allocateCustom(R_data, 2);

    EML_ASSERT_TENSOR_EQUALS(R, out);
}
inline int testConv1DCase18()
{
    qint32_t A_data[5]{} ;
	Tensor<qint32_t> A{ 1, 1, 5 };
	A.allocateCustom(A_data, 5);
	arange(A,  1);

    Conv1D<qint32_t> layer{ 1, 3, 3, 1, 0, true, PaddingMode::ZEROS };
    fill(layer.weights, 1);
    fill(layer.biases, 1);
    const auto out = layer.forward(A);

    qint32_t R_data[] = { 7, 10, 13, 7, 10, 13, 7, 10, 13 };
	Tensor<qint32_t> R{ 3, 1, 3 };
	R.allocateCustom(R_data, 9);

    EML_ASSERT_TENSOR_EQUALS(R, out);
}
inline int testConv1DCase19()
{
    qint32_t A_data[12]{} ;
	Tensor<qint32_t> A{ 3, 1, 4 };
	A.allocateCustom(A_data, 12);
	arange(A,  1);

    Conv1D<qint32_t> layer{ 3, 1, 1, 3, 0, false, PaddingMode::REFLECT };
    fill(layer.weights, 1);
    
    const auto out = layer.forward(A);

    qint32_t R_data[] = { 15, 24 };
	Tensor<qint32_t> R{ 1, 1, 2 };
	R.allocateCustom(R_data, 2);

    EML_ASSERT_TENSOR_EQUALS(R, out);
}
inline int testConv1DCase20()
{
    qint32_t A_data[5]{} ;
	Tensor<qint32_t> A{ 1, 1, 5 };
	A.allocateCustom(A_data, 5);
	arange(A,  1);

    Conv1D<qint32_t> layer{ 1, 1, 2, 1, 0, true, PaddingMode::REPLICATE };
    fill(layer.weights, 1);
    fill(layer.biases, 1);
    const auto out = layer.forward(A);

    qint32_t R_data[] = { 4, 6, 8, 10 };
	Tensor<qint32_t> R{ 1, 1, 4 };
	R.allocateCustom(R_data, 4);

    EML_ASSERT_TENSOR_EQUALS(R, out);
}
inline int testConv1DCase21()
{
    qint32_t A_data[15]{} ;
	Tensor<qint32_t> A{ 3, 1, 5 };
	A.allocateCustom(A_data, 15);
	arange(A,  1);

    Conv1D<qint32_t> layer{ 3, 1, 2, 2, 0, false, PaddingMode::CIRCULAR };
    fill(layer.weights, 1);
    
    const auto out = layer.forward(A);

    qint32_t R_data[] = { 39, 51 };
	Tensor<qint32_t> R{ 1, 1, 2 };
	R.allocateCustom(R_data, 2);

    EML_ASSERT_TENSOR_EQUALS(R, out);
}
inline int testConv1DCase22()
{
    qint32_t A_data[5]{} ;
	Tensor<qint32_t> A{ 1, 1, 5 };
	A.allocateCustom(A_data, 5);
	arange(A,  1);

    Conv1D<qint32_t> layer{ 1, 1, 1, 2, 0, true, PaddingMode::REFLECT };
    fill(layer.weights, 1);
    fill(layer.biases, 1);
    const auto out = layer.forward(A);

    qint32_t R_data[] = { 2, 4, 6 };
	Tensor<qint32_t> R{ 1, 1, 3 };
	R.allocateCustom(R_data, 3);

    EML_ASSERT_TENSOR_EQUALS(R, out);
}
inline int testConv1DCase23()
{
    qint32_t A_data[6]{} ;
	Tensor<qint32_t> A{ 1, 1, 6 };
	A.allocateCustom(A_data, 6);
	arange(A,  1);

    Conv1D<qint32_t> layer{ 1, 1, 3, 2, 0, false, PaddingMode::ZEROS };
    fill(layer.weights, 1);
    
    const auto out = layer.forward(A);

    qint32_t R_data[] = { 6, 12 };
	Tensor<qint32_t> R{ 1, 1, 2 };
	R.allocateCustom(R_data, 2);

    EML_ASSERT_TENSOR_EQUALS(R, out);
}
inline int testConv1DCase24()
{
    qint32_t A_data[6]{} ;
	Tensor<qint32_t> A{ 1, 1, 6 };
	A.allocateCustom(A_data, 6);
	arange(A,  1);

    Conv1D<qint32_t> layer{ 1, 3, 2, 2, 0, true, PaddingMode::CIRCULAR };
    fill(layer.weights, 1);
    fill(layer.biases, 1);
    const auto out = layer.forward(A);

    qint32_t R_data[] = { 4, 8, 12, 4, 8, 12, 4, 8, 12 };
	Tensor<qint32_t> R{ 3, 1, 3 };
	R.allocateCustom(R_data, 9);

    EML_ASSERT_TENSOR_EQUALS(R, out);
}
inline int testConv1DCase25()
{
    qint32_t A_data[18]{} ;
	Tensor<qint32_t> A{ 3, 1, 6 };
	A.allocateCustom(A_data, 18);
	arange(A,  1);

    Conv1D<qint32_t> layer{ 3, 1, 1, 3, 0, false, PaddingMode::CIRCULAR };
    fill(layer.weights, 1);
    
    const auto out = layer.forward(A);

    qint32_t R_data[] = { 21, 30 };
	Tensor<qint32_t> R{ 1, 1, 2 };
	R.allocateCustom(R_data, 2);

    EML_ASSERT_TENSOR_EQUALS(R, out);
}
inline int testConv1DCase26()
{
    qint32_t A_data[15]{} ;
	Tensor<qint32_t> A{ 3, 1, 5 };
	A.allocateCustom(A_data, 15);
	arange(A,  1);

    Conv1D<qint32_t> layer{ 3, 1, 1, 2, 0, true, PaddingMode::ZEROS };
    fill(layer.weights, 1);
    fill(layer.biases, 1);
    const auto out = layer.forward(A);

    qint32_t R_data[] = { 19, 25, 31 };
	Tensor<qint32_t> R{ 1, 1, 3 };
	R.allocateCustom(R_data, 3);

    EML_ASSERT_TENSOR_EQUALS(R, out);
}
inline int testConv1DCase27()
{
    qint32_t A_data[12]{} ;
	Tensor<qint32_t> A{ 3, 1, 4 };
	A.allocateCustom(A_data, 12);
	arange(A,  1);

    Conv1D<qint32_t> layer{ 3, 1, 1, 2, 0, false, PaddingMode::CIRCULAR };
    fill(layer.weights, 1);
    
    const auto out = layer.forward(A);

    qint32_t R_data[] = { 15, 21 };
	Tensor<qint32_t> R{ 1, 1, 2 };
	R.allocateCustom(R_data, 2);

    EML_ASSERT_TENSOR_EQUALS(R, out);
}
inline int testConv1DCase28()
{
    qint32_t A_data[4]{} ;
	Tensor<qint32_t> A{ 1, 1, 4 };
	A.allocateCustom(A_data, 4);
	arange(A,  1);

    Conv1D<qint32_t> layer{ 1, 3, 2, 1, 0, true, PaddingMode::CIRCULAR };
    fill(layer.weights, 1);
    fill(layer.biases, 1);
    const auto out = layer.forward(A);

    qint32_t R_data[] = { 4, 6, 8, 4, 6, 8, 4, 6, 8 };
	Tensor<qint32_t> R{ 3, 1, 3 };
	R.allocateCustom(R_data, 9);

    EML_ASSERT_TENSOR_EQUALS(R, out);
}
inline int testConv1DCase29()
{
    qint32_t A_data[4]{} ;
	Tensor<qint32_t> A{ 1, 1, 4 };
	A.allocateCustom(A_data, 4);
	arange(A,  1);

    Conv1D<qint32_t> layer{ 1, 1, 3, 3, 0, false, PaddingMode::REPLICATE };
    fill(layer.weights, 1);
    
    const auto out = layer.forward(A);

    qint32_t R_data[] = { 6 };
	Tensor<qint32_t> R{ 1, 1, 1 };
	R.allocateCustom(R_data, 1);

    EML_ASSERT_TENSOR_EQUALS(R, out);
}
inline int testConv1DCase30()
{
    qint32_t A_data[5]{} ;
	Tensor<qint32_t> A{ 1, 1, 5 };
	A.allocateCustom(A_data, 5);
	arange(A,  1);

    Conv1D<qint32_t> layer{ 1, 3, 3, 3, 0, true, PaddingMode::ZEROS };
    fill(layer.weights, 1);
    fill(layer.biases, 1);
    const auto out = layer.forward(A);

    qint32_t R_data[] = { 7, 7, 7 };
	Tensor<qint32_t> R{ 3, 1, 1 };
	R.allocateCustom(R_data, 3);

    EML_ASSERT_TENSOR_EQUALS(R, out);
}
inline int testConv1DCase31()
{
    qint32_t A_data[6]{} ;
	Tensor<qint32_t> A{ 1, 1, 6 };
	A.allocateCustom(A_data, 6);
	arange(A,  1);

    Conv1D<qint32_t> layer{ 1, 3, 3, 3, 0, false, PaddingMode::REPLICATE };
    fill(layer.weights, 1);
    
    const auto out = layer.forward(A);

    qint32_t R_data[] = { 6, 15, 6, 15, 6, 15 };
	Tensor<qint32_t> R{ 3, 1, 2 };
	R.allocateCustom(R_data, 6);

    EML_ASSERT_TENSOR_EQUALS(R, out);
}
inline int testConv1DCase32()
{
    qint32_t A_data[15]{} ;
	Tensor<qint32_t> A{ 3, 1, 5 };
	A.allocateCustom(A_data, 15);
	arange(A,  1);

    Conv1D<qint32_t> layer{ 3, 1, 2, 3, 0, true, PaddingMode::REPLICATE };
    fill(layer.weights, 1);
    fill(layer.biases, 1);
    const auto out = layer.forward(A);

    qint32_t R_data[] = { 40, 58 };
	Tensor<qint32_t> R{ 1, 1, 2 };
	R.allocateCustom(R_data, 2);

    EML_ASSERT_TENSOR_EQUALS(R, out);
}
inline int testConv1DCase33()
{
    qint32_t A_data[5]{} ;
	Tensor<qint32_t> A{ 1, 1, 5 };
	A.allocateCustom(A_data, 5);
	arange(A,  1);

    Conv1D<qint32_t> layer{ 1, 1, 2, 3, 0, false, PaddingMode::CIRCULAR };
    fill(layer.weights, 1);
    
    const auto out = layer.forward(A);

    qint32_t R_data[] = { 3, 9 };
	Tensor<qint32_t> R{ 1, 1, 2 };
	R.allocateCustom(R_data, 2);

    EML_ASSERT_TENSOR_EQUALS(R, out);
}
inline int testConv1DCase34()
{
    qint32_t A_data[6]{} ;
	Tensor<qint32_t> A{ 1, 1, 6 };
	A.allocateCustom(A_data, 6);
	arange(A,  1);

    Conv1D<qint32_t> layer{ 1, 3, 1, 1, 0, true, PaddingMode::CIRCULAR };
    fill(layer.weights, 1);
    fill(layer.biases, 1);
    const auto out = layer.forward(A);

    qint32_t R_data[] = { 2, 3, 4, 5, 6, 7, 2, 3, 4, 5, 6, 7, 2, 3, 4, 5, 6, 7 };
	Tensor<qint32_t> R{ 3, 1, 6 };
	R.allocateCustom(R_data, 18);

    EML_ASSERT_TENSOR_EQUALS(R, out);
}
inline int testConv1DCase35()
{
    qint32_t A_data[6]{} ;
	Tensor<qint32_t> A{ 1, 1, 6 };
	A.allocateCustom(A_data, 6);
	arange(A,  1);

    Conv1D<qint32_t> layer{ 1, 1, 1, 1, 0, false, PaddingMode::REFLECT };
    fill(layer.weights, 1);
    
    const auto out = layer.forward(A);

    qint32_t R_data[] = { 1, 2, 3, 4, 5, 6 };
	Tensor<qint32_t> R{ 1, 1, 6 };
	R.allocateCustom(R_data, 6);

    EML_ASSERT_TENSOR_EQUALS(R, out);
}
inline int testConv1DCase36()
{
    qint32_t A_data[15]{} ;
	Tensor<qint32_t> A{ 3, 1, 5 };
	A.allocateCustom(A_data, 15);
	arange(A,  1);

    Conv1D<qint32_t> layer{ 3, 1, 2, 2, 0, true, PaddingMode::ZEROS };
    fill(layer.weights, 1);
    fill(layer.biases, 1);
    const auto out = layer.forward(A);

    qint32_t R_data[] = { 40, 52 };
	Tensor<qint32_t> R{ 1, 1, 2 };
	R.allocateCustom(R_data, 2);

    EML_ASSERT_TENSOR_EQUALS(R, out);
}
inline int testConv1DCase37()
{
    qint32_t A_data[12]{} ;
	Tensor<qint32_t> A{ 3, 1, 4 };
	A.allocateCustom(A_data, 12);
	arange(A,  1);

    Conv1D<qint32_t> layer{ 3, 1, 1, 3, 0, false, PaddingMode::REPLICATE };
    fill(layer.weights, 1);
    
    const auto out = layer.forward(A);

    qint32_t R_data[] = { 15, 24 };
	Tensor<qint32_t> R{ 1, 1, 2 };
	R.allocateCustom(R_data, 2);

    EML_ASSERT_TENSOR_EQUALS(R, out);
}
inline int testConv1DCase38()
{
    qint32_t A_data[18]{} ;
	Tensor<qint32_t> A{ 3, 1, 6 };
	A.allocateCustom(A_data, 18);
	arange(A,  1);

    Conv1D<qint32_t> layer{ 3, 1, 3, 2, 0, true, PaddingMode::REFLECT };
    fill(layer.weights, 1);
    fill(layer.biases, 1);
    const auto out = layer.forward(A);

    qint32_t R_data[] = { 73, 91 };
	Tensor<qint32_t> R{ 1, 1, 2 };
	R.allocateCustom(R_data, 2);

    EML_ASSERT_TENSOR_EQUALS(R, out);
}
inline int testConv1DCase39()
{
    qint32_t A_data[15]{} ;
	Tensor<qint32_t> A{ 3, 1, 5 };
	A.allocateCustom(A_data, 15);
	arange(A,  1);

    Conv1D<qint32_t> layer{ 3, 1, 3, 3, 0, false, PaddingMode::ZEROS };
    fill(layer.weights, 1);
    
    const auto out = layer.forward(A);

    qint32_t R_data[] = { 63 };
	Tensor<qint32_t> R{ 1, 1, 1 };
	R.allocateCustom(R_data, 1);

    EML_ASSERT_TENSOR_EQUALS(R, out);
}
inline int testConv1DCase40()
{
    qint32_t A_data[5]{} ;
	Tensor<qint32_t> A{ 1, 1, 5 };
	A.allocateCustom(A_data, 5);
	arange(A,  1);

    Conv1D<qint32_t> layer{ 1, 3, 1, 2, 0, true, PaddingMode::ZEROS };
    fill(layer.weights, 1);
    fill(layer.biases, 1);
    const auto out = layer.forward(A);

    qint32_t R_data[] = { 2, 4, 6, 2, 4, 6, 2, 4, 6 };
	Tensor<qint32_t> R{ 3, 1, 3 };
	R.allocateCustom(R_data, 9);

    EML_ASSERT_TENSOR_EQUALS(R, out);
}
inline int testConv1DCase41()
{
    qint32_t A_data[5]{} ;
	Tensor<qint32_t> A{ 1, 1, 5 };
	A.allocateCustom(A_data, 5);
	arange(A,  1);

    Conv1D<qint32_t> layer{ 1, 1, 2, 3, 0, false, PaddingMode::REPLICATE };
    fill(layer.weights, 1);
    
    const auto out = layer.forward(A);

    qint32_t R_data[] = { 3, 9 };
	Tensor<qint32_t> R{ 1, 1, 2 };
	R.allocateCustom(R_data, 2);

    EML_ASSERT_TENSOR_EQUALS(R, out);
}
inline int testConv1DCase42()
{
    qint32_t A_data[5]{} ;
	Tensor<qint32_t> A{ 1, 1, 5 };
	A.allocateCustom(A_data, 5);
	arange(A,  1);

    Conv1D<qint32_t> layer{ 1, 3, 1, 3, 0, true, PaddingMode::REPLICATE };
    fill(layer.weights, 1);
    fill(layer.biases, 1);
    const auto out = layer.forward(A);

    qint32_t R_data[] = { 2, 5, 2, 5, 2, 5 };
	Tensor<qint32_t> R{ 3, 1, 2 };
	R.allocateCustom(R_data, 6);

    EML_ASSERT_TENSOR_EQUALS(R, out);
}
inline int testConv1DCase43()
{
    qint32_t A_data[5]{} ;
	Tensor<qint32_t> A{ 1, 1, 5 };
	A.allocateCustom(A_data, 5);
	arange(A,  1);

    Conv1D<qint32_t> layer{ 1, 1, 3, 2, 0, false, PaddingMode::ZEROS };
    fill(layer.weights, 1);
    
    const auto out = layer.forward(A);

    qint32_t R_data[] = { 6, 12 };
	Tensor<qint32_t> R{ 1, 1, 2 };
	R.allocateCustom(R_data, 2);

    EML_ASSERT_TENSOR_EQUALS(R, out);
}
inline int testConv1DCase44()
{
    qint32_t A_data[18]{} ;
	Tensor<qint32_t> A{ 3, 1, 6 };
	A.allocateCustom(A_data, 18);
	arange(A,  1);

    Conv1D<qint32_t> layer{ 3, 1, 2, 1, 0, true, PaddingMode::CIRCULAR };
    fill(layer.weights, 1);
    fill(layer.biases, 1);
    const auto out = layer.forward(A);

    qint32_t R_data[] = { 46, 52, 58, 64, 70 };
	Tensor<qint32_t> R{ 1, 1, 5 };
	R.allocateCustom(R_data, 5);

    EML_ASSERT_TENSOR_EQUALS(R, out);
}
inline int testConv1DCase45()
{
    qint32_t A_data[6]{} ;
	Tensor<qint32_t> A{ 1, 1, 6 };
	A.allocateCustom(A_data, 6);
	arange(A,  1);

    Conv1D<qint32_t> layer{ 1, 1, 2, 2, 0, false, PaddingMode::CIRCULAR };
    fill(layer.weights, 1);
    
    const auto out = layer.forward(A);

    qint32_t R_data[] = { 3, 7, 11 };
	Tensor<qint32_t> R{ 1, 1, 3 };
	R.allocateCustom(R_data, 3);

    EML_ASSERT_TENSOR_EQUALS(R, out);
}
inline int testConv1DCase46()
{
    qint32_t A_data[15]{} ;
	Tensor<qint32_t> A{ 3, 1, 5 };
	A.allocateCustom(A_data, 15);
	arange(A,  1);

    Conv1D<qint32_t> layer{ 3, 1, 2, 3, 0, true, PaddingMode::REFLECT };
    fill(layer.weights, 1);
    fill(layer.biases, 1);
    const auto out = layer.forward(A);

    qint32_t R_data[] = { 40, 58 };
	Tensor<qint32_t> R{ 1, 1, 2 };
	R.allocateCustom(R_data, 2);

    EML_ASSERT_TENSOR_EQUALS(R, out);
}
inline int testConv1DCase47()
{
    qint32_t A_data[12]{} ;
	Tensor<qint32_t> A{ 3, 1, 4 };
	A.allocateCustom(A_data, 12);
	arange(A,  1);

    Conv1D<qint32_t> layer{ 3, 1, 1, 1, 0, false, PaddingMode::REFLECT };
    fill(layer.weights, 1);
    
    const auto out = layer.forward(A);

    qint32_t R_data[] = { 15, 18, 21, 24 };
	Tensor<qint32_t> R{ 1, 1, 4 };
	R.allocateCustom(R_data, 4);

    EML_ASSERT_TENSOR_EQUALS(R, out);
}
inline int testConv1DCase48()
{
    qint32_t A_data[18]{} ;
	Tensor<qint32_t> A{ 3, 1, 6 };
	A.allocateCustom(A_data, 18);
	arange(A,  1);

    Conv1D<qint32_t> layer{ 3, 1, 2, 2, 0, true, PaddingMode::CIRCULAR };
    fill(layer.weights, 1);
    fill(layer.biases, 1);
    const auto out = layer.forward(A);

    qint32_t R_data[] = { 46, 58, 70 };
	Tensor<qint32_t> R{ 1, 1, 3 };
	R.allocateCustom(R_data, 3);

    EML_ASSERT_TENSOR_EQUALS(R, out);
}
inline int testConv1DCase49()
{
    qint32_t A_data[5]{} ;
	Tensor<qint32_t> A{ 1, 1, 5 };
	A.allocateCustom(A_data, 5);
	arange(A,  1);

    Conv1D<qint32_t> layer{ 1, 1, 1, 1, 0, false, PaddingMode::REPLICATE };
    fill(layer.weights, 1);
    
    const auto out = layer.forward(A);

    qint32_t R_data[] = { 1, 2, 3, 4, 5 };
	Tensor<qint32_t> R{ 1, 1, 5 };
	R.allocateCustom(R_data, 5);

    EML_ASSERT_TENSOR_EQUALS(R, out);
}
inline int testConv1DCase50()
{
    qint32_t A_data[15]{} ;
	Tensor<qint32_t> A{ 3, 1, 5 };
	A.allocateCustom(A_data, 15);
	arange(A,  1);

    Conv1D<qint32_t> layer{ 3, 1, 1, 2, 0, true, PaddingMode::REPLICATE };
    fill(layer.weights, 1);
    fill(layer.biases, 1);
    const auto out = layer.forward(A);

    qint32_t R_data[] = { 19, 25, 31 };
	Tensor<qint32_t> R{ 1, 1, 3 };
	R.allocateCustom(R_data, 3);

    EML_ASSERT_TENSOR_EQUALS(R, out);
}
inline int testConv1DCase51()
{
    qint32_t A_data[5]{} ;
	Tensor<qint32_t> A{ 1, 1, 5 };
	A.allocateCustom(A_data, 5);
	arange(A,  1);

    Conv1D<qint32_t> layer{ 1, 1, 3, 2, 0, false, PaddingMode::REFLECT };
    fill(layer.weights, 1);
    
    const auto out = layer.forward(A);

    qint32_t R_data[] = { 6, 12 };
	Tensor<qint32_t> R{ 1, 1, 2 };
	R.allocateCustom(R_data, 2);

    EML_ASSERT_TENSOR_EQUALS(R, out);
}
inline int testConv1DCase52()
{
    qint32_t A_data[18]{} ;
	Tensor<qint32_t> A{ 3, 1, 6 };
	A.allocateCustom(A_data, 18);
	arange(A,  1);

    Conv1D<qint32_t> layer{ 3, 1, 3, 2, 0, true, PaddingMode::REPLICATE };
    fill(layer.weights, 1);
    fill(layer.biases, 1);
    const auto out = layer.forward(A);

    qint32_t R_data[] = { 73, 91 };
	Tensor<qint32_t> R{ 1, 1, 2 };
	R.allocateCustom(R_data, 2);

    EML_ASSERT_TENSOR_EQUALS(R, out);
}
inline int testConv1DCase53()
{
    qint32_t A_data[12]{} ;
	Tensor<qint32_t> A{ 3, 1, 4 };
	A.allocateCustom(A_data, 12);
	arange(A,  1);

    Conv1D<qint32_t> layer{ 3, 1, 3, 2, 0, false, PaddingMode::CIRCULAR };
    fill(layer.weights, 1);
    
    const auto out = layer.forward(A);

    qint32_t R_data[] = { 54 };
	Tensor<qint32_t> R{ 1, 1, 1 };
	R.allocateCustom(R_data, 1);

    EML_ASSERT_TENSOR_EQUALS(R, out);
}
inline int testConv1DCase54()
{
    qint32_t A_data[4]{} ;
	Tensor<qint32_t> A{ 1, 1, 4 };
	A.allocateCustom(A_data, 4);
	arange(A,  1);

    Conv1D<qint32_t> layer{ 1, 3, 2, 2, 0, true, PaddingMode::REPLICATE };
    fill(layer.weights, 1);
    fill(layer.biases, 1);
    const auto out = layer.forward(A);

    qint32_t R_data[] = { 4, 8, 4, 8, 4, 8 };
	Tensor<qint32_t> R{ 3, 1, 2 };
	R.allocateCustom(R_data, 6);

    EML_ASSERT_TENSOR_EQUALS(R, out);
}
inline int testConv1DCase55()
{
    qint32_t A_data[18]{} ;
	Tensor<qint32_t> A{ 3, 1, 6 };
	A.allocateCustom(A_data, 18);
	arange(A,  1);

    Conv1D<qint32_t> layer{ 3, 1, 2, 2, 0, false, PaddingMode::REPLICATE };
    fill(layer.weights, 1);
    
    const auto out = layer.forward(A);

    qint32_t R_data[] = { 45, 57, 69 };
	Tensor<qint32_t> R{ 1, 1, 3 };
	R.allocateCustom(R_data, 3);

    EML_ASSERT_TENSOR_EQUALS(R, out);
}
inline int testConv1DCase56()
{
    qint32_t A_data[5]{} ;
	Tensor<qint32_t> A{ 1, 1, 5 };
	A.allocateCustom(A_data, 5);
	arange(A,  1);

    Conv1D<qint32_t> layer{ 1, 1, 1, 1, 0, true, PaddingMode::ZEROS };
    fill(layer.weights, 1);
    fill(layer.biases, 1);
    const auto out = layer.forward(A);

    qint32_t R_data[] = { 2, 3, 4, 5, 6 };
	Tensor<qint32_t> R{ 1, 1, 5 };
	R.allocateCustom(R_data, 5);

    EML_ASSERT_TENSOR_EQUALS(R, out);
}
inline int testConv1DCase57()
{
    qint32_t A_data[4]{} ;
	Tensor<qint32_t> A{ 1, 1, 4 };
	A.allocateCustom(A_data, 4);
	arange(A,  1);

    Conv1D<qint32_t> layer{ 1, 3, 1, 1, 0, false, PaddingMode::REFLECT };
    fill(layer.weights, 1);
    
    const auto out = layer.forward(A);

    qint32_t R_data[] = { 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4 };
	Tensor<qint32_t> R{ 3, 1, 4 };
	R.allocateCustom(R_data, 12);

    EML_ASSERT_TENSOR_EQUALS(R, out);
}
inline int testConv1DCase58()
{
    qint32_t A_data[4]{} ;
	Tensor<qint32_t> A{ 1, 1, 4 };
	A.allocateCustom(A_data, 4);
	arange(A,  1);

    Conv1D<qint32_t> layer{ 1, 1, 3, 2, 0, true, PaddingMode::REFLECT };
    fill(layer.weights, 1);
    fill(layer.biases, 1);
    const auto out = layer.forward(A);

    qint32_t R_data[] = { 7 };
	Tensor<qint32_t> R{ 1, 1, 1 };
	R.allocateCustom(R_data, 1);

    EML_ASSERT_TENSOR_EQUALS(R, out);
}
inline int testConv1DCase59()
{
    qint32_t A_data[5]{} ;
	Tensor<qint32_t> A{ 1, 1, 5 };
	A.allocateCustom(A_data, 5);
	arange(A,  1);

    Conv1D<qint32_t> layer{ 1, 3, 3, 3, 0, false, PaddingMode::REFLECT };
    fill(layer.weights, 1);
    
    const auto out = layer.forward(A);

    qint32_t R_data[] = { 6, 6, 6 };
	Tensor<qint32_t> R{ 3, 1, 1 };
	R.allocateCustom(R_data, 3);

    EML_ASSERT_TENSOR_EQUALS(R, out);
}
inline int testConv1DCase60()
{
    qint32_t A_data[4]{} ;
	Tensor<qint32_t> A{ 1, 1, 4 };
	A.allocateCustom(A_data, 4);
	arange(A,  1);

    Conv1D<qint32_t> layer{ 1, 1, 3, 2, 0, true, PaddingMode::REFLECT };
    fill(layer.weights, 1);
    fill(layer.biases, 1);
    const auto out = layer.forward(A);

    qint32_t R_data[] = { 7 };
	Tensor<qint32_t> R{ 1, 1, 1 };
	R.allocateCustom(R_data, 1);

    EML_ASSERT_TENSOR_EQUALS(R, out);
}
inline int testConv1DCase61()
{
    qint32_t A_data[18]{} ;
	Tensor<qint32_t> A{ 3, 1, 6 };
	A.allocateCustom(A_data, 18);
	arange(A,  1);

    Conv1D<qint32_t> layer{ 3, 1, 3, 3, 0, false, PaddingMode::REFLECT };
    fill(layer.weights, 1);
    
    const auto out = layer.forward(A);

    qint32_t R_data[] = { 72, 99 };
	Tensor<qint32_t> R{ 1, 1, 2 };
	R.allocateCustom(R_data, 2);

    EML_ASSERT_TENSOR_EQUALS(R, out);
}
inline int testConv1DCase62()
{
    qint32_t A_data[18]{} ;
	Tensor<qint32_t> A{ 3, 1, 6 };
	A.allocateCustom(A_data, 18);
	arange(A,  1);

    Conv1D<qint32_t> layer{ 3, 1, 1, 2, 0, true, PaddingMode::REPLICATE };
    fill(layer.weights, 1);
    fill(layer.biases, 1);
    const auto out = layer.forward(A);

    qint32_t R_data[] = { 22, 28, 34 };
	Tensor<qint32_t> R{ 1, 1, 3 };
	R.allocateCustom(R_data, 3);

    EML_ASSERT_TENSOR_EQUALS(R, out);
}
inline int testConv1DCase63()
{
    qint32_t A_data[4]{} ;
	Tensor<qint32_t> A{ 1, 1, 4 };
	A.allocateCustom(A_data, 4);
	arange(A,  1);

    Conv1D<qint32_t> layer{ 1, 1, 1, 1, 0, false, PaddingMode::REPLICATE };
    fill(layer.weights, 1);
    
    const auto out = layer.forward(A);

    qint32_t R_data[] = { 1, 2, 3, 4 };
	Tensor<qint32_t> R{ 1, 1, 4 };
	R.allocateCustom(R_data, 4);

    EML_ASSERT_TENSOR_EQUALS(R, out);
}
inline int testConv1DCase64()
{
    qint32_t A_data[4]{} ;
	Tensor<qint32_t> A{ 1, 1, 4 };
	A.allocateCustom(A_data, 4);
	arange(A,  1);

    Conv1D<qint32_t> layer{ 1, 1, 3, 2, 0, true, PaddingMode::REPLICATE };
    fill(layer.weights, 1);
    fill(layer.biases, 1);
    const auto out = layer.forward(A);

    qint32_t R_data[] = { 7 };
	Tensor<qint32_t> R{ 1, 1, 1 };
	R.allocateCustom(R_data, 1);

    EML_ASSERT_TENSOR_EQUALS(R, out);
}
inline int testConv1DCase65()
{
    qint32_t A_data[5]{} ;
	Tensor<qint32_t> A{ 1, 1, 5 };
	A.allocateCustom(A_data, 5);
	arange(A,  1);

    Conv1D<qint32_t> layer{ 1, 3, 2, 2, 0, false, PaddingMode::REPLICATE };
    fill(layer.weights, 1);
    
    const auto out = layer.forward(A);

    qint32_t R_data[] = { 3, 7, 3, 7, 3, 7 };
	Tensor<qint32_t> R{ 3, 1, 2 };
	R.allocateCustom(R_data, 6);

    EML_ASSERT_TENSOR_EQUALS(R, out);
}
inline int testConv1DCase66()
{
    qint32_t A_data[6]{} ;
	Tensor<qint32_t> A{ 1, 1, 6 };
	A.allocateCustom(A_data, 6);
	arange(A,  1);

    Conv1D<qint32_t> layer{ 1, 1, 2, 2, 0, true, PaddingMode::CIRCULAR };
    fill(layer.weights, 1);
    fill(layer.biases, 1);
    const auto out = layer.forward(A);

    qint32_t R_data[] = { 4, 8, 12 };
	Tensor<qint32_t> R{ 1, 1, 3 };
	R.allocateCustom(R_data, 3);

    EML_ASSERT_TENSOR_EQUALS(R, out);
}
inline int testConv1DCase67()
{
    qint32_t A_data[4]{} ;
	Tensor<qint32_t> A{ 1, 1, 4 };
	A.allocateCustom(A_data, 4);
	arange(A,  1);

    Conv1D<qint32_t> layer{ 1, 1, 2, 3, 0, false, PaddingMode::REPLICATE };
    fill(layer.weights, 1);
    
    const auto out = layer.forward(A);

    qint32_t R_data[] = { 3 };
	Tensor<qint32_t> R{ 1, 1, 1 };
	R.allocateCustom(R_data, 1);

    EML_ASSERT_TENSOR_EQUALS(R, out);
}
inline int testConv1DCase68()
{
    qint32_t A_data[18]{} ;
	Tensor<qint32_t> A{ 3, 1, 6 };
	A.allocateCustom(A_data, 18);
	arange(A,  1);

    Conv1D<qint32_t> layer{ 3, 1, 1, 2, 0, true, PaddingMode::REFLECT };
    fill(layer.weights, 1);
    fill(layer.biases, 1);
    const auto out = layer.forward(A);

    qint32_t R_data[] = { 22, 28, 34 };
	Tensor<qint32_t> R{ 1, 1, 3 };
	R.allocateCustom(R_data, 3);

    EML_ASSERT_TENSOR_EQUALS(R, out);
}
inline int testConv1DCase69()
{
    qint32_t A_data[5]{} ;
	Tensor<qint32_t> A{ 1, 1, 5 };
	A.allocateCustom(A_data, 5);
	arange(A,  1);

    Conv1D<qint32_t> layer{ 1, 3, 3, 3, 0, false, PaddingMode::REFLECT };
    fill(layer.weights, 1);
    
    const auto out = layer.forward(A);

    qint32_t R_data[] = { 6, 6, 6 };
	Tensor<qint32_t> R{ 3, 1, 1 };
	R.allocateCustom(R_data, 3);

    EML_ASSERT_TENSOR_EQUALS(R, out);
}
inline int testConv1DCase70()
{
    qint32_t A_data[12]{} ;
	Tensor<qint32_t> A{ 3, 1, 4 };
	A.allocateCustom(A_data, 12);
	arange(A,  1);

    Conv1D<qint32_t> layer{ 3, 1, 1, 1, 0, true, PaddingMode::ZEROS };
    fill(layer.weights, 1);
    fill(layer.biases, 1);
    const auto out = layer.forward(A);

    qint32_t R_data[] = { 16, 19, 22, 25 };
	Tensor<qint32_t> R{ 1, 1, 4 };
	R.allocateCustom(R_data, 4);

    EML_ASSERT_TENSOR_EQUALS(R, out);
}
inline int testConv1DCase71()
{
    qint32_t A_data[12]{} ;
	Tensor<qint32_t> A{ 3, 1, 4 };
	A.allocateCustom(A_data, 12);
	arange(A,  1);

    Conv1D<qint32_t> layer{ 3, 1, 3, 3, 0, false, PaddingMode::CIRCULAR };
    fill(layer.weights, 1);
    
    const auto out = layer.forward(A);

    qint32_t R_data[] = { 54 };
	Tensor<qint32_t> R{ 1, 1, 1 };
	R.allocateCustom(R_data, 1);

    EML_ASSERT_TENSOR_EQUALS(R, out);
}
inline int testConv1DCase72()
{
    qint32_t A_data[18]{} ;
	Tensor<qint32_t> A{ 3, 1, 6 };
	A.allocateCustom(A_data, 18);
	arange(A,  1);

    Conv1D<qint32_t> layer{ 3, 1, 1, 1, 0, true, PaddingMode::CIRCULAR };
    fill(layer.weights, 1);
    fill(layer.biases, 1);
    const auto out = layer.forward(A);

    qint32_t R_data[] = { 22, 25, 28, 31, 34, 37 };
	Tensor<qint32_t> R{ 1, 1, 6 };
	R.allocateCustom(R_data, 6);

    EML_ASSERT_TENSOR_EQUALS(R, out);
}
inline int testConv1DCase73()
{
    qint32_t A_data[15]{} ;
	Tensor<qint32_t> A{ 3, 1, 5 };
	A.allocateCustom(A_data, 15);
	arange(A,  1);

    Conv1D<qint32_t> layer{ 3, 1, 3, 3, 0, false, PaddingMode::REFLECT };
    fill(layer.weights, 1);
    
    const auto out = layer.forward(A);

    qint32_t R_data[] = { 63 };
	Tensor<qint32_t> R{ 1, 1, 1 };
	R.allocateCustom(R_data, 1);

    EML_ASSERT_TENSOR_EQUALS(R, out);
}
inline int testConv1DCase74()
{
    qint32_t A_data[4]{} ;
	Tensor<qint32_t> A{ 1, 1, 4 };
	A.allocateCustom(A_data, 4);
	arange(A,  1);

    Conv1D<qint32_t> layer{ 1, 1, 1, 1, 0, true, PaddingMode::REFLECT };
    fill(layer.weights, 1);
    fill(layer.biases, 1);
    const auto out = layer.forward(A);

    qint32_t R_data[] = { 2, 3, 4, 5 };
	Tensor<qint32_t> R{ 1, 1, 4 };
	R.allocateCustom(R_data, 4);

    EML_ASSERT_TENSOR_EQUALS(R, out);
}

inline void TestNNConv1D()
{	EML_RUN_TEST( testConv1DCase0 );
	EML_RUN_TEST( testConv1DCase1 );
	EML_RUN_TEST( testConv1DCase2 );
	EML_RUN_TEST( testConv1DCase3 );
	EML_RUN_TEST( testConv1DCase4 );
	EML_RUN_TEST( testConv1DCase5 );
	EML_RUN_TEST( testConv1DCase6 );
	EML_RUN_TEST( testConv1DCase7 );
	EML_RUN_TEST( testConv1DCase8 );
	EML_RUN_TEST( testConv1DCase9 );
	EML_RUN_TEST( testConv1DCase10 );
	EML_RUN_TEST( testConv1DCase11 );
	EML_RUN_TEST( testConv1DCase12 );
	EML_RUN_TEST( testConv1DCase13 );
	EML_RUN_TEST( testConv1DCase14 );
	EML_RUN_TEST( testConv1DCase15 );
	EML_RUN_TEST( testConv1DCase16 );
	EML_RUN_TEST( testConv1DCase17 );
	EML_RUN_TEST( testConv1DCase18 );
	EML_RUN_TEST( testConv1DCase19 );
	EML_RUN_TEST( testConv1DCase20 );
	EML_RUN_TEST( testConv1DCase21 );
	EML_RUN_TEST( testConv1DCase22 );
	EML_RUN_TEST( testConv1DCase23 );
	EML_RUN_TEST( testConv1DCase24 );
	EML_RUN_TEST( testConv1DCase25 );
#ifndef EML_TINY_TESTBENCH
	EML_RUN_TEST( testConv1DCase26 );
	EML_RUN_TEST( testConv1DCase27 );
	EML_RUN_TEST( testConv1DCase28 );
	EML_RUN_TEST( testConv1DCase29 );
	EML_RUN_TEST( testConv1DCase30 );
	EML_RUN_TEST( testConv1DCase31 );
	EML_RUN_TEST( testConv1DCase32 );
	EML_RUN_TEST( testConv1DCase33 );
	EML_RUN_TEST( testConv1DCase34 );
	EML_RUN_TEST( testConv1DCase35 );
	EML_RUN_TEST( testConv1DCase36 );
	EML_RUN_TEST( testConv1DCase37 );
	EML_RUN_TEST( testConv1DCase38 );
	EML_RUN_TEST( testConv1DCase39 );
	EML_RUN_TEST( testConv1DCase40 );
	EML_RUN_TEST( testConv1DCase41 );
	EML_RUN_TEST( testConv1DCase42 );
	EML_RUN_TEST( testConv1DCase43 );
	EML_RUN_TEST( testConv1DCase44 );
	EML_RUN_TEST( testConv1DCase45 );
	EML_RUN_TEST( testConv1DCase46 );
	EML_RUN_TEST( testConv1DCase47 );
	EML_RUN_TEST( testConv1DCase48 );
	EML_RUN_TEST( testConv1DCase49 );
	EML_RUN_TEST( testConv1DCase50 );
	EML_RUN_TEST( testConv1DCase51 );
	EML_RUN_TEST( testConv1DCase52 );
	EML_RUN_TEST( testConv1DCase53 );
	EML_RUN_TEST( testConv1DCase54 );
	EML_RUN_TEST( testConv1DCase55 );
	EML_RUN_TEST( testConv1DCase56 );
	EML_RUN_TEST( testConv1DCase57 );
	EML_RUN_TEST( testConv1DCase58 );
	EML_RUN_TEST( testConv1DCase59 );
	EML_RUN_TEST( testConv1DCase60 );
	EML_RUN_TEST( testConv1DCase61 );
	EML_RUN_TEST( testConv1DCase62 );
	EML_RUN_TEST( testConv1DCase63 );
	EML_RUN_TEST( testConv1DCase64 );
	EML_RUN_TEST( testConv1DCase65 );
	EML_RUN_TEST( testConv1DCase66 );
	EML_RUN_TEST( testConv1DCase67 );
	EML_RUN_TEST( testConv1DCase68 );
	EML_RUN_TEST( testConv1DCase69 );
	EML_RUN_TEST( testConv1DCase70 );
	EML_RUN_TEST( testConv1DCase71 );
	EML_RUN_TEST( testConv1DCase72 );
	EML_RUN_TEST( testConv1DCase73 );
	EML_RUN_TEST( testConv1DCase74 );
#endif
}

#endif // EML_TEST_NN_LAYERS_CONV1D_H
    