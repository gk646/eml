
#ifndef EML_TEST_NN_LAYERS_CONV2D_H
#define EML_TEST_NN_LAYERS_CONV2D_H

inline int testConv2DCase0()
{
    qint32_t A_data[12]{} ;
	Tensor<qint32_t> A{ 1, 4, 3 };
	A.allocateCustom(A_data, 12);
	Arange(A,  1);

    Conv2D<qint32_t> layer{ 1, 3, { 2, 1 }, { 2, 1 }, { 0, 1 }, true, PaddingMode::CIRCULAR };
    Fill(layer.weights, 1);
    Fill(layer.biases, 1);
    const auto out = layer.forward(A);
    
    qint32_t R_data[] = { 10, 6, 8, 10, 6, 22, 18, 20, 22, 18, 10, 6, 8, 10, 6, 22, 18, 20, 22, 18, 10, 6, 8, 10, 6, 22, 18, 20, 22, 18 };
	Tensor<qint32_t> R{ 1, 3, 2, 5 };
	R.allocateCustom(R_data, 30);

     EML_ASSERT_TENSOR_EQUALS(R, out);
}
inline int testConv2DCase1()
{
    qint32_t A_data[36]{} ;
	Tensor<qint32_t> A{ 3, 3, 4 };
	A.allocateCustom(A_data, 36);
	Arange(A,  1);

    Conv2D<qint32_t> layer{ 3, 1, { 1, 1 }, { 1, 2 }, { 2, 2 }, false, PaddingMode::ZEROS };
    Fill(layer.weights, 1);
    
    const auto out = layer.forward(A);
    
    qint32_t R_data[] = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 39, 45, 0, 0, 51, 57, 0, 0, 63, 69, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
	Tensor<qint32_t> R{ 1, 1, 7, 4 };
	R.allocateCustom(R_data, 28);

     EML_ASSERT_TENSOR_EQUALS(R, out);
}
inline int testConv2DCase2()
{
    qint32_t A_data[12]{} ;
	Tensor<qint32_t> A{ 1, 4, 3 };
	A.allocateCustom(A_data, 12);
	Arange(A,  1);

    Conv2D<qint32_t> layer{ 1, 3, { 1, 2 }, { 1, 1 }, { 2, 2 }, true, PaddingMode::ZEROS };
    Fill(layer.weights, 1);
    Fill(layer.biases, 1);
    const auto out = layer.forward(A);
    
    qint32_t R_data[] = { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 4, 6, 4, 1, 1, 5, 10, 12, 7, 1, 1, 8, 16, 18, 10, 1, 1, 11, 22, 24, 13, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 4, 6, 4, 1, 1, 5, 10, 12, 7, 1, 1, 8, 16, 18, 10, 1, 1, 11, 22, 24, 13, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 4, 6, 4, 1, 1, 5, 10, 12, 7, 1, 1, 8, 16, 18, 10, 1, 1, 11, 22, 24, 13, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 };
	Tensor<qint32_t> R{ 1, 3, 8, 6 };
	R.allocateCustom(R_data, 144);

     EML_ASSERT_TENSOR_EQUALS(R, out);
}
inline int testConv2DCase3()
{
    qint32_t A_data[12]{} ;
	Tensor<qint32_t> A{ 1, 4, 3 };
	A.allocateCustom(A_data, 12);
	Arange(A,  1);

    Conv2D<qint32_t> layer{ 1, 3, { 1, 1 }, { 1, 2 }, { 1, 0 }, false, PaddingMode::REFLECT };
    Fill(layer.weights, 1);
    
    const auto out = layer.forward(A);
    
    qint32_t R_data[] = { 4, 6, 1, 3, 4, 6, 7, 9, 10, 12, 7, 9, 4, 6, 1, 3, 4, 6, 7, 9, 10, 12, 7, 9, 4, 6, 1, 3, 4, 6, 7, 9, 10, 12, 7, 9 };
	Tensor<qint32_t> R{ 1, 3, 6, 2 };
	R.allocateCustom(R_data, 36);

     EML_ASSERT_TENSOR_EQUALS(R, out);
}
inline int testConv2DCase4()
{
    qint32_t A_data[36]{} ;
	Tensor<qint32_t> A{ 3, 4, 3 };
	A.allocateCustom(A_data, 36);
	Arange(A,  1);

    Conv2D<qint32_t> layer{ 3, 1, { 2, 1 }, { 1, 1 }, { 0, 0 }, true, PaddingMode::CIRCULAR };
    Fill(layer.weights, 1);
    Fill(layer.biases, 1);
    const auto out = layer.forward(A);
    
    qint32_t R_data[] = { 88, 94, 100, 106, 112, 118, 124, 130, 136 };
	Tensor<qint32_t> R{ 1, 1, 3, 3 };
	R.allocateCustom(R_data, 9);

     EML_ASSERT_TENSOR_EQUALS(R, out);
}
inline int testConv2DCase5()
{
    qint32_t A_data[48]{} ;
	Tensor<qint32_t> A{ 3, 4, 4 };
	A.allocateCustom(A_data, 48);
	Arange(A,  1);

    Conv2D<qint32_t> layer{ 3, 1, { 2, 1 }, { 1, 2 }, { 1, 0 }, false, PaddingMode::REPLICATE };
    Fill(layer.weights, 1);
    
    const auto out = layer.forward(A);
    
    qint32_t R_data[] = { 102, 114, 114, 126, 138, 150, 162, 174, 174, 186 };
	Tensor<qint32_t> R{ 1, 1, 5, 2 };
	R.allocateCustom(R_data, 10);

     EML_ASSERT_TENSOR_EQUALS(R, out);
}
inline int testConv2DCase6()
{
    qint32_t A_data[16]{} ;
	Tensor<qint32_t> A{ 1, 4, 4 };
	A.allocateCustom(A_data, 16);
	Arange(A,  1);

    Conv2D<qint32_t> layer{ 1, 3, { 2, 1 }, { 1, 1 }, { 1, 0 }, true, PaddingMode::REPLICATE };
    Fill(layer.weights, 1);
    Fill(layer.biases, 1);
    const auto out = layer.forward(A);
    
    qint32_t R_data[] = { 3, 5, 7, 9, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 27, 29, 31, 33, 3, 5, 7, 9, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 27, 29, 31, 33, 3, 5, 7, 9, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 27, 29, 31, 33 };
	Tensor<qint32_t> R{ 1, 3, 5, 4 };
	R.allocateCustom(R_data, 60);

     EML_ASSERT_TENSOR_EQUALS(R, out);
}
inline int testConv2DCase7()
{
    qint32_t A_data[36]{} ;
	Tensor<qint32_t> A{ 3, 3, 4 };
	A.allocateCustom(A_data, 36);
	Arange(A,  1);

    Conv2D<qint32_t> layer{ 3, 1, { 1, 1 }, { 1, 1 }, { 0, 0 }, false, PaddingMode::CIRCULAR };
    Fill(layer.weights, 1);
    
    const auto out = layer.forward(A);
    
    qint32_t R_data[] = { 39, 42, 45, 48, 51, 54, 57, 60, 63, 66, 69, 72 };
	Tensor<qint32_t> R{ 1, 1, 3, 4 };
	R.allocateCustom(R_data, 12);

     EML_ASSERT_TENSOR_EQUALS(R, out);
}
inline int testConv2DCase8()
{
    qint32_t A_data[48]{} ;
	Tensor<qint32_t> A{ 3, 4, 4 };
	A.allocateCustom(A_data, 48);
	Arange(A,  1);

    Conv2D<qint32_t> layer{ 3, 1, { 2, 1 }, { 2, 1 }, { 0, 1 }, true, PaddingMode::REPLICATE };
    Fill(layer.weights, 1);
    Fill(layer.biases, 1);
    const auto out = layer.forward(A);
    
    qint32_t R_data[] = { 115, 115, 121, 127, 133, 133, 163, 163, 169, 175, 181, 181 };
	Tensor<qint32_t> R{ 1, 1, 2, 6 };
	R.allocateCustom(R_data, 12);

     EML_ASSERT_TENSOR_EQUALS(R, out);
}
inline int testConv2DCase9()
{
    qint32_t A_data[12]{} ;
	Tensor<qint32_t> A{ 1, 3, 4 };
	A.allocateCustom(A_data, 12);
	Arange(A,  1);

    Conv2D<qint32_t> layer{ 1, 1, { 2, 1 }, { 1, 1 }, { 1, 0 }, false, PaddingMode::ZEROS };
    Fill(layer.weights, 1);
    
    const auto out = layer.forward(A);
    
    qint32_t R_data[] = { 1, 2, 3, 4, 6, 8, 10, 12, 14, 16, 18, 20, 9, 10, 11, 12 };
	Tensor<qint32_t> R{ 1, 1, 4, 4 };
	R.allocateCustom(R_data, 16);

     EML_ASSERT_TENSOR_EQUALS(R, out);
}
inline int testConv2DCase10()
{
    qint32_t A_data[12]{} ;
	Tensor<qint32_t> A{ 1, 4, 3 };
	A.allocateCustom(A_data, 12);
	Arange(A,  1);

    Conv2D<qint32_t> layer{ 1, 3, { 1, 2 }, { 2, 1 }, { 0, 1 }, true, PaddingMode::REFLECT };
    Fill(layer.weights, 1);
    Fill(layer.biases, 1);
    const auto out = layer.forward(A);
    
    qint32_t R_data[] = { 4, 4, 6, 6, 16, 16, 18, 18, 4, 4, 6, 6, 16, 16, 18, 18, 4, 4, 6, 6, 16, 16, 18, 18 };
	Tensor<qint32_t> R{ 1, 3, 2, 4 };
	R.allocateCustom(R_data, 24);

     EML_ASSERT_TENSOR_EQUALS(R, out);
}
inline int testConv2DCase11()
{
    qint32_t A_data[12]{} ;
	Tensor<qint32_t> A{ 1, 3, 4 };
	A.allocateCustom(A_data, 12);
	Arange(A,  1);

    Conv2D<qint32_t> layer{ 1, 1, { 1, 2 }, { 1, 2 }, { 2, 2 }, false, PaddingMode::ZEROS };
    Fill(layer.weights, 1);
    
    const auto out = layer.forward(A);
    
    qint32_t R_data[] = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 7, 0, 0, 11, 15, 0, 0, 19, 23, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
	Tensor<qint32_t> R{ 1, 1, 7, 4 };
	R.allocateCustom(R_data, 28);

     EML_ASSERT_TENSOR_EQUALS(R, out);
}
inline int testConv2DCase12()
{
    qint32_t A_data[16]{} ;
	Tensor<qint32_t> A{ 1, 4, 4 };
	A.allocateCustom(A_data, 16);
	Arange(A,  1);

    Conv2D<qint32_t> layer{ 1, 1, { 1, 2 }, { 1, 1 }, { 0, 1 }, true, PaddingMode::REFLECT };
    Fill(layer.weights, 1);
    Fill(layer.biases, 1);
    const auto out = layer.forward(A);
    
    qint32_t R_data[] = { 4, 4, 6, 8, 8, 12, 12, 14, 16, 16, 20, 20, 22, 24, 24, 28, 28, 30, 32, 32 };
	Tensor<qint32_t> R{ 1, 1, 4, 5 };
	R.allocateCustom(R_data, 20);

     EML_ASSERT_TENSOR_EQUALS(R, out);
}
inline int testConv2DCase13()
{
    qint32_t A_data[12]{} ;
	Tensor<qint32_t> A{ 1, 4, 3 };
	A.allocateCustom(A_data, 12);
	Arange(A,  1);

    Conv2D<qint32_t> layer{ 1, 3, { 1, 1 }, { 2, 1 }, { 1, 0 }, false, PaddingMode::REPLICATE };
    Fill(layer.weights, 1);
    
    const auto out = layer.forward(A);
    
    qint32_t R_data[] = { 1, 2, 3, 4, 5, 6, 10, 11, 12, 1, 2, 3, 4, 5, 6, 10, 11, 12, 1, 2, 3, 4, 5, 6, 10, 11, 12 };
	Tensor<qint32_t> R{ 1, 3, 3, 3 };
	R.allocateCustom(R_data, 27);

     EML_ASSERT_TENSOR_EQUALS(R, out);
}
inline int testConv2DCase14()
{
    qint32_t A_data[36]{} ;
	Tensor<qint32_t> A{ 3, 4, 3 };
	A.allocateCustom(A_data, 36);
	Arange(A,  1);

    Conv2D<qint32_t> layer{ 3, 1, { 1, 1 }, { 2, 1 }, { 1, 0 }, true, PaddingMode::ZEROS };
    Fill(layer.weights, 1);
    Fill(layer.biases, 1);
    const auto out = layer.forward(A);
    
    qint32_t R_data[] = { 1, 1, 1, 49, 52, 55, 67, 70, 73 };
	Tensor<qint32_t> R{ 1, 1, 3, 3 };
	R.allocateCustom(R_data, 9);

     EML_ASSERT_TENSOR_EQUALS(R, out);
}
inline int testConv2DCase15()
{
    qint32_t A_data[12]{} ;
	Tensor<qint32_t> A{ 1, 3, 4 };
	A.allocateCustom(A_data, 12);
	Arange(A,  1);

    Conv2D<qint32_t> layer{ 1, 1, { 1, 2 }, { 1, 1 }, { 0, 1 }, false, PaddingMode::REPLICATE };
    Fill(layer.weights, 1);
    
    const auto out = layer.forward(A);
    
    qint32_t R_data[] = { 2, 3, 5, 7, 8, 10, 11, 13, 15, 16, 18, 19, 21, 23, 24 };
	Tensor<qint32_t> R{ 1, 1, 3, 5 };
	R.allocateCustom(R_data, 15);

     EML_ASSERT_TENSOR_EQUALS(R, out);
}
inline int testConv2DCase16()
{
    qint32_t A_data[36]{} ;
	Tensor<qint32_t> A{ 3, 3, 4 };
	A.allocateCustom(A_data, 36);
	Arange(A,  1);

    Conv2D<qint32_t> layer{ 3, 1, { 2, 1 }, { 2, 1 }, { 0, 0 }, true, PaddingMode::REPLICATE };
    Fill(layer.weights, 1);
    Fill(layer.biases, 1);
    const auto out = layer.forward(A);
    
    qint32_t R_data[] = { 91, 97, 103, 109 };
	Tensor<qint32_t> R{ 1, 1, 1, 4 };
	R.allocateCustom(R_data, 4);

     EML_ASSERT_TENSOR_EQUALS(R, out);
}
inline int testConv2DCase17()
{
    qint32_t A_data[12]{} ;
	Tensor<qint32_t> A{ 1, 4, 3 };
	A.allocateCustom(A_data, 12);
	Arange(A,  1);

    Conv2D<qint32_t> layer{ 1, 1, { 2, 1 }, { 1, 1 }, { 0, 0 }, false, PaddingMode::ZEROS };
    Fill(layer.weights, 1);
    
    const auto out = layer.forward(A);
    
    qint32_t R_data[] = { 5, 7, 9, 11, 13, 15, 17, 19, 21 };
	Tensor<qint32_t> R{ 1, 1, 3, 3 };
	R.allocateCustom(R_data, 9);

     EML_ASSERT_TENSOR_EQUALS(R, out);
}
inline int testConv2DCase18()
{
    qint32_t A_data[12]{} ;
	Tensor<qint32_t> A{ 1, 4, 3 };
	A.allocateCustom(A_data, 12);
	Arange(A,  1);

    Conv2D<qint32_t> layer{ 1, 3, { 1, 2 }, { 2, 1 }, { 0, 0 }, true, PaddingMode::REFLECT };
    Fill(layer.weights, 1);
    Fill(layer.biases, 1);
    const auto out = layer.forward(A);
    
    qint32_t R_data[] = { 4, 6, 16, 18, 4, 6, 16, 18, 4, 6, 16, 18 };
	Tensor<qint32_t> R{ 1, 3, 2, 2 };
	R.allocateCustom(R_data, 12);

     EML_ASSERT_TENSOR_EQUALS(R, out);
}
inline int testConv2DCase19()
{
    qint32_t A_data[36]{} ;
	Tensor<qint32_t> A{ 3, 3, 4 };
	A.allocateCustom(A_data, 36);
	Arange(A,  1);

    Conv2D<qint32_t> layer{ 3, 1, { 1, 2 }, { 1, 2 }, { 0, 0 }, false, PaddingMode::REFLECT };
    Fill(layer.weights, 1);
    
    const auto out = layer.forward(A);
    
    qint32_t R_data[] = { 81, 93, 105, 117, 129, 141 };
	Tensor<qint32_t> R{ 1, 1, 3, 2 };
	R.allocateCustom(R_data, 6);

     EML_ASSERT_TENSOR_EQUALS(R, out);
}
inline int testConv2DCase20()
{
    qint32_t A_data[36]{} ;
	Tensor<qint32_t> A{ 3, 3, 4 };
	A.allocateCustom(A_data, 36);
	Arange(A,  1);

    Conv2D<qint32_t> layer{ 3, 1, { 2, 1 }, { 2, 1 }, { 0, 0 }, true, PaddingMode::CIRCULAR };
    Fill(layer.weights, 1);
    Fill(layer.biases, 1);
    const auto out = layer.forward(A);
    
    qint32_t R_data[] = { 91, 97, 103, 109 };
	Tensor<qint32_t> R{ 1, 1, 1, 4 };
	R.allocateCustom(R_data, 4);

     EML_ASSERT_TENSOR_EQUALS(R, out);
}
inline int testConv2DCase21()
{
    qint32_t A_data[12]{} ;
	Tensor<qint32_t> A{ 1, 3, 4 };
	A.allocateCustom(A_data, 12);
	Arange(A,  1);

    Conv2D<qint32_t> layer{ 1, 1, { 2, 1 }, { 2, 1 }, { 2, 2 }, false, PaddingMode::REFLECT };
    Fill(layer.weights, 1);
    
    const auto out = layer.forward(A);
    
    qint32_t R_data[] = { 18, 16, 14, 16, 18, 20, 18, 16, 10, 8, 6, 8, 10, 12, 10, 8, 18, 16, 14, 16, 18, 20, 18, 16 };
	Tensor<qint32_t> R{ 1, 1, 3, 8 };
	R.allocateCustom(R_data, 24);

     EML_ASSERT_TENSOR_EQUALS(R, out);
}
inline int testConv2DCase22()
{
    qint32_t A_data[36]{} ;
	Tensor<qint32_t> A{ 3, 3, 4 };
	A.allocateCustom(A_data, 36);
	Arange(A,  1);

    Conv2D<qint32_t> layer{ 3, 1, { 1, 1 }, { 2, 1 }, { 1, 0 }, true, PaddingMode::ZEROS };
    Fill(layer.weights, 1);
    Fill(layer.biases, 1);
    const auto out = layer.forward(A);
    
    qint32_t R_data[] = { 1, 1, 1, 1, 52, 55, 58, 61, 1, 1, 1, 1 };
	Tensor<qint32_t> R{ 1, 1, 3, 4 };
	R.allocateCustom(R_data, 12);

     EML_ASSERT_TENSOR_EQUALS(R, out);
}
inline int testConv2DCase23()
{
    qint32_t A_data[12]{} ;
	Tensor<qint32_t> A{ 1, 3, 4 };
	A.allocateCustom(A_data, 12);
	Arange(A,  1);

    Conv2D<qint32_t> layer{ 1, 3, { 1, 1 }, { 1, 1 }, { 0, 1 }, false, PaddingMode::ZEROS };
    Fill(layer.weights, 1);
    
    const auto out = layer.forward(A);
    
    qint32_t R_data[] = { 0, 1, 2, 3, 4, 0, 0, 5, 6, 7, 8, 0, 0, 9, 10, 11, 12, 0, 0, 1, 2, 3, 4, 0, 0, 5, 6, 7, 8, 0, 0, 9, 10, 11, 12, 0, 0, 1, 2, 3, 4, 0, 0, 5, 6, 7, 8, 0, 0, 9, 10, 11, 12, 0 };
	Tensor<qint32_t> R{ 1, 3, 3, 6 };
	R.allocateCustom(R_data, 54);

     EML_ASSERT_TENSOR_EQUALS(R, out);
}
inline int testConv2DCase24()
{
    qint32_t A_data[48]{} ;
	Tensor<qint32_t> A{ 3, 4, 4 };
	A.allocateCustom(A_data, 48);
	Arange(A,  1);

    Conv2D<qint32_t> layer{ 3, 1, { 1, 2 }, { 1, 1 }, { 0, 1 }, true, PaddingMode::REFLECT };
    Fill(layer.weights, 1);
    Fill(layer.biases, 1);
    const auto out = layer.forward(A);
    
    qint32_t R_data[] = { 106, 106, 112, 118, 118, 130, 130, 136, 142, 142, 154, 154, 160, 166, 166, 178, 178, 184, 190, 190 };
	Tensor<qint32_t> R{ 1, 1, 4, 5 };
	R.allocateCustom(R_data, 20);

     EML_ASSERT_TENSOR_EQUALS(R, out);
}
inline int testConv2DCase25()
{
    qint32_t A_data[16]{} ;
	Tensor<qint32_t> A{ 1, 4, 4 };
	A.allocateCustom(A_data, 16);
	Arange(A,  1);

    Conv2D<qint32_t> layer{ 1, 3, { 1, 1 }, { 1, 1 }, { 1, 0 }, false, PaddingMode::REPLICATE };
    Fill(layer.weights, 1);
    
    const auto out = layer.forward(A);
    
    qint32_t R_data[] = { 1, 2, 3, 4, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 13, 14, 15, 16, 1, 2, 3, 4, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 13, 14, 15, 16, 1, 2, 3, 4, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 13, 14, 15, 16 };
	Tensor<qint32_t> R{ 1, 3, 6, 4 };
	R.allocateCustom(R_data, 72);

     EML_ASSERT_TENSOR_EQUALS(R, out);
}
inline int testConv2DCase26()
{
    qint32_t A_data[12]{} ;
	Tensor<qint32_t> A{ 1, 3, 4 };
	A.allocateCustom(A_data, 12);
	Arange(A,  1);

    Conv2D<qint32_t> layer{ 1, 3, { 1, 2 }, { 1, 1 }, { 1, 0 }, true, PaddingMode::REFLECT };
    Fill(layer.weights, 1);
    Fill(layer.biases, 1);
    const auto out = layer.forward(A);
    
    qint32_t R_data[] = { 12, 14, 16, 4, 6, 8, 12, 14, 16, 20, 22, 24, 12, 14, 16, 12, 14, 16, 4, 6, 8, 12, 14, 16, 20, 22, 24, 12, 14, 16, 12, 14, 16, 4, 6, 8, 12, 14, 16, 20, 22, 24, 12, 14, 16 };
	Tensor<qint32_t> R{ 1, 3, 5, 3 };
	R.allocateCustom(R_data, 45);

     EML_ASSERT_TENSOR_EQUALS(R, out);
}
inline int testConv2DCase27()
{
    qint32_t A_data[12]{} ;
	Tensor<qint32_t> A{ 1, 3, 4 };
	A.allocateCustom(A_data, 12);
	Arange(A,  1);

    Conv2D<qint32_t> layer{ 1, 1, { 1, 2 }, { 1, 2 }, { 1, 0 }, false, PaddingMode::REFLECT };
    Fill(layer.weights, 1);
    
    const auto out = layer.forward(A);
    
    qint32_t R_data[] = { 11, 15, 3, 7, 11, 15, 19, 23, 11, 15 };
	Tensor<qint32_t> R{ 1, 1, 5, 2 };
	R.allocateCustom(R_data, 10);

     EML_ASSERT_TENSOR_EQUALS(R, out);
}
inline int testConv2DCase28()
{
    qint32_t A_data[48]{} ;
	Tensor<qint32_t> A{ 3, 4, 4 };
	A.allocateCustom(A_data, 48);
	Arange(A,  1);

    Conv2D<qint32_t> layer{ 3, 1, { 2, 1 }, { 1, 1 }, { 0, 0 }, true, PaddingMode::REFLECT };
    Fill(layer.weights, 1);
    Fill(layer.biases, 1);
    const auto out = layer.forward(A);
    
    qint32_t R_data[] = { 115, 121, 127, 133, 139, 145, 151, 157, 163, 169, 175, 181 };
	Tensor<qint32_t> R{ 1, 1, 3, 4 };
	R.allocateCustom(R_data, 12);

     EML_ASSERT_TENSOR_EQUALS(R, out);
}
inline int testConv2DCase29()
{
    qint32_t A_data[12]{} ;
	Tensor<qint32_t> A{ 1, 4, 3 };
	A.allocateCustom(A_data, 12);
	Arange(A,  1);

    Conv2D<qint32_t> layer{ 1, 1, { 2, 1 }, { 1, 1 }, { 1, 0 }, false, PaddingMode::CIRCULAR };
    Fill(layer.weights, 1);
    
    const auto out = layer.forward(A);
    
    qint32_t R_data[] = { 11, 13, 15, 5, 7, 9, 11, 13, 15, 17, 19, 21, 11, 13, 15 };
	Tensor<qint32_t> R{ 1, 1, 5, 3 };
	R.allocateCustom(R_data, 15);

     EML_ASSERT_TENSOR_EQUALS(R, out);
}
inline int testConv2DCase30()
{
    qint32_t A_data[16]{} ;
	Tensor<qint32_t> A{ 1, 4, 4 };
	A.allocateCustom(A_data, 16);
	Arange(A,  1);

    Conv2D<qint32_t> layer{ 1, 1, { 2, 1 }, { 2, 1 }, { 2, 2 }, true, PaddingMode::REPLICATE };
    Fill(layer.weights, 1);
    Fill(layer.biases, 1);
    const auto out = layer.forward(A);
    
    qint32_t R_data[] = { 3, 3, 3, 5, 7, 9, 9, 9, 7, 7, 7, 9, 11, 13, 13, 13, 23, 23, 23, 25, 27, 29, 29, 29, 27, 27, 27, 29, 31, 33, 33, 33 };
	Tensor<qint32_t> R{ 1, 1, 4, 8 };
	R.allocateCustom(R_data, 32);

     EML_ASSERT_TENSOR_EQUALS(R, out);
}
inline int testConv2DCase31()
{
    qint32_t A_data[48]{} ;
	Tensor<qint32_t> A{ 3, 4, 4 };
	A.allocateCustom(A_data, 48);
	Arange(A,  1);

    Conv2D<qint32_t> layer{ 3, 1, { 2, 1 }, { 2, 1 }, { 0, 1 }, false, PaddingMode::REPLICATE };
    Fill(layer.weights, 1);
    
    const auto out = layer.forward(A);
    
    qint32_t R_data[] = { 114, 114, 120, 126, 132, 132, 162, 162, 168, 174, 180, 180 };
	Tensor<qint32_t> R{ 1, 1, 2, 6 };
	R.allocateCustom(R_data, 12);

     EML_ASSERT_TENSOR_EQUALS(R, out);
}
inline int testConv2DCase32()
{
    qint32_t A_data[36]{} ;
	Tensor<qint32_t> A{ 3, 4, 3 };
	A.allocateCustom(A_data, 36);
	Arange(A,  1);

    Conv2D<qint32_t> layer{ 3, 1, { 1, 2 }, { 2, 1 }, { 1, 0 }, true, PaddingMode::CIRCULAR };
    Fill(layer.weights, 1);
    Fill(layer.biases, 1);
    const auto out = layer.forward(A);
    
    qint32_t R_data[] = { 136, 142, 100, 106, 136, 142 };
	Tensor<qint32_t> R{ 1, 1, 3, 2 };
	R.allocateCustom(R_data, 6);

     EML_ASSERT_TENSOR_EQUALS(R, out);
}
inline int testConv2DCase33()
{
    qint32_t A_data[12]{} ;
	Tensor<qint32_t> A{ 1, 3, 4 };
	A.allocateCustom(A_data, 12);
	Arange(A,  1);

    Conv2D<qint32_t> layer{ 1, 3, { 1, 2 }, { 2, 1 }, { 0, 0 }, false, PaddingMode::REPLICATE };
    Fill(layer.weights, 1);
    
    const auto out = layer.forward(A);
    
    qint32_t R_data[] = { 3, 5, 7, 19, 21, 23, 3, 5, 7, 19, 21, 23, 3, 5, 7, 19, 21, 23 };
	Tensor<qint32_t> R{ 1, 3, 2, 3 };
	R.allocateCustom(R_data, 18);

     EML_ASSERT_TENSOR_EQUALS(R, out);
}
inline int testConv2DCase34()
{
    qint32_t A_data[16]{} ;
	Tensor<qint32_t> A{ 1, 4, 4 };
	A.allocateCustom(A_data, 16);
	Arange(A,  1);

    Conv2D<qint32_t> layer{ 1, 1, { 1, 2 }, { 1, 2 }, { 2, 2 }, true, PaddingMode::REPLICATE };
    Fill(layer.weights, 1);
    Fill(layer.biases, 1);
    const auto out = layer.forward(A);
    
    qint32_t R_data[] = { 3, 4, 8, 9, 3, 4, 8, 9, 3, 4, 8, 9, 11, 12, 16, 17, 19, 20, 24, 25, 27, 28, 32, 33, 27, 28, 32, 33, 27, 28, 32, 33 };
	Tensor<qint32_t> R{ 1, 1, 8, 4 };
	R.allocateCustom(R_data, 32);

     EML_ASSERT_TENSOR_EQUALS(R, out);
}
inline int testConv2DCase35()
{
    qint32_t A_data[12]{} ;
	Tensor<qint32_t> A{ 1, 4, 3 };
	A.allocateCustom(A_data, 12);
	Arange(A,  1);

    Conv2D<qint32_t> layer{ 1, 3, { 1, 1 }, { 1, 1 }, { 2, 2 }, false, PaddingMode::REFLECT };
    Fill(layer.weights, 1);
    
    const auto out = layer.forward(A);
    
    qint32_t R_data[] = { 9, 8, 7, 8, 9, 8, 7, 6, 5, 4, 5, 6, 5, 4, 3, 2, 1, 2, 3, 2, 1, 6, 5, 4, 5, 6, 5, 4, 9, 8, 7, 8, 9, 8, 7, 12, 11, 10, 11, 12, 11, 10, 9, 8, 7, 8, 9, 8, 7, 6, 5, 4, 5, 6, 5, 4, 9, 8, 7, 8, 9, 8, 7, 6, 5, 4, 5, 6, 5, 4, 3, 2, 1, 2, 3, 2, 1, 6, 5, 4, 5, 6, 5, 4, 9, 8, 7, 8, 9, 8, 7, 12, 11, 10, 11, 12, 11, 10, 9, 8, 7, 8, 9, 8, 7, 6, 5, 4, 5, 6, 5, 4, 9, 8, 7, 8, 9, 8, 7, 6, 5, 4, 5, 6, 5, 4, 3, 2, 1, 2, 3, 2, 1, 6, 5, 4, 5, 6, 5, 4, 9, 8, 7, 8, 9, 8, 7, 12, 11, 10, 11, 12, 11, 10, 9, 8, 7, 8, 9, 8, 7, 6, 5, 4, 5, 6, 5, 4 };
	Tensor<qint32_t> R{ 1, 3, 8, 7 };
	R.allocateCustom(R_data, 168);

     EML_ASSERT_TENSOR_EQUALS(R, out);
}
inline int testConv2DCase36()
{
    qint32_t A_data[12]{} ;
	Tensor<qint32_t> A{ 1, 4, 3 };
	A.allocateCustom(A_data, 12);
	Arange(A,  1);

    Conv2D<qint32_t> layer{ 1, 1, { 1, 1 }, { 2, 1 }, { 2, 2 }, true, PaddingMode::CIRCULAR };
    Fill(layer.weights, 1);
    Fill(layer.biases, 1);
    const auto out = layer.forward(A);
    
    qint32_t R_data[] = { 9, 10, 8, 9, 10, 8, 9, 3, 4, 2, 3, 4, 2, 3, 9, 10, 8, 9, 10, 8, 9, 3, 4, 2, 3, 4, 2, 3 };
	Tensor<qint32_t> R{ 1, 1, 4, 7 };
	R.allocateCustom(R_data, 28);

     EML_ASSERT_TENSOR_EQUALS(R, out);
}
inline int testConv2DCase37()
{
    qint32_t A_data[36]{} ;
	Tensor<qint32_t> A{ 3, 3, 4 };
	A.allocateCustom(A_data, 36);
	Arange(A,  1);

    Conv2D<qint32_t> layer{ 3, 1, { 2, 1 }, { 1, 1 }, { 2, 2 }, false, PaddingMode::REPLICATE };
    Fill(layer.weights, 1);
    
    const auto out = layer.forward(A);
    
    qint32_t R_data[] = { 78, 78, 78, 84, 90, 96, 96, 96, 78, 78, 78, 84, 90, 96, 96, 96, 90, 90, 90, 96, 102, 108, 108, 108, 114, 114, 114, 120, 126, 132, 132, 132, 126, 126, 126, 132, 138, 144, 144, 144, 126, 126, 126, 132, 138, 144, 144, 144 };
	Tensor<qint32_t> R{ 1, 1, 6, 8 };
	R.allocateCustom(R_data, 48);

     EML_ASSERT_TENSOR_EQUALS(R, out);
}
inline int testConv2DCase38()
{
    qint32_t A_data[36]{} ;
	Tensor<qint32_t> A{ 3, 4, 3 };
	A.allocateCustom(A_data, 36);
	Arange(A,  1);

    Conv2D<qint32_t> layer{ 3, 1, { 2, 1 }, { 2, 1 }, { 1, 0 }, true, PaddingMode::CIRCULAR };
    Fill(layer.weights, 1);
    Fill(layer.biases, 1);
    const auto out = layer.forward(A);
    
    qint32_t R_data[] = { 106, 112, 118, 106, 112, 118, 106, 112, 118 };
	Tensor<qint32_t> R{ 1, 1, 3, 3 };
	R.allocateCustom(R_data, 9);

     EML_ASSERT_TENSOR_EQUALS(R, out);
}
inline int testConv2DCase39()
{
    qint32_t A_data[16]{} ;
	Tensor<qint32_t> A{ 1, 4, 4 };
	A.allocateCustom(A_data, 16);
	Arange(A,  1);

    Conv2D<qint32_t> layer{ 1, 3, { 1, 1 }, { 1, 2 }, { 0, 0 }, false, PaddingMode::REFLECT };
    Fill(layer.weights, 1);
    
    const auto out = layer.forward(A);
    
    qint32_t R_data[] = { 1, 3, 5, 7, 9, 11, 13, 15, 1, 3, 5, 7, 9, 11, 13, 15, 1, 3, 5, 7, 9, 11, 13, 15 };
	Tensor<qint32_t> R{ 1, 3, 4, 2 };
	R.allocateCustom(R_data, 24);

     EML_ASSERT_TENSOR_EQUALS(R, out);
}
inline int testConv2DCase40()
{
    qint32_t A_data[36]{} ;
	Tensor<qint32_t> A{ 3, 4, 3 };
	A.allocateCustom(A_data, 36);
	Arange(A,  1);

    Conv2D<qint32_t> layer{ 3, 1, { 2, 1 }, { 1, 2 }, { 0, 0 }, true, PaddingMode::REFLECT };
    Fill(layer.weights, 1);
    Fill(layer.biases, 1);
    const auto out = layer.forward(A);
    
    qint32_t R_data[] = { 88, 100, 106, 118, 124, 136 };
	Tensor<qint32_t> R{ 1, 1, 3, 2 };
	R.allocateCustom(R_data, 6);

     EML_ASSERT_TENSOR_EQUALS(R, out);
}
inline int testConv2DCase41()
{
    qint32_t A_data[48]{} ;
	Tensor<qint32_t> A{ 3, 4, 4 };
	A.allocateCustom(A_data, 48);
	Arange(A,  1);

    Conv2D<qint32_t> layer{ 3, 1, { 1, 1 }, { 1, 2 }, { 2, 2 }, false, PaddingMode::CIRCULAR };
    Fill(layer.weights, 1);
    
    const auto out = layer.forward(A);
    
    qint32_t R_data[] = { 81, 75, 81, 75, 93, 87, 93, 87, 57, 51, 57, 51, 69, 63, 69, 63, 81, 75, 81, 75, 93, 87, 93, 87, 57, 51, 57, 51, 69, 63, 69, 63 };
	Tensor<qint32_t> R{ 1, 1, 8, 4 };
	R.allocateCustom(R_data, 32);

     EML_ASSERT_TENSOR_EQUALS(R, out);
}
inline int testConv2DCase42()
{
    qint32_t A_data[48]{} ;
	Tensor<qint32_t> A{ 3, 4, 4 };
	A.allocateCustom(A_data, 48);
	Arange(A,  1);

    Conv2D<qint32_t> layer{ 3, 1, { 1, 2 }, { 1, 2 }, { 0, 1 }, true, PaddingMode::REFLECT };
    Fill(layer.weights, 1);
    Fill(layer.biases, 1);
    const auto out = layer.forward(A);
    
    qint32_t R_data[] = { 106, 112, 118, 130, 136, 142, 154, 160, 166, 178, 184, 190 };
	Tensor<qint32_t> R{ 1, 1, 4, 3 };
	R.allocateCustom(R_data, 12);

     EML_ASSERT_TENSOR_EQUALS(R, out);
}
inline int testConv2DCase43()
{
    qint32_t A_data[16]{} ;
	Tensor<qint32_t> A{ 1, 4, 4 };
	A.allocateCustom(A_data, 16);
	Arange(A,  1);

    Conv2D<qint32_t> layer{ 1, 3, { 1, 1 }, { 1, 2 }, { 1, 0 }, false, PaddingMode::REFLECT };
    Fill(layer.weights, 1);
    
    const auto out = layer.forward(A);
    
    qint32_t R_data[] = { 5, 7, 1, 3, 5, 7, 9, 11, 13, 15, 9, 11, 5, 7, 1, 3, 5, 7, 9, 11, 13, 15, 9, 11, 5, 7, 1, 3, 5, 7, 9, 11, 13, 15, 9, 11 };
	Tensor<qint32_t> R{ 1, 3, 6, 2 };
	R.allocateCustom(R_data, 36);

     EML_ASSERT_TENSOR_EQUALS(R, out);
}
inline int testConv2DCase44()
{
    qint32_t A_data[12]{} ;
	Tensor<qint32_t> A{ 1, 4, 3 };
	A.allocateCustom(A_data, 12);
	Arange(A,  1);

    Conv2D<qint32_t> layer{ 1, 1, { 1, 2 }, { 2, 1 }, { 0, 0 }, true, PaddingMode::CIRCULAR };
    Fill(layer.weights, 1);
    Fill(layer.biases, 1);
    const auto out = layer.forward(A);
    
    qint32_t R_data[] = { 4, 6, 16, 18 };
	Tensor<qint32_t> R{ 1, 1, 2, 2 };
	R.allocateCustom(R_data, 4);

     EML_ASSERT_TENSOR_EQUALS(R, out);
}
inline int testConv2DCase45()
{
    qint32_t A_data[12]{} ;
	Tensor<qint32_t> A{ 1, 3, 4 };
	A.allocateCustom(A_data, 12);
	Arange(A,  1);

    Conv2D<qint32_t> layer{ 1, 1, { 1, 1 }, { 2, 1 }, { 1, 0 }, false, PaddingMode::CIRCULAR };
    Fill(layer.weights, 1);
    
    const auto out = layer.forward(A);
    
    qint32_t R_data[] = { 9, 10, 11, 12, 5, 6, 7, 8, 1, 2, 3, 4 };
	Tensor<qint32_t> R{ 1, 1, 3, 4 };
	R.allocateCustom(R_data, 12);

     EML_ASSERT_TENSOR_EQUALS(R, out);
}
inline int testConv2DCase46()
{
    qint32_t A_data[12]{} ;
	Tensor<qint32_t> A{ 1, 4, 3 };
	A.allocateCustom(A_data, 12);
	Arange(A,  1);

    Conv2D<qint32_t> layer{ 1, 3, { 1, 2 }, { 1, 2 }, { 0, 1 }, true, PaddingMode::CIRCULAR };
    Fill(layer.weights, 1);
    Fill(layer.biases, 1);
    const auto out = layer.forward(A);
    
    qint32_t R_data[] = { 5, 6, 11, 12, 17, 18, 23, 24, 5, 6, 11, 12, 17, 18, 23, 24, 5, 6, 11, 12, 17, 18, 23, 24 };
	Tensor<qint32_t> R{ 1, 3, 4, 2 };
	R.allocateCustom(R_data, 24);

     EML_ASSERT_TENSOR_EQUALS(R, out);
}
inline int testConv2DCase47()
{
    qint32_t A_data[12]{} ;
	Tensor<qint32_t> A{ 1, 3, 4 };
	A.allocateCustom(A_data, 12);
	Arange(A,  1);

    Conv2D<qint32_t> layer{ 1, 1, { 1, 2 }, { 1, 2 }, { 1, 0 }, false, PaddingMode::REPLICATE };
    Fill(layer.weights, 1);
    
    const auto out = layer.forward(A);
    
    qint32_t R_data[] = { 3, 7, 3, 7, 11, 15, 19, 23, 19, 23 };
	Tensor<qint32_t> R{ 1, 1, 5, 2 };
	R.allocateCustom(R_data, 10);

     EML_ASSERT_TENSOR_EQUALS(R, out);
}
inline int testConv2DCase48()
{
    qint32_t A_data[12]{} ;
	Tensor<qint32_t> A{ 1, 4, 3 };
	A.allocateCustom(A_data, 12);
	Arange(A,  1);

    Conv2D<qint32_t> layer{ 1, 1, { 1, 2 }, { 2, 1 }, { 0, 0 }, true, PaddingMode::CIRCULAR };
    Fill(layer.weights, 1);
    Fill(layer.biases, 1);
    const auto out = layer.forward(A);
    
    qint32_t R_data[] = { 4, 6, 16, 18 };
	Tensor<qint32_t> R{ 1, 1, 2, 2 };
	R.allocateCustom(R_data, 4);

     EML_ASSERT_TENSOR_EQUALS(R, out);
}
inline int testConv2DCase49()
{
    qint32_t A_data[12]{} ;
	Tensor<qint32_t> A{ 1, 4, 3 };
	A.allocateCustom(A_data, 12);
	Arange(A,  1);

    Conv2D<qint32_t> layer{ 1, 3, { 1, 1 }, { 1, 1 }, { 2, 2 }, false, PaddingMode::REFLECT };
    Fill(layer.weights, 1);
    
    const auto out = layer.forward(A);
    
    qint32_t R_data[] = { 9, 8, 7, 8, 9, 8, 7, 6, 5, 4, 5, 6, 5, 4, 3, 2, 1, 2, 3, 2, 1, 6, 5, 4, 5, 6, 5, 4, 9, 8, 7, 8, 9, 8, 7, 12, 11, 10, 11, 12, 11, 10, 9, 8, 7, 8, 9, 8, 7, 6, 5, 4, 5, 6, 5, 4, 9, 8, 7, 8, 9, 8, 7, 6, 5, 4, 5, 6, 5, 4, 3, 2, 1, 2, 3, 2, 1, 6, 5, 4, 5, 6, 5, 4, 9, 8, 7, 8, 9, 8, 7, 12, 11, 10, 11, 12, 11, 10, 9, 8, 7, 8, 9, 8, 7, 6, 5, 4, 5, 6, 5, 4, 9, 8, 7, 8, 9, 8, 7, 6, 5, 4, 5, 6, 5, 4, 3, 2, 1, 2, 3, 2, 1, 6, 5, 4, 5, 6, 5, 4, 9, 8, 7, 8, 9, 8, 7, 12, 11, 10, 11, 12, 11, 10, 9, 8, 7, 8, 9, 8, 7, 6, 5, 4, 5, 6, 5, 4 };
	Tensor<qint32_t> R{ 1, 3, 8, 7 };
	R.allocateCustom(R_data, 168);

     EML_ASSERT_TENSOR_EQUALS(R, out);
}
inline int testConv2DCase50()
{
    qint32_t A_data[12]{} ;
	Tensor<qint32_t> A{ 1, 4, 3 };
	A.allocateCustom(A_data, 12);
	Arange(A,  1);

    Conv2D<qint32_t> layer{ 1, 1, { 1, 2 }, { 1, 2 }, { 0, 1 }, true, PaddingMode::REFLECT };
    Fill(layer.weights, 1);
    Fill(layer.biases, 1);
    const auto out = layer.forward(A);
    
    qint32_t R_data[] = { 4, 6, 10, 12, 16, 18, 22, 24 };
	Tensor<qint32_t> R{ 1, 1, 4, 2 };
	R.allocateCustom(R_data, 8);

     EML_ASSERT_TENSOR_EQUALS(R, out);
}
inline int testConv2DCase51()
{
    qint32_t A_data[12]{} ;
	Tensor<qint32_t> A{ 1, 3, 4 };
	A.allocateCustom(A_data, 12);
	Arange(A,  1);

    Conv2D<qint32_t> layer{ 1, 1, { 1, 1 }, { 2, 1 }, { 0, 0 }, false, PaddingMode::ZEROS };
    Fill(layer.weights, 1);
    
    const auto out = layer.forward(A);
    
    qint32_t R_data[] = { 1, 2, 3, 4, 9, 10, 11, 12 };
	Tensor<qint32_t> R{ 1, 1, 2, 4 };
	R.allocateCustom(R_data, 8);

     EML_ASSERT_TENSOR_EQUALS(R, out);
}
inline int testConv2DCase52()
{
    qint32_t A_data[16]{} ;
	Tensor<qint32_t> A{ 1, 4, 4 };
	A.allocateCustom(A_data, 16);
	Arange(A,  1);

    Conv2D<qint32_t> layer{ 1, 1, { 1, 2 }, { 1, 1 }, { 1, 0 }, true, PaddingMode::REPLICATE };
    Fill(layer.weights, 1);
    Fill(layer.biases, 1);
    const auto out = layer.forward(A);
    
    qint32_t R_data[] = { 4, 6, 8, 4, 6, 8, 12, 14, 16, 20, 22, 24, 28, 30, 32, 28, 30, 32 };
	Tensor<qint32_t> R{ 1, 1, 6, 3 };
	R.allocateCustom(R_data, 18);

     EML_ASSERT_TENSOR_EQUALS(R, out);
}
inline int testConv2DCase53()
{
    qint32_t A_data[36]{} ;
	Tensor<qint32_t> A{ 3, 3, 4 };
	A.allocateCustom(A_data, 36);
	Arange(A,  1);

    Conv2D<qint32_t> layer{ 3, 1, { 2, 1 }, { 1, 1 }, { 1, 0 }, false, PaddingMode::CIRCULAR };
    Fill(layer.weights, 1);
    
    const auto out = layer.forward(A);
    
    qint32_t R_data[] = { 102, 108, 114, 120, 90, 96, 102, 108, 114, 120, 126, 132, 102, 108, 114, 120 };
	Tensor<qint32_t> R{ 1, 1, 4, 4 };
	R.allocateCustom(R_data, 16);

     EML_ASSERT_TENSOR_EQUALS(R, out);
}
inline int testConv2DCase54()
{
    qint32_t A_data[12]{} ;
	Tensor<qint32_t> A{ 1, 4, 3 };
	A.allocateCustom(A_data, 12);
	Arange(A,  1);

    Conv2D<qint32_t> layer{ 1, 3, { 1, 2 }, { 2, 1 }, { 1, 0 }, true, PaddingMode::CIRCULAR };
    Fill(layer.weights, 1);
    Fill(layer.biases, 1);
    const auto out = layer.forward(A);
    
    qint32_t R_data[] = { 22, 24, 10, 12, 22, 24, 22, 24, 10, 12, 22, 24, 22, 24, 10, 12, 22, 24 };
	Tensor<qint32_t> R{ 1, 3, 3, 2 };
	R.allocateCustom(R_data, 18);

     EML_ASSERT_TENSOR_EQUALS(R, out);
}
inline int testConv2DCase55()
{
    qint32_t A_data[16]{} ;
	Tensor<qint32_t> A{ 1, 4, 4 };
	A.allocateCustom(A_data, 16);
	Arange(A,  1);

    Conv2D<qint32_t> layer{ 1, 1, { 1, 2 }, { 1, 1 }, { 0, 1 }, false, PaddingMode::REFLECT };
    Fill(layer.weights, 1);
    
    const auto out = layer.forward(A);
    
    qint32_t R_data[] = { 3, 3, 5, 7, 7, 11, 11, 13, 15, 15, 19, 19, 21, 23, 23, 27, 27, 29, 31, 31 };
	Tensor<qint32_t> R{ 1, 1, 4, 5 };
	R.allocateCustom(R_data, 20);

     EML_ASSERT_TENSOR_EQUALS(R, out);
}
inline int testConv2DCase56()
{
    qint32_t A_data[12]{} ;
	Tensor<qint32_t> A{ 1, 4, 3 };
	A.allocateCustom(A_data, 12);
	Arange(A,  1);

    Conv2D<qint32_t> layer{ 1, 3, { 1, 1 }, { 1, 2 }, { 2, 2 }, true, PaddingMode::REFLECT };
    Fill(layer.weights, 1);
    Fill(layer.biases, 1);
    const auto out = layer.forward(A);
    
    qint32_t R_data[] = { 10, 8, 10, 8, 7, 5, 7, 5, 4, 2, 4, 2, 7, 5, 7, 5, 10, 8, 10, 8, 13, 11, 13, 11, 10, 8, 10, 8, 7, 5, 7, 5, 10, 8, 10, 8, 7, 5, 7, 5, 4, 2, 4, 2, 7, 5, 7, 5, 10, 8, 10, 8, 13, 11, 13, 11, 10, 8, 10, 8, 7, 5, 7, 5, 10, 8, 10, 8, 7, 5, 7, 5, 4, 2, 4, 2, 7, 5, 7, 5, 10, 8, 10, 8, 13, 11, 13, 11, 10, 8, 10, 8, 7, 5, 7, 5 };
	Tensor<qint32_t> R{ 1, 3, 8, 4 };
	R.allocateCustom(R_data, 96);

     EML_ASSERT_TENSOR_EQUALS(R, out);
}
inline int testConv2DCase57()
{
    qint32_t A_data[36]{} ;
	Tensor<qint32_t> A{ 3, 4, 3 };
	A.allocateCustom(A_data, 36);
	Arange(A,  1);

    Conv2D<qint32_t> layer{ 3, 1, { 2, 1 }, { 1, 2 }, { 0, 0 }, false, PaddingMode::CIRCULAR };
    Fill(layer.weights, 1);
    
    const auto out = layer.forward(A);
    
    qint32_t R_data[] = { 87, 99, 105, 117, 123, 135 };
	Tensor<qint32_t> R{ 1, 1, 3, 2 };
	R.allocateCustom(R_data, 6);

     EML_ASSERT_TENSOR_EQUALS(R, out);
}
inline int testConv2DCase58()
{
    qint32_t A_data[12]{} ;
	Tensor<qint32_t> A{ 1, 4, 3 };
	A.allocateCustom(A_data, 12);
	Arange(A,  1);

    Conv2D<qint32_t> layer{ 1, 3, { 1, 1 }, { 2, 1 }, { 2, 2 }, true, PaddingMode::ZEROS };
    Fill(layer.weights, 1);
    Fill(layer.biases, 1);
    const auto out = layer.forward(A);
    
    qint32_t R_data[] = { 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 3, 4, 1, 1, 1, 1, 8, 9, 10, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 3, 4, 1, 1, 1, 1, 8, 9, 10, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 3, 4, 1, 1, 1, 1, 8, 9, 10, 1, 1, 1, 1, 1, 1, 1, 1, 1 };
	Tensor<qint32_t> R{ 1, 3, 4, 7 };
	R.allocateCustom(R_data, 84);

     EML_ASSERT_TENSOR_EQUALS(R, out);
}
inline int testConv2DCase59()
{
    qint32_t A_data[16]{} ;
	Tensor<qint32_t> A{ 1, 4, 4 };
	A.allocateCustom(A_data, 16);
	Arange(A,  1);

    Conv2D<qint32_t> layer{ 1, 3, { 1, 2 }, { 1, 2 }, { 0, 0 }, false, PaddingMode::ZEROS };
    Fill(layer.weights, 1);
    
    const auto out = layer.forward(A);
    
    qint32_t R_data[] = { 3, 7, 11, 15, 19, 23, 27, 31, 3, 7, 11, 15, 19, 23, 27, 31, 3, 7, 11, 15, 19, 23, 27, 31 };
	Tensor<qint32_t> R{ 1, 3, 4, 2 };
	R.allocateCustom(R_data, 24);

     EML_ASSERT_TENSOR_EQUALS(R, out);
}
inline int testConv2DCase60()
{
    qint32_t A_data[12]{} ;
	Tensor<qint32_t> A{ 1, 4, 3 };
	A.allocateCustom(A_data, 12);
	Arange(A,  1);

    Conv2D<qint32_t> layer{ 1, 1, { 1, 2 }, { 2, 1 }, { 1, 0 }, true, PaddingMode::CIRCULAR };
    Fill(layer.weights, 1);
    Fill(layer.biases, 1);
    const auto out = layer.forward(A);
    
    qint32_t R_data[] = { 22, 24, 10, 12, 22, 24 };
	Tensor<qint32_t> R{ 1, 1, 3, 2 };
	R.allocateCustom(R_data, 6);

     EML_ASSERT_TENSOR_EQUALS(R, out);
}
inline int testConv2DCase61()
{
    qint32_t A_data[12]{} ;
	Tensor<qint32_t> A{ 1, 3, 4 };
	A.allocateCustom(A_data, 12);
	Arange(A,  1);

    Conv2D<qint32_t> layer{ 1, 1, { 2, 1 }, { 1, 2 }, { 1, 0 }, false, PaddingMode::CIRCULAR };
    Fill(layer.weights, 1);
    
    const auto out = layer.forward(A);
    
    qint32_t R_data[] = { 10, 14, 6, 10, 14, 18, 10, 14 };
	Tensor<qint32_t> R{ 1, 1, 4, 2 };
	R.allocateCustom(R_data, 8);

     EML_ASSERT_TENSOR_EQUALS(R, out);
}
inline int testConv2DCase62()
{
    qint32_t A_data[12]{} ;
	Tensor<qint32_t> A{ 1, 4, 3 };
	A.allocateCustom(A_data, 12);
	Arange(A,  1);

    Conv2D<qint32_t> layer{ 1, 1, { 1, 1 }, { 2, 1 }, { 1, 0 }, true, PaddingMode::CIRCULAR };
    Fill(layer.weights, 1);
    Fill(layer.biases, 1);
    const auto out = layer.forward(A);
    
    qint32_t R_data[] = { 11, 12, 13, 5, 6, 7, 11, 12, 13 };
	Tensor<qint32_t> R{ 1, 1, 3, 3 };
	R.allocateCustom(R_data, 9);

     EML_ASSERT_TENSOR_EQUALS(R, out);
}
inline int testConv2DCase63()
{
    qint32_t A_data[12]{} ;
	Tensor<qint32_t> A{ 1, 3, 4 };
	A.allocateCustom(A_data, 12);
	Arange(A,  1);

    Conv2D<qint32_t> layer{ 1, 3, { 1, 2 }, { 1, 2 }, { 1, 0 }, false, PaddingMode::CIRCULAR };
    Fill(layer.weights, 1);
    
    const auto out = layer.forward(A);
    
    qint32_t R_data[] = { 19, 23, 3, 7, 11, 15, 19, 23, 3, 7, 19, 23, 3, 7, 11, 15, 19, 23, 3, 7, 19, 23, 3, 7, 11, 15, 19, 23, 3, 7 };
	Tensor<qint32_t> R{ 1, 3, 5, 2 };
	R.allocateCustom(R_data, 30);

     EML_ASSERT_TENSOR_EQUALS(R, out);
}
inline int testConv2DCase64()
{
    qint32_t A_data[12]{} ;
	Tensor<qint32_t> A{ 1, 3, 4 };
	A.allocateCustom(A_data, 12);
	Arange(A,  1);

    Conv2D<qint32_t> layer{ 1, 3, { 1, 2 }, { 1, 2 }, { 1, 0 }, true, PaddingMode::REFLECT };
    Fill(layer.weights, 1);
    Fill(layer.biases, 1);
    const auto out = layer.forward(A);
    
    qint32_t R_data[] = { 12, 16, 4, 8, 12, 16, 20, 24, 12, 16, 12, 16, 4, 8, 12, 16, 20, 24, 12, 16, 12, 16, 4, 8, 12, 16, 20, 24, 12, 16 };
	Tensor<qint32_t> R{ 1, 3, 5, 2 };
	R.allocateCustom(R_data, 30);

     EML_ASSERT_TENSOR_EQUALS(R, out);
}
inline int testConv2DCase65()
{
    qint32_t A_data[48]{} ;
	Tensor<qint32_t> A{ 3, 4, 4 };
	A.allocateCustom(A_data, 48);
	Arange(A,  1);

    Conv2D<qint32_t> layer{ 3, 1, { 2, 1 }, { 1, 2 }, { 1, 0 }, false, PaddingMode::REPLICATE };
    Fill(layer.weights, 1);
    
    const auto out = layer.forward(A);
    
    qint32_t R_data[] = { 102, 114, 114, 126, 138, 150, 162, 174, 174, 186 };
	Tensor<qint32_t> R{ 1, 1, 5, 2 };
	R.allocateCustom(R_data, 10);

     EML_ASSERT_TENSOR_EQUALS(R, out);
}
inline int testConv2DCase66()
{
    qint32_t A_data[36]{} ;
	Tensor<qint32_t> A{ 3, 3, 4 };
	A.allocateCustom(A_data, 36);
	Arange(A,  1);

    Conv2D<qint32_t> layer{ 3, 1, { 1, 1 }, { 2, 1 }, { 1, 0 }, true, PaddingMode::CIRCULAR };
    Fill(layer.weights, 1);
    Fill(layer.biases, 1);
    const auto out = layer.forward(A);
    
    qint32_t R_data[] = { 64, 67, 70, 73, 52, 55, 58, 61, 40, 43, 46, 49 };
	Tensor<qint32_t> R{ 1, 1, 3, 4 };
	R.allocateCustom(R_data, 12);

     EML_ASSERT_TENSOR_EQUALS(R, out);
}
inline int testConv2DCase67()
{
    qint32_t A_data[36]{} ;
	Tensor<qint32_t> A{ 3, 4, 3 };
	A.allocateCustom(A_data, 36);
	Arange(A,  1);

    Conv2D<qint32_t> layer{ 3, 1, { 1, 1 }, { 1, 2 }, { 1, 0 }, false, PaddingMode::REPLICATE };
    Fill(layer.weights, 1);
    
    const auto out = layer.forward(A);
    
    qint32_t R_data[] = { 39, 45, 39, 45, 48, 54, 57, 63, 66, 72, 66, 72 };
	Tensor<qint32_t> R{ 1, 1, 6, 2 };
	R.allocateCustom(R_data, 12);

     EML_ASSERT_TENSOR_EQUALS(R, out);
}
inline int testConv2DCase68()
{
    qint32_t A_data[12]{} ;
	Tensor<qint32_t> A{ 1, 4, 3 };
	A.allocateCustom(A_data, 12);
	Arange(A,  1);

    Conv2D<qint32_t> layer{ 1, 1, { 1, 2 }, { 1, 2 }, { 0, 1 }, true, PaddingMode::REPLICATE };
    Fill(layer.weights, 1);
    Fill(layer.biases, 1);
    const auto out = layer.forward(A);
    
    qint32_t R_data[] = { 3, 6, 9, 12, 15, 18, 21, 24 };
	Tensor<qint32_t> R{ 1, 1, 4, 2 };
	R.allocateCustom(R_data, 8);

     EML_ASSERT_TENSOR_EQUALS(R, out);
}
inline int testConv2DCase69()
{
    qint32_t A_data[16]{} ;
	Tensor<qint32_t> A{ 1, 4, 4 };
	A.allocateCustom(A_data, 16);
	Arange(A,  1);

    Conv2D<qint32_t> layer{ 1, 3, { 1, 1 }, { 1, 2 }, { 0, 1 }, false, PaddingMode::ZEROS };
    Fill(layer.weights, 1);
    
    const auto out = layer.forward(A);
    
    qint32_t R_data[] = { 0, 2, 4, 0, 6, 8, 0, 10, 12, 0, 14, 16, 0, 2, 4, 0, 6, 8, 0, 10, 12, 0, 14, 16, 0, 2, 4, 0, 6, 8, 0, 10, 12, 0, 14, 16 };
	Tensor<qint32_t> R{ 1, 3, 4, 3 };
	R.allocateCustom(R_data, 36);

     EML_ASSERT_TENSOR_EQUALS(R, out);
}
inline int testConv2DCase70()
{
    qint32_t A_data[16]{} ;
	Tensor<qint32_t> A{ 1, 4, 4 };
	A.allocateCustom(A_data, 16);
	Arange(A,  1);

    Conv2D<qint32_t> layer{ 1, 1, { 1, 2 }, { 1, 1 }, { 2, 2 }, true, PaddingMode::ZEROS };
    Fill(layer.weights, 1);
    Fill(layer.biases, 1);
    const auto out = layer.forward(A);
    
    qint32_t R_data[] = { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 4, 6, 8, 5, 1, 1, 6, 12, 14, 16, 9, 1, 1, 10, 20, 22, 24, 13, 1, 1, 14, 28, 30, 32, 17, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 };
	Tensor<qint32_t> R{ 1, 1, 8, 7 };
	R.allocateCustom(R_data, 56);

     EML_ASSERT_TENSOR_EQUALS(R, out);
}
inline int testConv2DCase71()
{
    qint32_t A_data[16]{} ;
	Tensor<qint32_t> A{ 1, 4, 4 };
	A.allocateCustom(A_data, 16);
	Arange(A,  1);

    Conv2D<qint32_t> layer{ 1, 1, { 1, 2 }, { 1, 2 }, { 0, 0 }, false, PaddingMode::REFLECT };
    Fill(layer.weights, 1);
    
    const auto out = layer.forward(A);
    
    qint32_t R_data[] = { 3, 7, 11, 15, 19, 23, 27, 31 };
	Tensor<qint32_t> R{ 1, 1, 4, 2 };
	R.allocateCustom(R_data, 8);

     EML_ASSERT_TENSOR_EQUALS(R, out);
}
inline int testConv2DCase72()
{
    qint32_t A_data[12]{} ;
	Tensor<qint32_t> A{ 1, 4, 3 };
	A.allocateCustom(A_data, 12);
	Arange(A,  1);

    Conv2D<qint32_t> layer{ 1, 3, { 1, 2 }, { 1, 1 }, { 0, 0 }, true, PaddingMode::REFLECT };
    Fill(layer.weights, 1);
    Fill(layer.biases, 1);
    const auto out = layer.forward(A);
    
    qint32_t R_data[] = { 4, 6, 10, 12, 16, 18, 22, 24, 4, 6, 10, 12, 16, 18, 22, 24, 4, 6, 10, 12, 16, 18, 22, 24 };
	Tensor<qint32_t> R{ 1, 3, 4, 2 };
	R.allocateCustom(R_data, 24);

     EML_ASSERT_TENSOR_EQUALS(R, out);
}
inline int testConv2DCase73()
{
    qint32_t A_data[16]{} ;
	Tensor<qint32_t> A{ 1, 4, 4 };
	A.allocateCustom(A_data, 16);
	Arange(A,  1);

    Conv2D<qint32_t> layer{ 1, 1, { 1, 1 }, { 1, 1 }, { 2, 2 }, false, PaddingMode::REFLECT };
    Fill(layer.weights, 1);
    
    const auto out = layer.forward(A);
    
    qint32_t R_data[] = { 11, 10, 9, 10, 11, 12, 11, 10, 7, 6, 5, 6, 7, 8, 7, 6, 3, 2, 1, 2, 3, 4, 3, 2, 7, 6, 5, 6, 7, 8, 7, 6, 11, 10, 9, 10, 11, 12, 11, 10, 15, 14, 13, 14, 15, 16, 15, 14, 11, 10, 9, 10, 11, 12, 11, 10, 7, 6, 5, 6, 7, 8, 7, 6 };
	Tensor<qint32_t> R{ 1, 1, 8, 8 };
	R.allocateCustom(R_data, 64);

     EML_ASSERT_TENSOR_EQUALS(R, out);
}
inline int testConv2DCase74()
{
    qint32_t A_data[12]{} ;
	Tensor<qint32_t> A{ 1, 3, 4 };
	A.allocateCustom(A_data, 12);
	Arange(A,  1);

    Conv2D<qint32_t> layer{ 1, 1, { 1, 1 }, { 1, 2 }, { 0, 1 }, true, PaddingMode::REPLICATE };
    Fill(layer.weights, 1);
    Fill(layer.biases, 1);
    const auto out = layer.forward(A);
    
    qint32_t R_data[] = { 2, 3, 5, 6, 7, 9, 10, 11, 13 };
	Tensor<qint32_t> R{ 1, 1, 3, 3 };
	R.allocateCustom(R_data, 9);

     EML_ASSERT_TENSOR_EQUALS(R, out);
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
    