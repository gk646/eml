
#ifndef EML_TEST_NN_LAYERS_CONV2D_H
#define EML_TEST_NN_LAYERS_CONV2D_H

inline int testConv2DCase0()
{
    qint32_t A_data[12]{} ;
	Tensor<qint32_t> A{ 1, 3, 4 };
	A.allocateCustom(A_data, 12);
	arange(A,  1);

    Conv2D<qint32_t> layer{ 1, 1, { 1, 1 }, { 1, 1 }, { 2, 2 }, true, PaddingMode::CIRCULAR };
    fill(layer.weights, 1);
    fill(layer.biases, 1);
    const auto out = layer.forward(A);
    
    qint32_t R_data[] = { 8, 9, 6, 7, 8, 9, 6, 7, 12, 13, 10, 11, 12, 13, 10, 11, 4, 5, 2, 3, 4, 5, 2, 3, 8, 9, 6, 7, 8, 9, 6, 7, 12, 13, 10, 11, 12, 13, 10, 11, 4, 5, 2, 3, 4, 5, 2, 3, 8, 9, 6, 7, 8, 9, 6, 7 };
	Tensor<qint32_t> R{ 1, 1, 7, 8 };
	R.allocateCustom(R_data, 56);

     EML_ASSERT_TENSOR_EQUALS(R, out);
}
inline int testConv2DCase1()
{
    qint32_t A_data[12]{} ;
	Tensor<qint32_t> A{ 1, 4, 3 };
	A.allocateCustom(A_data, 12);
	arange(A,  1);

    Conv2D<qint32_t> layer{ 1, 3, { 1, 1 }, { 1, 2 }, { 0, 1 }, false, PaddingMode::REPLICATE };
    fill(layer.weights, 1);
    
    const auto out = layer.forward(A);
    
    qint32_t R_data[] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 };
	Tensor<qint32_t> R{ 1, 3, 4, 3 };
	R.allocateCustom(R_data, 36);

     EML_ASSERT_TENSOR_EQUALS(R, out);
}
inline int testConv2DCase2()
{
    qint32_t A_data[16]{} ;
	Tensor<qint32_t> A{ 1, 4, 4 };
	A.allocateCustom(A_data, 16);
	arange(A,  1);

    Conv2D<qint32_t> layer{ 1, 3, { 1, 1 }, { 2, 1 }, { 0, 1 }, true, PaddingMode::REPLICATE };
    fill(layer.weights, 1);
    fill(layer.biases, 1);
    const auto out = layer.forward(A);
    
    qint32_t R_data[] = { 2, 2, 3, 4, 5, 5, 10, 10, 11, 12, 13, 13, 2, 2, 3, 4, 5, 5, 10, 10, 11, 12, 13, 13, 2, 2, 3, 4, 5, 5, 10, 10, 11, 12, 13, 13 };
	Tensor<qint32_t> R{ 1, 3, 2, 6 };
	R.allocateCustom(R_data, 36);

     EML_ASSERT_TENSOR_EQUALS(R, out);
}
inline int testConv2DCase3()
{
    qint32_t A_data[12]{} ;
	Tensor<qint32_t> A{ 1, 4, 3 };
	A.allocateCustom(A_data, 12);
	arange(A,  1);

    Conv2D<qint32_t> layer{ 1, 3, { 1, 1 }, { 2, 1 }, { 2, 2 }, false, PaddingMode::CIRCULAR };
    fill(layer.weights, 1);
    
    const auto out = layer.forward(A);
    
    qint32_t R_data[] = { 8, 9, 7, 8, 9, 7, 8, 2, 3, 1, 2, 3, 1, 2, 8, 9, 7, 8, 9, 7, 8, 2, 3, 1, 2, 3, 1, 2, 8, 9, 7, 8, 9, 7, 8, 2, 3, 1, 2, 3, 1, 2, 8, 9, 7, 8, 9, 7, 8, 2, 3, 1, 2, 3, 1, 2, 8, 9, 7, 8, 9, 7, 8, 2, 3, 1, 2, 3, 1, 2, 8, 9, 7, 8, 9, 7, 8, 2, 3, 1, 2, 3, 1, 2 };
	Tensor<qint32_t> R{ 1, 3, 4, 7 };
	R.allocateCustom(R_data, 84);

     EML_ASSERT_TENSOR_EQUALS(R, out);
}
inline int testConv2DCase4()
{
    qint32_t A_data[16]{} ;
	Tensor<qint32_t> A{ 1, 4, 4 };
	A.allocateCustom(A_data, 16);
	arange(A,  1);

    Conv2D<qint32_t> layer{ 1, 1, { 1, 1 }, { 2, 1 }, { 2, 2 }, true, PaddingMode::REFLECT };
    fill(layer.weights, 1);
    fill(layer.biases, 1);
    const auto out = layer.forward(A);
    
    qint32_t R_data[] = { 12, 11, 10, 11, 12, 13, 12, 11, 4, 3, 2, 3, 4, 5, 4, 3, 12, 11, 10, 11, 12, 13, 12, 11, 12, 11, 10, 11, 12, 13, 12, 11 };
	Tensor<qint32_t> R{ 1, 1, 4, 8 };
	R.allocateCustom(R_data, 32);

     EML_ASSERT_TENSOR_EQUALS(R, out);
}
inline int testConv2DCase5()
{
    qint32_t A_data[36]{} ;
	Tensor<qint32_t> A{ 3, 4, 3 };
	A.allocateCustom(A_data, 36);
	arange(A,  1);

    Conv2D<qint32_t> layer{ 3, 1, { 2, 1 }, { 2, 1 }, { 2, 2 }, false, PaddingMode::REPLICATE };
    fill(layer.weights, 1);
    
    const auto out = layer.forward(A);
    
    qint32_t R_data[] = { 78, 78, 78, 84, 90, 90, 90, 87, 87, 87, 93, 99, 99, 99, 123, 123, 123, 129, 135, 135, 135, 132, 132, 132, 138, 144, 144, 144 };
	Tensor<qint32_t> R{ 1, 1, 4, 7 };
	R.allocateCustom(R_data, 28);

     EML_ASSERT_TENSOR_EQUALS(R, out);
}
inline int testConv2DCase6()
{
    qint32_t A_data[16]{} ;
	Tensor<qint32_t> A{ 1, 4, 4 };
	A.allocateCustom(A_data, 16);
	arange(A,  1);

    Conv2D<qint32_t> layer{ 1, 3, { 1, 2 }, { 2, 1 }, { 1, 0 }, true, PaddingMode::REPLICATE };
    fill(layer.weights, 1);
    fill(layer.biases, 1);
    const auto out = layer.forward(A);
    
    qint32_t R_data[] = { 4, 6, 8, 12, 14, 16, 28, 30, 32, 4, 6, 8, 12, 14, 16, 28, 30, 32, 4, 6, 8, 12, 14, 16, 28, 30, 32 };
	Tensor<qint32_t> R{ 1, 3, 3, 3 };
	R.allocateCustom(R_data, 27);

     EML_ASSERT_TENSOR_EQUALS(R, out);
}
inline int testConv2DCase7()
{
    qint32_t A_data[48]{} ;
	Tensor<qint32_t> A{ 3, 4, 4 };
	A.allocateCustom(A_data, 48);
	arange(A,  1);

    Conv2D<qint32_t> layer{ 3, 1, { 2, 1 }, { 2, 1 }, { 2, 2 }, false, PaddingMode::CIRCULAR };
    fill(layer.weights, 1);
    
    const auto out = layer.forward(A);
    
    qint32_t R_data[] = { 174, 180, 162, 168, 174, 180, 162, 168, 126, 132, 114, 120, 126, 132, 114, 120, 174, 180, 162, 168, 174, 180, 162, 168, 126, 132, 114, 120, 126, 132, 114, 120 };
	Tensor<qint32_t> R{ 1, 1, 4, 8 };
	R.allocateCustom(R_data, 32);

     EML_ASSERT_TENSOR_EQUALS(R, out);
}
inline int testConv2DCase8()
{
    qint32_t A_data[16]{} ;
	Tensor<qint32_t> A{ 1, 4, 4 };
	A.allocateCustom(A_data, 16);
	arange(A,  1);

    Conv2D<qint32_t> layer{ 1, 3, { 1, 1 }, { 1, 1 }, { 0, 0 }, true, PaddingMode::REPLICATE };
    fill(layer.weights, 1);
    fill(layer.biases, 1);
    const auto out = layer.forward(A);
    
    qint32_t R_data[] = { 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17 };
	Tensor<qint32_t> R{ 1, 3, 4, 4 };
	R.allocateCustom(R_data, 48);

     EML_ASSERT_TENSOR_EQUALS(R, out);
}
inline int testConv2DCase9()
{
    qint32_t A_data[36]{} ;
	Tensor<qint32_t> A{ 3, 3, 4 };
	A.allocateCustom(A_data, 36);
	arange(A,  1);

    Conv2D<qint32_t> layer{ 3, 1, { 1, 2 }, { 1, 2 }, { 2, 2 }, false, PaddingMode::REPLICATE };
    fill(layer.weights, 1);
    
    const auto out = layer.forward(A);
    
    qint32_t R_data[] = { 78, 81, 93, 96, 78, 81, 93, 96, 78, 81, 93, 96, 102, 105, 117, 120, 126, 129, 141, 144, 126, 129, 141, 144, 126, 129, 141, 144 };
	Tensor<qint32_t> R{ 1, 1, 7, 4 };
	R.allocateCustom(R_data, 28);

     EML_ASSERT_TENSOR_EQUALS(R, out);
}
inline int testConv2DCase10()
{
    qint32_t A_data[12]{} ;
	Tensor<qint32_t> A{ 1, 4, 3 };
	A.allocateCustom(A_data, 12);
	arange(A,  1);

    Conv2D<qint32_t> layer{ 1, 1, { 2, 1 }, { 1, 2 }, { 0, 0 }, true, PaddingMode::REFLECT };
    fill(layer.weights, 1);
    fill(layer.biases, 1);
    const auto out = layer.forward(A);
    
    qint32_t R_data[] = { 6, 10, 12, 16, 18, 22 };
	Tensor<qint32_t> R{ 1, 1, 3, 2 };
	R.allocateCustom(R_data, 6);

     EML_ASSERT_TENSOR_EQUALS(R, out);
}
inline int testConv2DCase11()
{
    qint32_t A_data[16]{} ;
	Tensor<qint32_t> A{ 1, 4, 4 };
	A.allocateCustom(A_data, 16);
	arange(A,  1);

    Conv2D<qint32_t> layer{ 1, 1, { 1, 2 }, { 1, 1 }, { 0, 0 }, false, PaddingMode::CIRCULAR };
    fill(layer.weights, 1);
    
    const auto out = layer.forward(A);
    
    qint32_t R_data[] = { 3, 5, 7, 11, 13, 15, 19, 21, 23, 27, 29, 31 };
	Tensor<qint32_t> R{ 1, 1, 4, 3 };
	R.allocateCustom(R_data, 12);

     EML_ASSERT_TENSOR_EQUALS(R, out);
}
inline int testConv2DCase12()
{
    qint32_t A_data[12]{} ;
	Tensor<qint32_t> A{ 1, 3, 4 };
	A.allocateCustom(A_data, 12);
	arange(A,  1);

    Conv2D<qint32_t> layer{ 1, 1, { 1, 2 }, { 2, 1 }, { 1, 0 }, true, PaddingMode::CIRCULAR };
    fill(layer.weights, 1);
    fill(layer.biases, 1);
    const auto out = layer.forward(A);
    
    qint32_t R_data[] = { 20, 22, 24, 12, 14, 16, 4, 6, 8 };
	Tensor<qint32_t> R{ 1, 1, 3, 3 };
	R.allocateCustom(R_data, 9);

     EML_ASSERT_TENSOR_EQUALS(R, out);
}
inline int testConv2DCase13()
{
    qint32_t A_data[36]{} ;
	Tensor<qint32_t> A{ 3, 3, 4 };
	A.allocateCustom(A_data, 36);
	arange(A,  1);

    Conv2D<qint32_t> layer{ 3, 1, { 2, 1 }, { 1, 2 }, { 2, 2 }, false, PaddingMode::REPLICATE };
    fill(layer.weights, 1);
    
    const auto out = layer.forward(A);
    
    qint32_t R_data[] = { 78, 78, 90, 96, 78, 78, 90, 96, 90, 90, 102, 108, 114, 114, 126, 132, 126, 126, 138, 144, 126, 126, 138, 144 };
	Tensor<qint32_t> R{ 1, 1, 6, 4 };
	R.allocateCustom(R_data, 24);

     EML_ASSERT_TENSOR_EQUALS(R, out);
}
inline int testConv2DCase14()
{
    qint32_t A_data[16]{} ;
	Tensor<qint32_t> A{ 1, 4, 4 };
	A.allocateCustom(A_data, 16);
	arange(A,  1);

    Conv2D<qint32_t> layer{ 1, 3, { 1, 1 }, { 1, 1 }, { 2, 2 }, true, PaddingMode::CIRCULAR };
    fill(layer.weights, 1);
    fill(layer.biases, 1);
    const auto out = layer.forward(A);
    
    qint32_t R_data[] = { 12, 13, 10, 11, 12, 13, 10, 11, 16, 17, 14, 15, 16, 17, 14, 15, 4, 5, 2, 3, 4, 5, 2, 3, 8, 9, 6, 7, 8, 9, 6, 7, 12, 13, 10, 11, 12, 13, 10, 11, 16, 17, 14, 15, 16, 17, 14, 15, 4, 5, 2, 3, 4, 5, 2, 3, 8, 9, 6, 7, 8, 9, 6, 7, 12, 13, 10, 11, 12, 13, 10, 11, 16, 17, 14, 15, 16, 17, 14, 15, 4, 5, 2, 3, 4, 5, 2, 3, 8, 9, 6, 7, 8, 9, 6, 7, 12, 13, 10, 11, 12, 13, 10, 11, 16, 17, 14, 15, 16, 17, 14, 15, 4, 5, 2, 3, 4, 5, 2, 3, 8, 9, 6, 7, 8, 9, 6, 7, 12, 13, 10, 11, 12, 13, 10, 11, 16, 17, 14, 15, 16, 17, 14, 15, 4, 5, 2, 3, 4, 5, 2, 3, 8, 9, 6, 7, 8, 9, 6, 7, 12, 13, 10, 11, 12, 13, 10, 11, 16, 17, 14, 15, 16, 17, 14, 15, 4, 5, 2, 3, 4, 5, 2, 3, 8, 9, 6, 7, 8, 9, 6, 7 };
	Tensor<qint32_t> R{ 1, 3, 8, 8 };
	R.allocateCustom(R_data, 192);

     EML_ASSERT_TENSOR_EQUALS(R, out);
}
inline int testConv2DCase15()
{
    qint32_t A_data[12]{} ;
	Tensor<qint32_t> A{ 1, 4, 3 };
	A.allocateCustom(A_data, 12);
	arange(A,  1);

    Conv2D<qint32_t> layer{ 1, 3, { 1, 1 }, { 1, 2 }, { 0, 0 }, false, PaddingMode::CIRCULAR };
    fill(layer.weights, 1);
    
    const auto out = layer.forward(A);
    
    qint32_t R_data[] = { 1, 3, 4, 6, 7, 9, 10, 12, 1, 3, 4, 6, 7, 9, 10, 12, 1, 3, 4, 6, 7, 9, 10, 12 };
	Tensor<qint32_t> R{ 1, 3, 4, 2 };
	R.allocateCustom(R_data, 24);

     EML_ASSERT_TENSOR_EQUALS(R, out);
}
inline int testConv2DCase16()
{
    qint32_t A_data[12]{} ;
	Tensor<qint32_t> A{ 1, 4, 3 };
	A.allocateCustom(A_data, 12);
	arange(A,  1);

    Conv2D<qint32_t> layer{ 1, 3, { 2, 1 }, { 2, 1 }, { 0, 1 }, true, PaddingMode::ZEROS };
    fill(layer.weights, 1);
    fill(layer.biases, 1);
    const auto out = layer.forward(A);
    
    qint32_t R_data[] = { 1, 6, 8, 10, 1, 1, 18, 20, 22, 1, 1, 6, 8, 10, 1, 1, 18, 20, 22, 1, 1, 6, 8, 10, 1, 1, 18, 20, 22, 1 };
	Tensor<qint32_t> R{ 1, 3, 2, 5 };
	R.allocateCustom(R_data, 30);

     EML_ASSERT_TENSOR_EQUALS(R, out);
}
inline int testConv2DCase17()
{
    qint32_t A_data[12]{} ;
	Tensor<qint32_t> A{ 1, 4, 3 };
	A.allocateCustom(A_data, 12);
	arange(A,  1);

    Conv2D<qint32_t> layer{ 1, 3, { 1, 2 }, { 1, 1 }, { 0, 1 }, false, PaddingMode::REFLECT };
    fill(layer.weights, 1);
    
    const auto out = layer.forward(A);
    
    qint32_t R_data[] = { 3, 3, 5, 5, 9, 9, 11, 11, 15, 15, 17, 17, 21, 21, 23, 23, 3, 3, 5, 5, 9, 9, 11, 11, 15, 15, 17, 17, 21, 21, 23, 23, 3, 3, 5, 5, 9, 9, 11, 11, 15, 15, 17, 17, 21, 21, 23, 23 };
	Tensor<qint32_t> R{ 1, 3, 4, 4 };
	R.allocateCustom(R_data, 48);

     EML_ASSERT_TENSOR_EQUALS(R, out);
}
inline int testConv2DCase18()
{
    qint32_t A_data[12]{} ;
	Tensor<qint32_t> A{ 1, 3, 4 };
	A.allocateCustom(A_data, 12);
	arange(A,  1);

    Conv2D<qint32_t> layer{ 1, 1, { 2, 1 }, { 1, 1 }, { 2, 2 }, true, PaddingMode::ZEROS };
    fill(layer.weights, 1);
    fill(layer.biases, 1);
    const auto out = layer.forward(A);
    
    qint32_t R_data[] = { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 3, 4, 5, 1, 1, 1, 1, 7, 9, 11, 13, 1, 1, 1, 1, 15, 17, 19, 21, 1, 1, 1, 1, 10, 11, 12, 13, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 };
	Tensor<qint32_t> R{ 1, 1, 6, 8 };
	R.allocateCustom(R_data, 48);

     EML_ASSERT_TENSOR_EQUALS(R, out);
}
inline int testConv2DCase19()
{
    qint32_t A_data[36]{} ;
	Tensor<qint32_t> A{ 3, 4, 3 };
	A.allocateCustom(A_data, 36);
	arange(A,  1);

    Conv2D<qint32_t> layer{ 3, 1, { 1, 2 }, { 1, 2 }, { 0, 0 }, false, PaddingMode::REPLICATE };
    fill(layer.weights, 1);
    
    const auto out = layer.forward(A);
    
    qint32_t R_data[] = { 81, 99, 117, 135 };
	Tensor<qint32_t> R{ 1, 1, 4, 1 };
	R.allocateCustom(R_data, 4);

     EML_ASSERT_TENSOR_EQUALS(R, out);
}
inline int testConv2DCase20()
{
    qint32_t A_data[36]{} ;
	Tensor<qint32_t> A{ 3, 3, 4 };
	A.allocateCustom(A_data, 36);
	arange(A,  1);

    Conv2D<qint32_t> layer{ 3, 1, { 2, 1 }, { 1, 1 }, { 1, 0 }, true, PaddingMode::REFLECT };
    fill(layer.weights, 1);
    fill(layer.biases, 1);
    const auto out = layer.forward(A);
    
    qint32_t R_data[] = { 91, 97, 103, 109, 91, 97, 103, 109, 115, 121, 127, 133, 115, 121, 127, 133 };
	Tensor<qint32_t> R{ 1, 1, 4, 4 };
	R.allocateCustom(R_data, 16);

     EML_ASSERT_TENSOR_EQUALS(R, out);
}
inline int testConv2DCase21()
{
    qint32_t A_data[12]{} ;
	Tensor<qint32_t> A{ 1, 3, 4 };
	A.allocateCustom(A_data, 12);
	arange(A,  1);

    Conv2D<qint32_t> layer{ 1, 1, { 1, 1 }, { 2, 1 }, { 2, 2 }, false, PaddingMode::ZEROS };
    fill(layer.weights, 1);
    
    const auto out = layer.forward(A);
    
    qint32_t R_data[] = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 0, 0, 0, 0, 9, 10, 11, 12, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
	Tensor<qint32_t> R{ 1, 1, 4, 8 };
	R.allocateCustom(R_data, 32);

     EML_ASSERT_TENSOR_EQUALS(R, out);
}
inline int testConv2DCase22()
{
    qint32_t A_data[16]{} ;
	Tensor<qint32_t> A{ 1, 4, 4 };
	A.allocateCustom(A_data, 16);
	arange(A,  1);

    Conv2D<qint32_t> layer{ 1, 3, { 1, 1 }, { 1, 2 }, { 1, 0 }, true, PaddingMode::ZEROS };
    fill(layer.weights, 1);
    fill(layer.biases, 1);
    const auto out = layer.forward(A);
    
    qint32_t R_data[] = { 1, 1, 2, 4, 6, 8, 10, 12, 14, 16, 1, 1, 1, 1, 2, 4, 6, 8, 10, 12, 14, 16, 1, 1, 1, 1, 2, 4, 6, 8, 10, 12, 14, 16, 1, 1 };
	Tensor<qint32_t> R{ 1, 3, 6, 2 };
	R.allocateCustom(R_data, 36);

     EML_ASSERT_TENSOR_EQUALS(R, out);
}
inline int testConv2DCase23()
{
    qint32_t A_data[16]{} ;
	Tensor<qint32_t> A{ 1, 4, 4 };
	A.allocateCustom(A_data, 16);
	arange(A,  1);

    Conv2D<qint32_t> layer{ 1, 3, { 1, 1 }, { 1, 2 }, { 0, 1 }, false, PaddingMode::ZEROS };
    fill(layer.weights, 1);
    
    const auto out = layer.forward(A);
    
    qint32_t R_data[] = { 0, 2, 4, 0, 6, 8, 0, 10, 12, 0, 14, 16, 0, 2, 4, 0, 6, 8, 0, 10, 12, 0, 14, 16, 0, 2, 4, 0, 6, 8, 0, 10, 12, 0, 14, 16 };
	Tensor<qint32_t> R{ 1, 3, 4, 3 };
	R.allocateCustom(R_data, 36);

     EML_ASSERT_TENSOR_EQUALS(R, out);
}
inline int testConv2DCase24()
{
    qint32_t A_data[36]{} ;
	Tensor<qint32_t> A{ 3, 3, 4 };
	A.allocateCustom(A_data, 36);
	arange(A,  1);

    Conv2D<qint32_t> layer{ 3, 1, { 1, 2 }, { 2, 1 }, { 0, 1 }, true, PaddingMode::ZEROS };
    fill(layer.weights, 1);
    fill(layer.biases, 1);
    const auto out = layer.forward(A);
    
    qint32_t R_data[] = { 40, 82, 88, 94, 49, 64, 130, 136, 142, 73 };
	Tensor<qint32_t> R{ 1, 1, 2, 5 };
	R.allocateCustom(R_data, 10);

     EML_ASSERT_TENSOR_EQUALS(R, out);
}
inline int testConv2DCase25()
{
    qint32_t A_data[12]{} ;
	Tensor<qint32_t> A{ 1, 3, 4 };
	A.allocateCustom(A_data, 12);
	arange(A,  1);

    Conv2D<qint32_t> layer{ 1, 1, { 2, 1 }, { 2, 1 }, { 0, 0 }, false, PaddingMode::REFLECT };
    fill(layer.weights, 1);
    
    const auto out = layer.forward(A);
    
    qint32_t R_data[] = { 6, 8, 10, 12 };
	Tensor<qint32_t> R{ 1, 1, 1, 4 };
	R.allocateCustom(R_data, 4);

     EML_ASSERT_TENSOR_EQUALS(R, out);
}
inline int testConv2DCase26()
{
    qint32_t A_data[12]{} ;
	Tensor<qint32_t> A{ 1, 3, 4 };
	A.allocateCustom(A_data, 12);
	arange(A,  1);

    Conv2D<qint32_t> layer{ 1, 3, { 1, 2 }, { 1, 1 }, { 0, 0 }, true, PaddingMode::CIRCULAR };
    fill(layer.weights, 1);
    fill(layer.biases, 1);
    const auto out = layer.forward(A);
    
    qint32_t R_data[] = { 4, 6, 8, 12, 14, 16, 20, 22, 24, 4, 6, 8, 12, 14, 16, 20, 22, 24, 4, 6, 8, 12, 14, 16, 20, 22, 24 };
	Tensor<qint32_t> R{ 1, 3, 3, 3 };
	R.allocateCustom(R_data, 27);

     EML_ASSERT_TENSOR_EQUALS(R, out);
}
inline int testConv2DCase27()
{
    qint32_t A_data[12]{} ;
	Tensor<qint32_t> A{ 1, 4, 3 };
	A.allocateCustom(A_data, 12);
	arange(A,  1);

    Conv2D<qint32_t> layer{ 1, 3, { 1, 2 }, { 1, 1 }, { 1, 0 }, false, PaddingMode::REFLECT };
    fill(layer.weights, 1);
    
    const auto out = layer.forward(A);
    
    qint32_t R_data[] = { 9, 11, 3, 5, 9, 11, 15, 17, 21, 23, 15, 17, 9, 11, 3, 5, 9, 11, 15, 17, 21, 23, 15, 17, 9, 11, 3, 5, 9, 11, 15, 17, 21, 23, 15, 17 };
	Tensor<qint32_t> R{ 1, 3, 6, 2 };
	R.allocateCustom(R_data, 36);

     EML_ASSERT_TENSOR_EQUALS(R, out);
}
inline int testConv2DCase28()
{
    qint32_t A_data[16]{} ;
	Tensor<qint32_t> A{ 1, 4, 4 };
	A.allocateCustom(A_data, 16);
	arange(A,  1);

    Conv2D<qint32_t> layer{ 1, 3, { 1, 1 }, { 1, 2 }, { 0, 1 }, true, PaddingMode::ZEROS };
    fill(layer.weights, 1);
    fill(layer.biases, 1);
    const auto out = layer.forward(A);
    
    qint32_t R_data[] = { 1, 3, 5, 1, 7, 9, 1, 11, 13, 1, 15, 17, 1, 3, 5, 1, 7, 9, 1, 11, 13, 1, 15, 17, 1, 3, 5, 1, 7, 9, 1, 11, 13, 1, 15, 17 };
	Tensor<qint32_t> R{ 1, 3, 4, 3 };
	R.allocateCustom(R_data, 36);

     EML_ASSERT_TENSOR_EQUALS(R, out);
}
inline int testConv2DCase29()
{
    qint32_t A_data[16]{} ;
	Tensor<qint32_t> A{ 1, 4, 4 };
	A.allocateCustom(A_data, 16);
	arange(A,  1);

    Conv2D<qint32_t> layer{ 1, 3, { 1, 1 }, { 2, 1 }, { 1, 0 }, false, PaddingMode::CIRCULAR };
    fill(layer.weights, 1);
    
    const auto out = layer.forward(A);
    
    qint32_t R_data[] = { 13, 14, 15, 16, 5, 6, 7, 8, 13, 14, 15, 16, 13, 14, 15, 16, 5, 6, 7, 8, 13, 14, 15, 16, 13, 14, 15, 16, 5, 6, 7, 8, 13, 14, 15, 16 };
	Tensor<qint32_t> R{ 1, 3, 3, 4 };
	R.allocateCustom(R_data, 36);

     EML_ASSERT_TENSOR_EQUALS(R, out);
}
inline int testConv2DCase30()
{
    qint32_t A_data[48]{} ;
	Tensor<qint32_t> A{ 3, 4, 4 };
	A.allocateCustom(A_data, 48);
	arange(A,  1);

    Conv2D<qint32_t> layer{ 3, 1, { 2, 1 }, { 1, 1 }, { 0, 0 }, true, PaddingMode::REFLECT };
    fill(layer.weights, 1);
    fill(layer.biases, 1);
    const auto out = layer.forward(A);
    
    qint32_t R_data[] = { 115, 121, 127, 133, 139, 145, 151, 157, 163, 169, 175, 181 };
	Tensor<qint32_t> R{ 1, 1, 3, 4 };
	R.allocateCustom(R_data, 12);

     EML_ASSERT_TENSOR_EQUALS(R, out);
}
inline int testConv2DCase31()
{
    qint32_t A_data[36]{} ;
	Tensor<qint32_t> A{ 3, 3, 4 };
	A.allocateCustom(A_data, 36);
	arange(A,  1);

    Conv2D<qint32_t> layer{ 3, 1, { 1, 2 }, { 1, 1 }, { 1, 0 }, false, PaddingMode::REFLECT };
    fill(layer.weights, 1);
    
    const auto out = layer.forward(A);
    
    qint32_t R_data[] = { 105, 111, 117, 81, 87, 93, 105, 111, 117, 129, 135, 141, 105, 111, 117 };
	Tensor<qint32_t> R{ 1, 1, 5, 3 };
	R.allocateCustom(R_data, 15);

     EML_ASSERT_TENSOR_EQUALS(R, out);
}
inline int testConv2DCase32()
{
    qint32_t A_data[12]{} ;
	Tensor<qint32_t> A{ 1, 3, 4 };
	A.allocateCustom(A_data, 12);
	arange(A,  1);

    Conv2D<qint32_t> layer{ 1, 3, { 1, 1 }, { 1, 2 }, { 2, 2 }, true, PaddingMode::REFLECT };
    fill(layer.weights, 1);
    fill(layer.biases, 1);
    const auto out = layer.forward(A);
    
    qint32_t R_data[] = { 12, 10, 12, 12, 8, 6, 8, 8, 4, 2, 4, 4, 8, 6, 8, 8, 12, 10, 12, 12, 8, 6, 8, 8, 4, 2, 4, 4, 12, 10, 12, 12, 8, 6, 8, 8, 4, 2, 4, 4, 8, 6, 8, 8, 12, 10, 12, 12, 8, 6, 8, 8, 4, 2, 4, 4, 12, 10, 12, 12, 8, 6, 8, 8, 4, 2, 4, 4, 8, 6, 8, 8, 12, 10, 12, 12, 8, 6, 8, 8, 4, 2, 4, 4 };
	Tensor<qint32_t> R{ 1, 3, 7, 4 };
	R.allocateCustom(R_data, 84);

     EML_ASSERT_TENSOR_EQUALS(R, out);
}
inline int testConv2DCase33()
{
    qint32_t A_data[12]{} ;
	Tensor<qint32_t> A{ 1, 3, 4 };
	A.allocateCustom(A_data, 12);
	arange(A,  1);

    Conv2D<qint32_t> layer{ 1, 3, { 1, 1 }, { 2, 1 }, { 1, 0 }, false, PaddingMode::ZEROS };
    fill(layer.weights, 1);
    
    const auto out = layer.forward(A);
    
    qint32_t R_data[] = { 0, 0, 0, 0, 5, 6, 7, 8, 0, 0, 0, 0, 0, 0, 0, 0, 5, 6, 7, 8, 0, 0, 0, 0, 0, 0, 0, 0, 5, 6, 7, 8, 0, 0, 0, 0 };
	Tensor<qint32_t> R{ 1, 3, 3, 4 };
	R.allocateCustom(R_data, 36);

     EML_ASSERT_TENSOR_EQUALS(R, out);
}
inline int testConv2DCase34()
{
    qint32_t A_data[48]{} ;
	Tensor<qint32_t> A{ 3, 4, 4 };
	A.allocateCustom(A_data, 48);
	arange(A,  1);

    Conv2D<qint32_t> layer{ 3, 1, { 1, 2 }, { 1, 1 }, { 0, 0 }, true, PaddingMode::REPLICATE };
    fill(layer.weights, 1);
    fill(layer.biases, 1);
    const auto out = layer.forward(A);
    
    qint32_t R_data[] = { 106, 112, 118, 130, 136, 142, 154, 160, 166, 178, 184, 190 };
	Tensor<qint32_t> R{ 1, 1, 4, 3 };
	R.allocateCustom(R_data, 12);

     EML_ASSERT_TENSOR_EQUALS(R, out);
}
inline int testConv2DCase35()
{
    qint32_t A_data[12]{} ;
	Tensor<qint32_t> A{ 1, 3, 4 };
	A.allocateCustom(A_data, 12);
	arange(A,  1);

    Conv2D<qint32_t> layer{ 1, 1, { 1, 2 }, { 1, 2 }, { 0, 0 }, false, PaddingMode::ZEROS };
    fill(layer.weights, 1);
    
    const auto out = layer.forward(A);
    
    qint32_t R_data[] = { 3, 7, 11, 15, 19, 23 };
	Tensor<qint32_t> R{ 1, 1, 3, 2 };
	R.allocateCustom(R_data, 6);

     EML_ASSERT_TENSOR_EQUALS(R, out);
}
inline int testConv2DCase36()
{
    qint32_t A_data[48]{} ;
	Tensor<qint32_t> A{ 3, 4, 4 };
	A.allocateCustom(A_data, 48);
	arange(A,  1);

    Conv2D<qint32_t> layer{ 3, 1, { 1, 2 }, { 1, 2 }, { 0, 1 }, true, PaddingMode::REPLICATE };
    fill(layer.weights, 1);
    fill(layer.biases, 1);
    const auto out = layer.forward(A);
    
    qint32_t R_data[] = { 103, 112, 121, 127, 136, 145, 151, 160, 169, 175, 184, 193 };
	Tensor<qint32_t> R{ 1, 1, 4, 3 };
	R.allocateCustom(R_data, 12);

     EML_ASSERT_TENSOR_EQUALS(R, out);
}
inline int testConv2DCase37()
{
    qint32_t A_data[36]{} ;
	Tensor<qint32_t> A{ 3, 3, 4 };
	A.allocateCustom(A_data, 36);
	arange(A,  1);

    Conv2D<qint32_t> layer{ 3, 1, { 1, 2 }, { 1, 1 }, { 1, 0 }, false, PaddingMode::REFLECT };
    fill(layer.weights, 1);
    
    const auto out = layer.forward(A);
    
    qint32_t R_data[] = { 105, 111, 117, 81, 87, 93, 105, 111, 117, 129, 135, 141, 105, 111, 117 };
	Tensor<qint32_t> R{ 1, 1, 5, 3 };
	R.allocateCustom(R_data, 15);

     EML_ASSERT_TENSOR_EQUALS(R, out);
}
inline int testConv2DCase38()
{
    qint32_t A_data[36]{} ;
	Tensor<qint32_t> A{ 3, 4, 3 };
	A.allocateCustom(A_data, 36);
	arange(A,  1);

    Conv2D<qint32_t> layer{ 3, 1, { 1, 1 }, { 2, 1 }, { 1, 0 }, true, PaddingMode::REPLICATE };
    fill(layer.weights, 1);
    fill(layer.biases, 1);
    const auto out = layer.forward(A);
    
    qint32_t R_data[] = { 40, 43, 46, 49, 52, 55, 67, 70, 73 };
	Tensor<qint32_t> R{ 1, 1, 3, 3 };
	R.allocateCustom(R_data, 9);

     EML_ASSERT_TENSOR_EQUALS(R, out);
}
inline int testConv2DCase39()
{
    qint32_t A_data[12]{} ;
	Tensor<qint32_t> A{ 1, 3, 4 };
	A.allocateCustom(A_data, 12);
	arange(A,  1);

    Conv2D<qint32_t> layer{ 1, 3, { 1, 2 }, { 2, 1 }, { 1, 0 }, false, PaddingMode::ZEROS };
    fill(layer.weights, 1);
    
    const auto out = layer.forward(A);
    
    qint32_t R_data[] = { 0, 0, 0, 11, 13, 15, 0, 0, 0, 0, 0, 0, 11, 13, 15, 0, 0, 0, 0, 0, 0, 11, 13, 15, 0, 0, 0 };
	Tensor<qint32_t> R{ 1, 3, 3, 3 };
	R.allocateCustom(R_data, 27);

     EML_ASSERT_TENSOR_EQUALS(R, out);
}
inline int testConv2DCase40()
{
    qint32_t A_data[12]{} ;
	Tensor<qint32_t> A{ 1, 3, 4 };
	A.allocateCustom(A_data, 12);
	arange(A,  1);

    Conv2D<qint32_t> layer{ 1, 1, { 1, 1 }, { 1, 2 }, { 0, 0 }, true, PaddingMode::CIRCULAR };
    fill(layer.weights, 1);
    fill(layer.biases, 1);
    const auto out = layer.forward(A);
    
    qint32_t R_data[] = { 2, 4, 6, 8, 10, 12 };
	Tensor<qint32_t> R{ 1, 1, 3, 2 };
	R.allocateCustom(R_data, 6);

     EML_ASSERT_TENSOR_EQUALS(R, out);
}
inline int testConv2DCase41()
{
    qint32_t A_data[12]{} ;
	Tensor<qint32_t> A{ 1, 4, 3 };
	A.allocateCustom(A_data, 12);
	arange(A,  1);

    Conv2D<qint32_t> layer{ 1, 3, { 1, 1 }, { 1, 1 }, { 0, 1 }, false, PaddingMode::REFLECT };
    fill(layer.weights, 1);
    
    const auto out = layer.forward(A);
    
    qint32_t R_data[] = { 2, 1, 2, 3, 2, 5, 4, 5, 6, 5, 8, 7, 8, 9, 8, 11, 10, 11, 12, 11, 2, 1, 2, 3, 2, 5, 4, 5, 6, 5, 8, 7, 8, 9, 8, 11, 10, 11, 12, 11, 2, 1, 2, 3, 2, 5, 4, 5, 6, 5, 8, 7, 8, 9, 8, 11, 10, 11, 12, 11 };
	Tensor<qint32_t> R{ 1, 3, 4, 5 };
	R.allocateCustom(R_data, 60);

     EML_ASSERT_TENSOR_EQUALS(R, out);
}
inline int testConv2DCase42()
{
    qint32_t A_data[36]{} ;
	Tensor<qint32_t> A{ 3, 4, 3 };
	A.allocateCustom(A_data, 36);
	arange(A,  1);

    Conv2D<qint32_t> layer{ 3, 1, { 2, 1 }, { 2, 1 }, { 0, 0 }, true, PaddingMode::ZEROS };
    fill(layer.weights, 1);
    fill(layer.biases, 1);
    const auto out = layer.forward(A);
    
    qint32_t R_data[] = { 88, 94, 100, 124, 130, 136 };
	Tensor<qint32_t> R{ 1, 1, 2, 3 };
	R.allocateCustom(R_data, 6);

     EML_ASSERT_TENSOR_EQUALS(R, out);
}
inline int testConv2DCase43()
{
    qint32_t A_data[12]{} ;
	Tensor<qint32_t> A{ 1, 4, 3 };
	A.allocateCustom(A_data, 12);
	arange(A,  1);

    Conv2D<qint32_t> layer{ 1, 1, { 1, 1 }, { 1, 1 }, { 2, 2 }, false, PaddingMode::REFLECT };
    fill(layer.weights, 1);
    
    const auto out = layer.forward(A);
    
    qint32_t R_data[] = { 9, 8, 7, 8, 9, 8, 7, 6, 5, 4, 5, 6, 5, 4, 3, 2, 1, 2, 3, 2, 1, 6, 5, 4, 5, 6, 5, 4, 9, 8, 7, 8, 9, 8, 7, 12, 11, 10, 11, 12, 11, 10, 9, 8, 7, 8, 9, 8, 7, 6, 5, 4, 5, 6, 5, 4 };
	Tensor<qint32_t> R{ 1, 1, 8, 7 };
	R.allocateCustom(R_data, 56);

     EML_ASSERT_TENSOR_EQUALS(R, out);
}
inline int testConv2DCase44()
{
    qint32_t A_data[16]{} ;
	Tensor<qint32_t> A{ 1, 4, 4 };
	A.allocateCustom(A_data, 16);
	arange(A,  1);

    Conv2D<qint32_t> layer{ 1, 1, { 2, 1 }, { 2, 1 }, { 0, 0 }, true, PaddingMode::REPLICATE };
    fill(layer.weights, 1);
    fill(layer.biases, 1);
    const auto out = layer.forward(A);
    
    qint32_t R_data[] = { 7, 9, 11, 13, 23, 25, 27, 29 };
	Tensor<qint32_t> R{ 1, 1, 2, 4 };
	R.allocateCustom(R_data, 8);

     EML_ASSERT_TENSOR_EQUALS(R, out);
}
inline int testConv2DCase45()
{
    qint32_t A_data[16]{} ;
	Tensor<qint32_t> A{ 1, 4, 4 };
	A.allocateCustom(A_data, 16);
	arange(A,  1);

    Conv2D<qint32_t> layer{ 1, 3, { 1, 1 }, { 2, 1 }, { 1, 0 }, false, PaddingMode::CIRCULAR };
    fill(layer.weights, 1);
    
    const auto out = layer.forward(A);
    
    qint32_t R_data[] = { 13, 14, 15, 16, 5, 6, 7, 8, 13, 14, 15, 16, 13, 14, 15, 16, 5, 6, 7, 8, 13, 14, 15, 16, 13, 14, 15, 16, 5, 6, 7, 8, 13, 14, 15, 16 };
	Tensor<qint32_t> R{ 1, 3, 3, 4 };
	R.allocateCustom(R_data, 36);

     EML_ASSERT_TENSOR_EQUALS(R, out);
}
inline int testConv2DCase46()
{
    qint32_t A_data[36]{} ;
	Tensor<qint32_t> A{ 3, 3, 4 };
	A.allocateCustom(A_data, 36);
	arange(A,  1);

    Conv2D<qint32_t> layer{ 3, 1, { 1, 2 }, { 1, 1 }, { 2, 2 }, true, PaddingMode::ZEROS };
    fill(layer.weights, 1);
    fill(layer.biases, 1);
    const auto out = layer.forward(A);
    
    qint32_t R_data[] = { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 40, 82, 88, 94, 49, 1, 1, 52, 106, 112, 118, 61, 1, 1, 64, 130, 136, 142, 73, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 };
	Tensor<qint32_t> R{ 1, 1, 7, 7 };
	R.allocateCustom(R_data, 49);

     EML_ASSERT_TENSOR_EQUALS(R, out);
}
inline int testConv2DCase47()
{
    qint32_t A_data[16]{} ;
	Tensor<qint32_t> A{ 1, 4, 4 };
	A.allocateCustom(A_data, 16);
	arange(A,  1);

    Conv2D<qint32_t> layer{ 1, 3, { 1, 2 }, { 1, 2 }, { 2, 2 }, false, PaddingMode::REPLICATE };
    fill(layer.weights, 1);
    
    const auto out = layer.forward(A);
    
    qint32_t R_data[] = { 2, 3, 7, 8, 2, 3, 7, 8, 2, 3, 7, 8, 10, 11, 15, 16, 18, 19, 23, 24, 26, 27, 31, 32, 26, 27, 31, 32, 26, 27, 31, 32, 2, 3, 7, 8, 2, 3, 7, 8, 2, 3, 7, 8, 10, 11, 15, 16, 18, 19, 23, 24, 26, 27, 31, 32, 26, 27, 31, 32, 26, 27, 31, 32, 2, 3, 7, 8, 2, 3, 7, 8, 2, 3, 7, 8, 10, 11, 15, 16, 18, 19, 23, 24, 26, 27, 31, 32, 26, 27, 31, 32, 26, 27, 31, 32 };
	Tensor<qint32_t> R{ 1, 3, 8, 4 };
	R.allocateCustom(R_data, 96);

     EML_ASSERT_TENSOR_EQUALS(R, out);
}
inline int testConv2DCase48()
{
    qint32_t A_data[16]{} ;
	Tensor<qint32_t> A{ 1, 4, 4 };
	A.allocateCustom(A_data, 16);
	arange(A,  1);

    Conv2D<qint32_t> layer{ 1, 1, { 2, 1 }, { 1, 1 }, { 1, 0 }, true, PaddingMode::REPLICATE };
    fill(layer.weights, 1);
    fill(layer.biases, 1);
    const auto out = layer.forward(A);
    
    qint32_t R_data[] = { 3, 5, 7, 9, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 27, 29, 31, 33 };
	Tensor<qint32_t> R{ 1, 1, 5, 4 };
	R.allocateCustom(R_data, 20);

     EML_ASSERT_TENSOR_EQUALS(R, out);
}
inline int testConv2DCase49()
{
    qint32_t A_data[12]{} ;
	Tensor<qint32_t> A{ 1, 4, 3 };
	A.allocateCustom(A_data, 12);
	arange(A,  1);

    Conv2D<qint32_t> layer{ 1, 1, { 2, 1 }, { 1, 2 }, { 1, 0 }, false, PaddingMode::ZEROS };
    fill(layer.weights, 1);
    
    const auto out = layer.forward(A);
    
    qint32_t R_data[] = { 1, 3, 5, 9, 11, 15, 17, 21, 10, 12 };
	Tensor<qint32_t> R{ 1, 1, 5, 2 };
	R.allocateCustom(R_data, 10);

     EML_ASSERT_TENSOR_EQUALS(R, out);
}
inline int testConv2DCase50()
{
    qint32_t A_data[12]{} ;
	Tensor<qint32_t> A{ 1, 4, 3 };
	A.allocateCustom(A_data, 12);
	arange(A,  1);

    Conv2D<qint32_t> layer{ 1, 1, { 2, 1 }, { 2, 1 }, { 0, 0 }, true, PaddingMode::REPLICATE };
    fill(layer.weights, 1);
    fill(layer.biases, 1);
    const auto out = layer.forward(A);
    
    qint32_t R_data[] = { 6, 8, 10, 18, 20, 22 };
	Tensor<qint32_t> R{ 1, 1, 2, 3 };
	R.allocateCustom(R_data, 6);

     EML_ASSERT_TENSOR_EQUALS(R, out);
}
inline int testConv2DCase51()
{
    qint32_t A_data[12]{} ;
	Tensor<qint32_t> A{ 1, 4, 3 };
	A.allocateCustom(A_data, 12);
	arange(A,  1);

    Conv2D<qint32_t> layer{ 1, 1, { 2, 1 }, { 1, 1 }, { 0, 0 }, false, PaddingMode::ZEROS };
    fill(layer.weights, 1);
    
    const auto out = layer.forward(A);
    
    qint32_t R_data[] = { 5, 7, 9, 11, 13, 15, 17, 19, 21 };
	Tensor<qint32_t> R{ 1, 1, 3, 3 };
	R.allocateCustom(R_data, 9);

     EML_ASSERT_TENSOR_EQUALS(R, out);
}
inline int testConv2DCase52()
{
    qint32_t A_data[16]{} ;
	Tensor<qint32_t> A{ 1, 4, 4 };
	A.allocateCustom(A_data, 16);
	arange(A,  1);

    Conv2D<qint32_t> layer{ 1, 3, { 1, 2 }, { 1, 1 }, { 1, 0 }, true, PaddingMode::REFLECT };
    fill(layer.weights, 1);
    fill(layer.biases, 1);
    const auto out = layer.forward(A);
    
    qint32_t R_data[] = { 12, 14, 16, 4, 6, 8, 12, 14, 16, 20, 22, 24, 28, 30, 32, 20, 22, 24, 12, 14, 16, 4, 6, 8, 12, 14, 16, 20, 22, 24, 28, 30, 32, 20, 22, 24, 12, 14, 16, 4, 6, 8, 12, 14, 16, 20, 22, 24, 28, 30, 32, 20, 22, 24 };
	Tensor<qint32_t> R{ 1, 3, 6, 3 };
	R.allocateCustom(R_data, 54);

     EML_ASSERT_TENSOR_EQUALS(R, out);
}
inline int testConv2DCase53()
{
    qint32_t A_data[36]{} ;
	Tensor<qint32_t> A{ 3, 4, 3 };
	A.allocateCustom(A_data, 36);
	arange(A,  1);

    Conv2D<qint32_t> layer{ 3, 1, { 1, 2 }, { 1, 2 }, { 2, 2 }, false, PaddingMode::CIRCULAR };
    fill(layer.weights, 1);
    
    const auto out = layer.forward(A);
    
    qint32_t R_data[] = { 123, 117, 120, 141, 135, 138, 87, 81, 84, 105, 99, 102, 123, 117, 120, 141, 135, 138, 87, 81, 84, 105, 99, 102 };
	Tensor<qint32_t> R{ 1, 1, 8, 3 };
	R.allocateCustom(R_data, 24);

     EML_ASSERT_TENSOR_EQUALS(R, out);
}
inline int testConv2DCase54()
{
    qint32_t A_data[12]{} ;
	Tensor<qint32_t> A{ 1, 3, 4 };
	A.allocateCustom(A_data, 12);
	arange(A,  1);

    Conv2D<qint32_t> layer{ 1, 1, { 2, 1 }, { 2, 1 }, { 2, 2 }, true, PaddingMode::REFLECT };
    fill(layer.weights, 1);
    fill(layer.biases, 1);
    const auto out = layer.forward(A);
    
    qint32_t R_data[] = { 19, 17, 15, 17, 19, 21, 19, 17, 11, 9, 7, 9, 11, 13, 11, 9, 19, 17, 15, 17, 19, 21, 19, 17 };
	Tensor<qint32_t> R{ 1, 1, 3, 8 };
	R.allocateCustom(R_data, 24);

     EML_ASSERT_TENSOR_EQUALS(R, out);
}
inline int testConv2DCase55()
{
    qint32_t A_data[12]{} ;
	Tensor<qint32_t> A{ 1, 3, 4 };
	A.allocateCustom(A_data, 12);
	arange(A,  1);

    Conv2D<qint32_t> layer{ 1, 3, { 2, 1 }, { 1, 2 }, { 1, 0 }, false, PaddingMode::CIRCULAR };
    fill(layer.weights, 1);
    
    const auto out = layer.forward(A);
    
    qint32_t R_data[] = { 10, 14, 6, 10, 14, 18, 10, 14, 10, 14, 6, 10, 14, 18, 10, 14, 10, 14, 6, 10, 14, 18, 10, 14 };
	Tensor<qint32_t> R{ 1, 3, 4, 2 };
	R.allocateCustom(R_data, 24);

     EML_ASSERT_TENSOR_EQUALS(R, out);
}
inline int testConv2DCase56()
{
    qint32_t A_data[48]{} ;
	Tensor<qint32_t> A{ 3, 4, 4 };
	A.allocateCustom(A_data, 48);
	arange(A,  1);

    Conv2D<qint32_t> layer{ 3, 1, { 2, 1 }, { 1, 2 }, { 2, 2 }, true, PaddingMode::CIRCULAR };
    fill(layer.weights, 1);
    fill(layer.biases, 1);
    const auto out = layer.forward(A);
    
    qint32_t R_data[] = { 175, 163, 175, 163, 151, 139, 151, 139, 127, 115, 127, 115, 151, 139, 151, 139, 175, 163, 175, 163, 151, 139, 151, 139, 127, 115, 127, 115 };
	Tensor<qint32_t> R{ 1, 1, 7, 4 };
	R.allocateCustom(R_data, 28);

     EML_ASSERT_TENSOR_EQUALS(R, out);
}
inline int testConv2DCase57()
{
    qint32_t A_data[48]{} ;
	Tensor<qint32_t> A{ 3, 4, 4 };
	A.allocateCustom(A_data, 48);
	arange(A,  1);

    Conv2D<qint32_t> layer{ 3, 1, { 1, 1 }, { 1, 2 }, { 1, 0 }, false, PaddingMode::REFLECT };
    fill(layer.weights, 1);
    
    const auto out = layer.forward(A);
    
    qint32_t R_data[] = { 63, 69, 51, 57, 63, 69, 75, 81, 87, 93, 75, 81 };
	Tensor<qint32_t> R{ 1, 1, 6, 2 };
	R.allocateCustom(R_data, 12);

     EML_ASSERT_TENSOR_EQUALS(R, out);
}
inline int testConv2DCase58()
{
    qint32_t A_data[48]{} ;
	Tensor<qint32_t> A{ 3, 4, 4 };
	A.allocateCustom(A_data, 48);
	arange(A,  1);

    Conv2D<qint32_t> layer{ 3, 1, { 1, 1 }, { 1, 2 }, { 2, 2 }, true, PaddingMode::ZEROS };
    fill(layer.weights, 1);
    fill(layer.biases, 1);
    const auto out = layer.forward(A);
    
    qint32_t R_data[] = { 1, 1, 1, 1, 1, 1, 1, 1, 1, 52, 58, 1, 1, 64, 70, 1, 1, 76, 82, 1, 1, 88, 94, 1, 1, 1, 1, 1, 1, 1, 1, 1 };
	Tensor<qint32_t> R{ 1, 1, 8, 4 };
	R.allocateCustom(R_data, 32);

     EML_ASSERT_TENSOR_EQUALS(R, out);
}
inline int testConv2DCase59()
{
    qint32_t A_data[36]{} ;
	Tensor<qint32_t> A{ 3, 3, 4 };
	A.allocateCustom(A_data, 36);
	arange(A,  1);

    Conv2D<qint32_t> layer{ 3, 1, { 2, 1 }, { 2, 1 }, { 0, 1 }, false, PaddingMode::REFLECT };
    fill(layer.weights, 1);
    
    const auto out = layer.forward(A);
    
    qint32_t R_data[] = { 96, 90, 96, 102, 108, 102 };
	Tensor<qint32_t> R{ 1, 1, 1, 6 };
	R.allocateCustom(R_data, 6);

     EML_ASSERT_TENSOR_EQUALS(R, out);
}
inline int testConv2DCase60()
{
    qint32_t A_data[48]{} ;
	Tensor<qint32_t> A{ 3, 4, 4 };
	A.allocateCustom(A_data, 48);
	arange(A,  1);

    Conv2D<qint32_t> layer{ 3, 1, { 1, 2 }, { 2, 1 }, { 2, 2 }, true, PaddingMode::REPLICATE };
    fill(layer.weights, 1);
    fill(layer.biases, 1);
    const auto out = layer.forward(A);
    
    qint32_t R_data[] = { 103, 103, 106, 112, 118, 121, 121, 103, 103, 106, 112, 118, 121, 121, 151, 151, 154, 160, 166, 169, 169, 175, 175, 178, 184, 190, 193, 193 };
	Tensor<qint32_t> R{ 1, 1, 4, 7 };
	R.allocateCustom(R_data, 28);

     EML_ASSERT_TENSOR_EQUALS(R, out);
}
inline int testConv2DCase61()
{
    qint32_t A_data[16]{} ;
	Tensor<qint32_t> A{ 1, 4, 4 };
	A.allocateCustom(A_data, 16);
	arange(A,  1);

    Conv2D<qint32_t> layer{ 1, 3, { 1, 2 }, { 1, 1 }, { 0, 0 }, false, PaddingMode::REPLICATE };
    fill(layer.weights, 1);
    
    const auto out = layer.forward(A);
    
    qint32_t R_data[] = { 3, 5, 7, 11, 13, 15, 19, 21, 23, 27, 29, 31, 3, 5, 7, 11, 13, 15, 19, 21, 23, 27, 29, 31, 3, 5, 7, 11, 13, 15, 19, 21, 23, 27, 29, 31 };
	Tensor<qint32_t> R{ 1, 3, 4, 3 };
	R.allocateCustom(R_data, 36);

     EML_ASSERT_TENSOR_EQUALS(R, out);
}
inline int testConv2DCase62()
{
    qint32_t A_data[12]{} ;
	Tensor<qint32_t> A{ 1, 4, 3 };
	A.allocateCustom(A_data, 12);
	arange(A,  1);

    Conv2D<qint32_t> layer{ 1, 1, { 1, 2 }, { 1, 1 }, { 0, 0 }, true, PaddingMode::REFLECT };
    fill(layer.weights, 1);
    fill(layer.biases, 1);
    const auto out = layer.forward(A);
    
    qint32_t R_data[] = { 4, 6, 10, 12, 16, 18, 22, 24 };
	Tensor<qint32_t> R{ 1, 1, 4, 2 };
	R.allocateCustom(R_data, 8);

     EML_ASSERT_TENSOR_EQUALS(R, out);
}
inline int testConv2DCase63()
{
    qint32_t A_data[12]{} ;
	Tensor<qint32_t> A{ 1, 3, 4 };
	A.allocateCustom(A_data, 12);
	arange(A,  1);

    Conv2D<qint32_t> layer{ 1, 1, { 2, 1 }, { 1, 2 }, { 2, 2 }, false, PaddingMode::REFLECT };
    fill(layer.weights, 1);
    
    const auto out = layer.forward(A);
    
    qint32_t R_data[] = { 18, 14, 18, 18, 10, 6, 10, 10, 10, 6, 10, 10, 18, 14, 18, 18, 18, 14, 18, 18, 10, 6, 10, 10 };
	Tensor<qint32_t> R{ 1, 1, 6, 4 };
	R.allocateCustom(R_data, 24);

     EML_ASSERT_TENSOR_EQUALS(R, out);
}
inline int testConv2DCase64()
{
    qint32_t A_data[48]{} ;
	Tensor<qint32_t> A{ 3, 4, 4 };
	A.allocateCustom(A_data, 48);
	arange(A,  1);

    Conv2D<qint32_t> layer{ 3, 1, { 1, 1 }, { 2, 1 }, { 0, 1 }, true, PaddingMode::REFLECT };
    fill(layer.weights, 1);
    fill(layer.biases, 1);
    const auto out = layer.forward(A);
    
    qint32_t R_data[] = { 55, 52, 55, 58, 61, 58, 79, 76, 79, 82, 85, 82 };
	Tensor<qint32_t> R{ 1, 1, 2, 6 };
	R.allocateCustom(R_data, 12);

     EML_ASSERT_TENSOR_EQUALS(R, out);
}
inline int testConv2DCase65()
{
    qint32_t A_data[48]{} ;
	Tensor<qint32_t> A{ 3, 4, 4 };
	A.allocateCustom(A_data, 48);
	arange(A,  1);

    Conv2D<qint32_t> layer{ 3, 1, { 1, 2 }, { 1, 1 }, { 1, 0 }, false, PaddingMode::ZEROS };
    fill(layer.weights, 1);
    
    const auto out = layer.forward(A);
    
    qint32_t R_data[] = { 0, 0, 0, 105, 111, 117, 129, 135, 141, 153, 159, 165, 177, 183, 189, 0, 0, 0 };
	Tensor<qint32_t> R{ 1, 1, 6, 3 };
	R.allocateCustom(R_data, 18);

     EML_ASSERT_TENSOR_EQUALS(R, out);
}
inline int testConv2DCase66()
{
    qint32_t A_data[12]{} ;
	Tensor<qint32_t> A{ 1, 4, 3 };
	A.allocateCustom(A_data, 12);
	arange(A,  1);

    Conv2D<qint32_t> layer{ 1, 1, { 1, 1 }, { 2, 1 }, { 0, 1 }, true, PaddingMode::ZEROS };
    fill(layer.weights, 1);
    fill(layer.biases, 1);
    const auto out = layer.forward(A);
    
    qint32_t R_data[] = { 1, 2, 3, 4, 1, 1, 8, 9, 10, 1 };
	Tensor<qint32_t> R{ 1, 1, 2, 5 };
	R.allocateCustom(R_data, 10);

     EML_ASSERT_TENSOR_EQUALS(R, out);
}
inline int testConv2DCase67()
{
    qint32_t A_data[12]{} ;
	Tensor<qint32_t> A{ 1, 4, 3 };
	A.allocateCustom(A_data, 12);
	arange(A,  1);

    Conv2D<qint32_t> layer{ 1, 1, { 1, 1 }, { 2, 1 }, { 0, 1 }, false, PaddingMode::CIRCULAR };
    fill(layer.weights, 1);
    
    const auto out = layer.forward(A);
    
    qint32_t R_data[] = { 3, 1, 2, 3, 1, 9, 7, 8, 9, 7 };
	Tensor<qint32_t> R{ 1, 1, 2, 5 };
	R.allocateCustom(R_data, 10);

     EML_ASSERT_TENSOR_EQUALS(R, out);
}
inline int testConv2DCase68()
{
    qint32_t A_data[12]{} ;
	Tensor<qint32_t> A{ 1, 3, 4 };
	A.allocateCustom(A_data, 12);
	arange(A,  1);

    Conv2D<qint32_t> layer{ 1, 1, { 1, 2 }, { 1, 1 }, { 2, 2 }, true, PaddingMode::REPLICATE };
    fill(layer.weights, 1);
    fill(layer.biases, 1);
    const auto out = layer.forward(A);
    
    qint32_t R_data[] = { 3, 3, 4, 6, 8, 9, 9, 3, 3, 4, 6, 8, 9, 9, 3, 3, 4, 6, 8, 9, 9, 11, 11, 12, 14, 16, 17, 17, 19, 19, 20, 22, 24, 25, 25, 19, 19, 20, 22, 24, 25, 25, 19, 19, 20, 22, 24, 25, 25 };
	Tensor<qint32_t> R{ 1, 1, 7, 7 };
	R.allocateCustom(R_data, 49);

     EML_ASSERT_TENSOR_EQUALS(R, out);
}
inline int testConv2DCase69()
{
    qint32_t A_data[16]{} ;
	Tensor<qint32_t> A{ 1, 4, 4 };
	A.allocateCustom(A_data, 16);
	arange(A,  1);

    Conv2D<qint32_t> layer{ 1, 3, { 1, 1 }, { 1, 1 }, { 0, 0 }, false, PaddingMode::CIRCULAR };
    fill(layer.weights, 1);
    
    const auto out = layer.forward(A);
    
    qint32_t R_data[] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 };
	Tensor<qint32_t> R{ 1, 3, 4, 4 };
	R.allocateCustom(R_data, 48);

     EML_ASSERT_TENSOR_EQUALS(R, out);
}
inline int testConv2DCase70()
{
    qint32_t A_data[12]{} ;
	Tensor<qint32_t> A{ 1, 4, 3 };
	A.allocateCustom(A_data, 12);
	arange(A,  1);

    Conv2D<qint32_t> layer{ 1, 1, { 2, 1 }, { 1, 2 }, { 0, 0 }, true, PaddingMode::REPLICATE };
    fill(layer.weights, 1);
    fill(layer.biases, 1);
    const auto out = layer.forward(A);
    
    qint32_t R_data[] = { 6, 10, 12, 16, 18, 22 };
	Tensor<qint32_t> R{ 1, 1, 3, 2 };
	R.allocateCustom(R_data, 6);

     EML_ASSERT_TENSOR_EQUALS(R, out);
}
inline int testConv2DCase71()
{
    qint32_t A_data[36]{} ;
	Tensor<qint32_t> A{ 3, 4, 3 };
	A.allocateCustom(A_data, 36);
	arange(A,  1);

    Conv2D<qint32_t> layer{ 3, 1, { 1, 1 }, { 2, 1 }, { 2, 2 }, false, PaddingMode::REPLICATE };
    fill(layer.weights, 1);
    
    const auto out = layer.forward(A);
    
    qint32_t R_data[] = { 39, 39, 39, 42, 45, 45, 45, 39, 39, 39, 42, 45, 45, 45, 57, 57, 57, 60, 63, 63, 63, 66, 66, 66, 69, 72, 72, 72 };
	Tensor<qint32_t> R{ 1, 1, 4, 7 };
	R.allocateCustom(R_data, 28);

     EML_ASSERT_TENSOR_EQUALS(R, out);
}
inline int testConv2DCase72()
{
    qint32_t A_data[12]{} ;
	Tensor<qint32_t> A{ 1, 4, 3 };
	A.allocateCustom(A_data, 12);
	arange(A,  1);

    Conv2D<qint32_t> layer{ 1, 1, { 1, 1 }, { 1, 2 }, { 2, 2 }, true, PaddingMode::ZEROS };
    fill(layer.weights, 1);
    fill(layer.biases, 1);
    const auto out = layer.forward(A);
    
    qint32_t R_data[] = { 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 4, 1, 1, 5, 7, 1, 1, 8, 10, 1, 1, 11, 13, 1, 1, 1, 1, 1, 1, 1, 1, 1 };
	Tensor<qint32_t> R{ 1, 1, 8, 4 };
	R.allocateCustom(R_data, 32);

     EML_ASSERT_TENSOR_EQUALS(R, out);
}
inline int testConv2DCase73()
{
    qint32_t A_data[12]{} ;
	Tensor<qint32_t> A{ 1, 4, 3 };
	A.allocateCustom(A_data, 12);
	arange(A,  1);

    Conv2D<qint32_t> layer{ 1, 1, { 2, 1 }, { 1, 2 }, { 0, 1 }, false, PaddingMode::ZEROS };
    fill(layer.weights, 1);
    
    const auto out = layer.forward(A);
    
    qint32_t R_data[] = { 0, 7, 0, 0, 13, 0, 0, 19, 0 };
	Tensor<qint32_t> R{ 1, 1, 3, 3 };
	R.allocateCustom(R_data, 9);

     EML_ASSERT_TENSOR_EQUALS(R, out);
}
inline int testConv2DCase74()
{
    qint32_t A_data[16]{} ;
	Tensor<qint32_t> A{ 1, 4, 4 };
	A.allocateCustom(A_data, 16);
	arange(A,  1);

    Conv2D<qint32_t> layer{ 1, 1, { 1, 2 }, { 1, 2 }, { 0, 1 }, true, PaddingMode::CIRCULAR };
    fill(layer.weights, 1);
    fill(layer.biases, 1);
    const auto out = layer.forward(A);
    
    qint32_t R_data[] = { 6, 6, 6, 14, 14, 14, 22, 22, 22, 30, 30, 30 };
	Tensor<qint32_t> R{ 1, 1, 4, 3 };
	R.allocateCustom(R_data, 12);

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
    