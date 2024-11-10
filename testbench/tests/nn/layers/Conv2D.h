
#ifndef EML_TEST_NN_LAYERS_CONV2D_H
#define EML_TEST_NN_LAYERS_CONV2D_H

inline int testConv2DCase0()
{
    qint32_t A_data[36] ;
	Tensor<qint32_t> A{ 3, 4, 3 };
	A.allocateCustom(A_data, 36);
	Arange(A,1);

    Conv2D<qint32_t> layer{ 3, 1, { 2, 1 }, { 1, 1 }, { 1, 0 }, true, PaddingMode::REPLICATE };
    Fill(layer.weights, 1);
    Fill(layer.biases, 1);
    const auto out = layer.forward(A);
    
    qint32_t R_data[] = { 79, 85, 91, 88, 94, 100, 106, 112, 118, 124, 130, 136, 133, 139, 145 };
	Tensor<qint32_t> R{ 1, 1, 5, 3 };
	R.allocateCustom(R_data, 15);

    return Equals(out, R);
}
inline int testConv2DCase1()
{
    qint32_t A_data[48] ;
	Tensor<qint32_t> A{ 3, 4, 4 };
	A.allocateCustom(A_data, 48);
	Arange(A,1);

    Conv2D<qint32_t> layer{ 3, 1, { 1, 2 }, { 1, 2 }, { 0, 0 }, false, PaddingMode::REPLICATE };
    Fill(layer.weights, 1);
    
    const auto out = layer.forward(A);
    
    qint32_t R_data[] = { 105, 117, 129, 141, 153, 165, 177, 189 };
	Tensor<qint32_t> R{ 1, 1, 4, 2 };
	R.allocateCustom(R_data, 8);

    return Equals(out, R);
}
inline int testConv2DCase2()
{
    qint32_t A_data[12] ;
	Tensor<qint32_t> A{ 1, 4, 3 };
	A.allocateCustom(A_data, 12);
	Arange(A,1);

    Conv2D<qint32_t> layer{ 1, 1, { 1, 2 }, { 1, 2 }, { 2, 2 }, true, PaddingMode::CIRCULAR };
    Fill(layer.weights, 1);
    Fill(layer.biases, 1);
    const auto out = layer.forward(A);
    
    qint32_t R_data[] = { 18, 16, 17, 24, 22, 23, 6, 4, 5, 12, 10, 11, 18, 16, 17, 24, 22, 23, 6, 4, 5, 12, 10, 11 };
	Tensor<qint32_t> R{ 1, 1, 8, 3 };
	R.allocateCustom(R_data, 24);

    return Equals(out, R);
}
inline int testConv2DCase3()
{
    qint32_t A_data[12] ;
	Tensor<qint32_t> A{ 1, 4, 3 };
	A.allocateCustom(A_data, 12);
	Arange(A,1);

    Conv2D<qint32_t> layer{ 1, 3, { 1, 2 }, { 1, 2 }, { 0, 1 }, false, PaddingMode::ZEROS };
    Fill(layer.weights, 1);
    
    const auto out = layer.forward(A);
    
    qint32_t R_data[] = { 1, 5, 4, 11, 7, 17, 10, 23, 1, 5, 4, 11, 7, 17, 10, 23, 1, 5, 4, 11, 7, 17, 10, 23 };
	Tensor<qint32_t> R{ 1, 3, 4, 2 };
	R.allocateCustom(R_data, 24);

    return Equals(out, R);
}
inline int testConv2DCase4()
{
    qint32_t A_data[12] ;
	Tensor<qint32_t> A{ 1, 3, 4 };
	A.allocateCustom(A_data, 12);
	Arange(A,1);

    Conv2D<qint32_t> layer{ 1, 3, { 1, 1 }, { 1, 1 }, { 2, 2 }, true, PaddingMode::ZEROS };
    Fill(layer.weights, 1);
    Fill(layer.biases, 1);
    const auto out = layer.forward(A);
    
    qint32_t R_data[] = { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 3, 4, 5, 1, 1, 1, 1, 6, 7, 8, 9, 1, 1, 1, 1, 10, 11, 12, 13, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 3, 4, 5, 1, 1, 1, 1, 6, 7, 8, 9, 1, 1, 1, 1, 10, 11, 12, 13, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 3, 4, 5, 1, 1, 1, 1, 6, 7, 8, 9, 1, 1, 1, 1, 10, 11, 12, 13, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 };
	Tensor<qint32_t> R{ 1, 3, 7, 8 };
	R.allocateCustom(R_data, 168);

    return Equals(out, R);
}
inline int testConv2DCase5()
{
    qint32_t A_data[16] ;
	Tensor<qint32_t> A{ 1, 4, 4 };
	A.allocateCustom(A_data, 16);
	Arange(A,1);

    Conv2D<qint32_t> layer{ 1, 3, { 1, 1 }, { 1, 1 }, { 0, 1 }, false, PaddingMode::ZEROS };
    Fill(layer.weights, 1);
    
    const auto out = layer.forward(A);
    
    qint32_t R_data[] = { 0, 1, 2, 3, 4, 0, 0, 5, 6, 7, 8, 0, 0, 9, 10, 11, 12, 0, 0, 13, 14, 15, 16, 0, 0, 1, 2, 3, 4, 0, 0, 5, 6, 7, 8, 0, 0, 9, 10, 11, 12, 0, 0, 13, 14, 15, 16, 0, 0, 1, 2, 3, 4, 0, 0, 5, 6, 7, 8, 0, 0, 9, 10, 11, 12, 0, 0, 13, 14, 15, 16, 0 };
	Tensor<qint32_t> R{ 1, 3, 4, 6 };
	R.allocateCustom(R_data, 72);

    return Equals(out, R);
}
inline int testConv2DCase6()
{
    qint32_t A_data[12] ;
	Tensor<qint32_t> A{ 1, 3, 4 };
	A.allocateCustom(A_data, 12);
	Arange(A,1);

    Conv2D<qint32_t> layer{ 1, 3, { 1, 2 }, { 1, 1 }, { 0, 0 }, true, PaddingMode::ZEROS };
    Fill(layer.weights, 1);
    Fill(layer.biases, 1);
    const auto out = layer.forward(A);
    
    qint32_t R_data[] = { 4, 6, 8, 12, 14, 16, 20, 22, 24, 4, 6, 8, 12, 14, 16, 20, 22, 24, 4, 6, 8, 12, 14, 16, 20, 22, 24 };
	Tensor<qint32_t> R{ 1, 3, 3, 3 };
	R.allocateCustom(R_data, 27);

    return Equals(out, R);
}
inline int testConv2DCase7()
{
    qint32_t A_data[36] ;
	Tensor<qint32_t> A{ 3, 4, 3 };
	A.allocateCustom(A_data, 36);
	Arange(A,1);

    Conv2D<qint32_t> layer{ 3, 1, { 1, 1 }, { 2, 1 }, { 1, 0 }, false, PaddingMode::ZEROS };
    Fill(layer.weights, 1);
    
    const auto out = layer.forward(A);
    
    qint32_t R_data[] = { 0, 0, 0, 48, 51, 54, 66, 69, 72 };
	Tensor<qint32_t> R{ 1, 1, 3, 3 };
	R.allocateCustom(R_data, 9);

    return Equals(out, R);
}
inline int testConv2DCase8()
{
    qint32_t A_data[16] ;
	Tensor<qint32_t> A{ 1, 4, 4 };
	A.allocateCustom(A_data, 16);
	Arange(A,1);

    Conv2D<qint32_t> layer{ 1, 1, { 1, 1 }, { 1, 1 }, { 2, 2 }, true, PaddingMode::REPLICATE };
    Fill(layer.weights, 1);
    Fill(layer.biases, 1);
    const auto out = layer.forward(A);
    
    qint32_t R_data[] = { 2, 2, 2, 3, 4, 5, 5, 5, 2, 2, 2, 3, 4, 5, 5, 5, 2, 2, 2, 3, 4, 5, 5, 5, 6, 6, 6, 7, 8, 9, 9, 9, 10, 10, 10, 11, 12, 13, 13, 13, 14, 14, 14, 15, 16, 17, 17, 17, 14, 14, 14, 15, 16, 17, 17, 17, 14, 14, 14, 15, 16, 17, 17, 17 };
	Tensor<qint32_t> R{ 1, 1, 8, 8 };
	R.allocateCustom(R_data, 64);

    return Equals(out, R);
}
inline int testConv2DCase9()
{
    qint32_t A_data[12] ;
	Tensor<qint32_t> A{ 1, 4, 3 };
	A.allocateCustom(A_data, 12);
	Arange(A,1);

    Conv2D<qint32_t> layer{ 1, 3, { 2, 1 }, { 1, 1 }, { 2, 2 }, false, PaddingMode::REFLECT };
    Fill(layer.weights, 1);
    
    const auto out = layer.forward(A);
    
    qint32_t R_data[] = { 15, 13, 11, 13, 15, 13, 11, 9, 7, 5, 7, 9, 7, 5, 9, 7, 5, 7, 9, 7, 5, 15, 13, 11, 13, 15, 13, 11, 21, 19, 17, 19, 21, 19, 17, 21, 19, 17, 19, 21, 19, 17, 15, 13, 11, 13, 15, 13, 11, 15, 13, 11, 13, 15, 13, 11, 9, 7, 5, 7, 9, 7, 5, 9, 7, 5, 7, 9, 7, 5, 15, 13, 11, 13, 15, 13, 11, 21, 19, 17, 19, 21, 19, 17, 21, 19, 17, 19, 21, 19, 17, 15, 13, 11, 13, 15, 13, 11, 15, 13, 11, 13, 15, 13, 11, 9, 7, 5, 7, 9, 7, 5, 9, 7, 5, 7, 9, 7, 5, 15, 13, 11, 13, 15, 13, 11, 21, 19, 17, 19, 21, 19, 17, 21, 19, 17, 19, 21, 19, 17, 15, 13, 11, 13, 15, 13, 11 };
	Tensor<qint32_t> R{ 1, 3, 7, 7 };
	R.allocateCustom(R_data, 147);

    return Equals(out, R);
}
inline int testConv2DCase10()
{
    qint32_t A_data[36] ;
	Tensor<qint32_t> A{ 3, 4, 3 };
	A.allocateCustom(A_data, 36);
	Arange(A,1);

    Conv2D<qint32_t> layer{ 3, 1, { 1, 1 }, { 2, 1 }, { 0, 0 }, true, PaddingMode::ZEROS };
    Fill(layer.weights, 1);
    Fill(layer.biases, 1);
    const auto out = layer.forward(A);
    
    qint32_t R_data[] = { 40, 43, 46, 58, 61, 64 };
	Tensor<qint32_t> R{ 1, 1, 2, 3 };
	R.allocateCustom(R_data, 6);

    return Equals(out, R);
}
inline int testConv2DCase11()
{
    qint32_t A_data[12] ;
	Tensor<qint32_t> A{ 1, 3, 4 };
	A.allocateCustom(A_data, 12);
	Arange(A,1);

    Conv2D<qint32_t> layer{ 1, 1, { 2, 1 }, { 1, 2 }, { 0, 0 }, false, PaddingMode::ZEROS };
    Fill(layer.weights, 1);
    
    const auto out = layer.forward(A);
    
    qint32_t R_data[] = { 6, 10, 14, 18 };
	Tensor<qint32_t> R{ 1, 1, 2, 2 };
	R.allocateCustom(R_data, 4);

    return Equals(out, R);
}
inline int testConv2DCase12()
{
    qint32_t A_data[16] ;
	Tensor<qint32_t> A{ 1, 4, 4 };
	A.allocateCustom(A_data, 16);
	Arange(A,1);

    Conv2D<qint32_t> layer{ 1, 3, { 1, 1 }, { 2, 1 }, { 1, 0 }, true, PaddingMode::ZEROS };
    Fill(layer.weights, 1);
    Fill(layer.biases, 1);
    const auto out = layer.forward(A);
    
    qint32_t R_data[] = { 1, 1, 1, 1, 6, 7, 8, 9, 14, 15, 16, 17, 1, 1, 1, 1, 6, 7, 8, 9, 14, 15, 16, 17, 1, 1, 1, 1, 6, 7, 8, 9, 14, 15, 16, 17 };
	Tensor<qint32_t> R{ 1, 3, 3, 4 };
	R.allocateCustom(R_data, 36);

    return Equals(out, R);
}
inline int testConv2DCase13()
{
    qint32_t A_data[36] ;
	Tensor<qint32_t> A{ 3, 3, 4 };
	A.allocateCustom(A_data, 36);
	Arange(A,1);

    Conv2D<qint32_t> layer{ 3, 1, { 1, 2 }, { 2, 1 }, { 0, 0 }, false, PaddingMode::ZEROS };
    Fill(layer.weights, 1);
    
    const auto out = layer.forward(A);
    
    qint32_t R_data[] = { 81, 87, 93, 129, 135, 141 };
	Tensor<qint32_t> R{ 1, 1, 2, 3 };
	R.allocateCustom(R_data, 6);

    return Equals(out, R);
}
inline int testConv2DCase14()
{
    qint32_t A_data[36] ;
	Tensor<qint32_t> A{ 3, 4, 3 };
	A.allocateCustom(A_data, 36);
	Arange(A,1);

    Conv2D<qint32_t> layer{ 3, 1, { 1, 1 }, { 1, 2 }, { 2, 2 }, true, PaddingMode::ZEROS };
    Fill(layer.weights, 1);
    Fill(layer.biases, 1);
    const auto out = layer.forward(A);
    
    qint32_t R_data[] = { 1, 1, 1, 1, 1, 1, 1, 1, 1, 40, 46, 1, 1, 49, 55, 1, 1, 58, 64, 1, 1, 67, 73, 1, 1, 1, 1, 1, 1, 1, 1, 1 };
	Tensor<qint32_t> R{ 1, 1, 8, 4 };
	R.allocateCustom(R_data, 32);

    return Equals(out, R);
}
inline int testConv2DCase15()
{
    qint32_t A_data[12] ;
	Tensor<qint32_t> A{ 1, 3, 4 };
	A.allocateCustom(A_data, 12);
	Arange(A,1);

    Conv2D<qint32_t> layer{ 1, 3, { 1, 2 }, { 1, 1 }, { 0, 1 }, false, PaddingMode::CIRCULAR };
    Fill(layer.weights, 1);
    
    const auto out = layer.forward(A);
    
    qint32_t R_data[] = { 5, 3, 5, 7, 5, 13, 11, 13, 15, 13, 21, 19, 21, 23, 21, 5, 3, 5, 7, 5, 13, 11, 13, 15, 13, 21, 19, 21, 23, 21, 5, 3, 5, 7, 5, 13, 11, 13, 15, 13, 21, 19, 21, 23, 21 };
	Tensor<qint32_t> R{ 1, 3, 3, 5 };
	R.allocateCustom(R_data, 45);

    return Equals(out, R);
}
inline int testConv2DCase16()
{
    qint32_t A_data[48] ;
	Tensor<qint32_t> A{ 3, 4, 4 };
	A.allocateCustom(A_data, 48);
	Arange(A,1);

    Conv2D<qint32_t> layer{ 3, 1, { 1, 1 }, { 1, 2 }, { 2, 2 }, true, PaddingMode::REPLICATE };
    Fill(layer.weights, 1);
    Fill(layer.biases, 1);
    const auto out = layer.forward(A);
    
    qint32_t R_data[] = { 52, 52, 58, 61, 52, 52, 58, 61, 52, 52, 58, 61, 64, 64, 70, 73, 76, 76, 82, 85, 88, 88, 94, 97, 88, 88, 94, 97, 88, 88, 94, 97 };
	Tensor<qint32_t> R{ 1, 1, 8, 4 };
	R.allocateCustom(R_data, 32);

    return Equals(out, R);
}
inline int testConv2DCase17()
{
    qint32_t A_data[36] ;
	Tensor<qint32_t> A{ 3, 4, 3 };
	A.allocateCustom(A_data, 36);
	Arange(A,1);

    Conv2D<qint32_t> layer{ 3, 1, { 1, 1 }, { 1, 1 }, { 1, 0 }, false, PaddingMode::REFLECT };
    Fill(layer.weights, 1);
    
    const auto out = layer.forward(A);
    
    qint32_t R_data[] = { 48, 51, 54, 39, 42, 45, 48, 51, 54, 57, 60, 63, 66, 69, 72, 57, 60, 63 };
	Tensor<qint32_t> R{ 1, 1, 6, 3 };
	R.allocateCustom(R_data, 18);

    return Equals(out, R);
}
inline int testConv2DCase18()
{
    qint32_t A_data[12] ;
	Tensor<qint32_t> A{ 1, 4, 3 };
	A.allocateCustom(A_data, 12);
	Arange(A,1);

    Conv2D<qint32_t> layer{ 1, 1, { 1, 2 }, { 2, 1 }, { 2, 2 }, true, PaddingMode::REPLICATE };
    Fill(layer.weights, 1);
    Fill(layer.biases, 1);
    const auto out = layer.forward(A);
    
    qint32_t R_data[] = { 3, 3, 4, 6, 7, 7, 3, 3, 4, 6, 7, 7, 15, 15, 16, 18, 19, 19, 21, 21, 22, 24, 25, 25 };
	Tensor<qint32_t> R{ 1, 1, 4, 6 };
	R.allocateCustom(R_data, 24);

    return Equals(out, R);
}
inline int testConv2DCase19()
{
    qint32_t A_data[48] ;
	Tensor<qint32_t> A{ 3, 4, 4 };
	A.allocateCustom(A_data, 48);
	Arange(A,1);

    Conv2D<qint32_t> layer{ 3, 1, { 1, 1 }, { 1, 2 }, { 1, 0 }, false, PaddingMode::REFLECT };
    Fill(layer.weights, 1);
    
    const auto out = layer.forward(A);
    
    qint32_t R_data[] = { 63, 69, 51, 57, 63, 69, 75, 81, 87, 93, 75, 81 };
	Tensor<qint32_t> R{ 1, 1, 6, 2 };
	R.allocateCustom(R_data, 12);

    return Equals(out, R);
}
inline int testConv2DCase20()
{
    qint32_t A_data[12] ;
	Tensor<qint32_t> A{ 1, 4, 3 };
	A.allocateCustom(A_data, 12);
	Arange(A,1);

    Conv2D<qint32_t> layer{ 1, 3, { 1, 2 }, { 1, 1 }, { 0, 1 }, true, PaddingMode::REPLICATE };
    Fill(layer.weights, 1);
    Fill(layer.biases, 1);
    const auto out = layer.forward(A);
    
    qint32_t R_data[] = { 3, 4, 6, 7, 9, 10, 12, 13, 15, 16, 18, 19, 21, 22, 24, 25, 3, 4, 6, 7, 9, 10, 12, 13, 15, 16, 18, 19, 21, 22, 24, 25, 3, 4, 6, 7, 9, 10, 12, 13, 15, 16, 18, 19, 21, 22, 24, 25 };
	Tensor<qint32_t> R{ 1, 3, 4, 4 };
	R.allocateCustom(R_data, 48);

    return Equals(out, R);
}
inline int testConv2DCase21()
{
    qint32_t A_data[12] ;
	Tensor<qint32_t> A{ 1, 4, 3 };
	A.allocateCustom(A_data, 12);
	Arange(A,1);

    Conv2D<qint32_t> layer{ 1, 3, { 1, 1 }, { 2, 1 }, { 2, 2 }, false, PaddingMode::CIRCULAR };
    Fill(layer.weights, 1);
    
    const auto out = layer.forward(A);
    
    qint32_t R_data[] = { 8, 9, 7, 8, 9, 7, 8, 2, 3, 1, 2, 3, 1, 2, 8, 9, 7, 8, 9, 7, 8, 2, 3, 1, 2, 3, 1, 2, 8, 9, 7, 8, 9, 7, 8, 2, 3, 1, 2, 3, 1, 2, 8, 9, 7, 8, 9, 7, 8, 2, 3, 1, 2, 3, 1, 2, 8, 9, 7, 8, 9, 7, 8, 2, 3, 1, 2, 3, 1, 2, 8, 9, 7, 8, 9, 7, 8, 2, 3, 1, 2, 3, 1, 2 };
	Tensor<qint32_t> R{ 1, 3, 4, 7 };
	R.allocateCustom(R_data, 84);

    return Equals(out, R);
}
inline int testConv2DCase22()
{
    qint32_t A_data[12] ;
	Tensor<qint32_t> A{ 1, 4, 3 };
	A.allocateCustom(A_data, 12);
	Arange(A,1);

    Conv2D<qint32_t> layer{ 1, 1, { 2, 1 }, { 1, 2 }, { 0, 0 }, true, PaddingMode::REPLICATE };
    Fill(layer.weights, 1);
    Fill(layer.biases, 1);
    const auto out = layer.forward(A);
    
    qint32_t R_data[] = { 6, 10, 12, 16, 18, 22 };
	Tensor<qint32_t> R{ 1, 1, 3, 2 };
	R.allocateCustom(R_data, 6);

    return Equals(out, R);
}
inline int testConv2DCase23()
{
    qint32_t A_data[16] ;
	Tensor<qint32_t> A{ 1, 4, 4 };
	A.allocateCustom(A_data, 16);
	Arange(A,1);

    Conv2D<qint32_t> layer{ 1, 3, { 1, 1 }, { 1, 2 }, { 0, 0 }, false, PaddingMode::REPLICATE };
    Fill(layer.weights, 1);
    
    const auto out = layer.forward(A);
    
    qint32_t R_data[] = { 1, 3, 5, 7, 9, 11, 13, 15, 1, 3, 5, 7, 9, 11, 13, 15, 1, 3, 5, 7, 9, 11, 13, 15 };
	Tensor<qint32_t> R{ 1, 3, 4, 2 };
	R.allocateCustom(R_data, 24);

    return Equals(out, R);
}
inline int testConv2DCase24()
{
    qint32_t A_data[48] ;
	Tensor<qint32_t> A{ 3, 4, 4 };
	A.allocateCustom(A_data, 48);
	Arange(A,1);

    Conv2D<qint32_t> layer{ 3, 1, { 2, 1 }, { 2, 1 }, { 1, 0 }, true, PaddingMode::ZEROS };
    Fill(layer.weights, 1);
    Fill(layer.biases, 1);
    const auto out = layer.forward(A);
    
    qint32_t R_data[] = { 52, 55, 58, 61, 139, 145, 151, 157, 88, 91, 94, 97 };
	Tensor<qint32_t> R{ 1, 1, 3, 4 };
	R.allocateCustom(R_data, 12);

    return Equals(out, R);
}
inline int testConv2DCase25()
{
    qint32_t A_data[36] ;
	Tensor<qint32_t> A{ 3, 4, 3 };
	A.allocateCustom(A_data, 36);
	Arange(A,1);

    Conv2D<qint32_t> layer{ 3, 1, { 2, 1 }, { 1, 2 }, { 0, 1 }, false, PaddingMode::REFLECT };
    Fill(layer.weights, 1);
    
    const auto out = layer.forward(A);
    
    qint32_t R_data[] = { 93, 93, 93, 111, 111, 111, 129, 129, 129 };
	Tensor<qint32_t> R{ 1, 1, 3, 3 };
	R.allocateCustom(R_data, 9);

    return Equals(out, R);
}
inline int testConv2DCase26()
{
    qint32_t A_data[16] ;
	Tensor<qint32_t> A{ 1, 4, 4 };
	A.allocateCustom(A_data, 16);
	Arange(A,1);

    Conv2D<qint32_t> layer{ 1, 1, { 1, 2 }, { 1, 1 }, { 2, 2 }, true, PaddingMode::ZEROS };
    Fill(layer.weights, 1);
    Fill(layer.biases, 1);
    const auto out = layer.forward(A);
    
    qint32_t R_data[] = { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 4, 6, 8, 5, 1, 1, 6, 12, 14, 16, 9, 1, 1, 10, 20, 22, 24, 13, 1, 1, 14, 28, 30, 32, 17, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 };
	Tensor<qint32_t> R{ 1, 1, 8, 7 };
	R.allocateCustom(R_data, 56);

    return Equals(out, R);
}
inline int testConv2DCase27()
{
    qint32_t A_data[36] ;
	Tensor<qint32_t> A{ 3, 4, 3 };
	A.allocateCustom(A_data, 36);
	Arange(A,1);

    Conv2D<qint32_t> layer{ 3, 1, { 1, 1 }, { 2, 1 }, { 0, 1 }, false, PaddingMode::ZEROS };
    Fill(layer.weights, 1);
    
    const auto out = layer.forward(A);
    
    qint32_t R_data[] = { 0, 39, 42, 45, 0, 0, 57, 60, 63, 0 };
	Tensor<qint32_t> R{ 1, 1, 2, 5 };
	R.allocateCustom(R_data, 10);

    return Equals(out, R);
}
inline int testConv2DCase28()
{
    qint32_t A_data[36] ;
	Tensor<qint32_t> A{ 3, 4, 3 };
	A.allocateCustom(A_data, 36);
	Arange(A,1);

    Conv2D<qint32_t> layer{ 3, 1, { 1, 1 }, { 1, 2 }, { 1, 0 }, true, PaddingMode::REFLECT };
    Fill(layer.weights, 1);
    Fill(layer.biases, 1);
    const auto out = layer.forward(A);
    
    qint32_t R_data[] = { 49, 55, 40, 46, 49, 55, 58, 64, 67, 73, 58, 64 };
	Tensor<qint32_t> R{ 1, 1, 6, 2 };
	R.allocateCustom(R_data, 12);

    return Equals(out, R);
}
inline int testConv2DCase29()
{
    qint32_t A_data[12] ;
	Tensor<qint32_t> A{ 1, 4, 3 };
	A.allocateCustom(A_data, 12);
	Arange(A,1);

    Conv2D<qint32_t> layer{ 1, 1, { 2, 1 }, { 1, 1 }, { 1, 0 }, false, PaddingMode::CIRCULAR };
    Fill(layer.weights, 1);
    
    const auto out = layer.forward(A);
    
    qint32_t R_data[] = { 11, 13, 15, 5, 7, 9, 11, 13, 15, 17, 19, 21, 11, 13, 15 };
	Tensor<qint32_t> R{ 1, 1, 5, 3 };
	R.allocateCustom(R_data, 15);

    return Equals(out, R);
}
inline int testConv2DCase30()
{
    qint32_t A_data[48] ;
	Tensor<qint32_t> A{ 3, 4, 4 };
	A.allocateCustom(A_data, 48);
	Arange(A,1);

    Conv2D<qint32_t> layer{ 3, 1, { 2, 1 }, { 1, 1 }, { 1, 0 }, true, PaddingMode::ZEROS };
    Fill(layer.weights, 1);
    Fill(layer.biases, 1);
    const auto out = layer.forward(A);
    
    qint32_t R_data[] = { 52, 55, 58, 61, 115, 121, 127, 133, 139, 145, 151, 157, 163, 169, 175, 181, 88, 91, 94, 97 };
	Tensor<qint32_t> R{ 1, 1, 5, 4 };
	R.allocateCustom(R_data, 20);

    return Equals(out, R);
}
inline int testConv2DCase31()
{
    qint32_t A_data[16] ;
	Tensor<qint32_t> A{ 1, 4, 4 };
	A.allocateCustom(A_data, 16);
	Arange(A,1);

    Conv2D<qint32_t> layer{ 1, 1, { 2, 1 }, { 1, 1 }, { 0, 1 }, false, PaddingMode::ZEROS };
    Fill(layer.weights, 1);
    
    const auto out = layer.forward(A);
    
    qint32_t R_data[] = { 0, 6, 8, 10, 12, 0, 0, 14, 16, 18, 20, 0, 0, 22, 24, 26, 28, 0 };
	Tensor<qint32_t> R{ 1, 1, 3, 6 };
	R.allocateCustom(R_data, 18);

    return Equals(out, R);
}
inline int testConv2DCase32()
{
    qint32_t A_data[48] ;
	Tensor<qint32_t> A{ 3, 4, 4 };
	A.allocateCustom(A_data, 48);
	Arange(A,1);

    Conv2D<qint32_t> layer{ 3, 1, { 2, 1 }, { 1, 1 }, { 0, 0 }, true, PaddingMode::REPLICATE };
    Fill(layer.weights, 1);
    Fill(layer.biases, 1);
    const auto out = layer.forward(A);
    
    qint32_t R_data[] = { 115, 121, 127, 133, 139, 145, 151, 157, 163, 169, 175, 181 };
	Tensor<qint32_t> R{ 1, 1, 3, 4 };
	R.allocateCustom(R_data, 12);

    return Equals(out, R);
}
inline int testConv2DCase33()
{
    qint32_t A_data[36] ;
	Tensor<qint32_t> A{ 3, 3, 4 };
	A.allocateCustom(A_data, 36);
	Arange(A,1);

    Conv2D<qint32_t> layer{ 3, 1, { 1, 1 }, { 1, 2 }, { 0, 0 }, false, PaddingMode::REFLECT };
    Fill(layer.weights, 1);
    
    const auto out = layer.forward(A);
    
    qint32_t R_data[] = { 39, 45, 51, 57, 63, 69 };
	Tensor<qint32_t> R{ 1, 1, 3, 2 };
	R.allocateCustom(R_data, 6);

    return Equals(out, R);
}
inline int testConv2DCase34()
{
    qint32_t A_data[36] ;
	Tensor<qint32_t> A{ 3, 4, 3 };
	A.allocateCustom(A_data, 36);
	Arange(A,1);

    Conv2D<qint32_t> layer{ 3, 1, { 1, 2 }, { 1, 1 }, { 1, 0 }, true, PaddingMode::CIRCULAR };
    Fill(layer.weights, 1);
    Fill(layer.biases, 1);
    const auto out = layer.forward(A);
    
    qint32_t R_data[] = { 136, 142, 82, 88, 100, 106, 118, 124, 136, 142, 82, 88 };
	Tensor<qint32_t> R{ 1, 1, 6, 2 };
	R.allocateCustom(R_data, 12);

    return Equals(out, R);
}
inline int testConv2DCase35()
{
    qint32_t A_data[36] ;
	Tensor<qint32_t> A{ 3, 3, 4 };
	A.allocateCustom(A_data, 36);
	Arange(A,1);

    Conv2D<qint32_t> layer{ 3, 1, { 1, 1 }, { 1, 1 }, { 0, 1 }, false, PaddingMode::REPLICATE };
    Fill(layer.weights, 1);
    
    const auto out = layer.forward(A);
    
    qint32_t R_data[] = { 39, 39, 42, 45, 48, 48, 51, 51, 54, 57, 60, 60, 63, 63, 66, 69, 72, 72 };
	Tensor<qint32_t> R{ 1, 1, 3, 6 };
	R.allocateCustom(R_data, 18);

    return Equals(out, R);
}
inline int testConv2DCase36()
{
    qint32_t A_data[48] ;
	Tensor<qint32_t> A{ 3, 4, 4 };
	A.allocateCustom(A_data, 48);
	Arange(A,1);

    Conv2D<qint32_t> layer{ 3, 1, { 2, 1 }, { 1, 2 }, { 0, 1 }, true, PaddingMode::CIRCULAR };
    Fill(layer.weights, 1);
    Fill(layer.biases, 1);
    const auto out = layer.forward(A);
    
    qint32_t R_data[] = { 133, 121, 133, 157, 145, 157, 181, 169, 181 };
	Tensor<qint32_t> R{ 1, 1, 3, 3 };
	R.allocateCustom(R_data, 9);

    return Equals(out, R);
}
inline int testConv2DCase37()
{
    qint32_t A_data[12] ;
	Tensor<qint32_t> A{ 1, 4, 3 };
	A.allocateCustom(A_data, 12);
	Arange(A,1);

    Conv2D<qint32_t> layer{ 1, 1, { 2, 1 }, { 1, 2 }, { 0, 1 }, false, PaddingMode::ZEROS };
    Fill(layer.weights, 1);
    
    const auto out = layer.forward(A);
    
    qint32_t R_data[] = { 0, 7, 0, 0, 13, 0, 0, 19, 0 };
	Tensor<qint32_t> R{ 1, 1, 3, 3 };
	R.allocateCustom(R_data, 9);

    return Equals(out, R);
}
inline int testConv2DCase38()
{
    qint32_t A_data[12] ;
	Tensor<qint32_t> A{ 1, 4, 3 };
	A.allocateCustom(A_data, 12);
	Arange(A,1);

    Conv2D<qint32_t> layer{ 1, 3, { 1, 2 }, { 2, 1 }, { 0, 1 }, true, PaddingMode::CIRCULAR };
    Fill(layer.weights, 1);
    Fill(layer.biases, 1);
    const auto out = layer.forward(A);
    
    qint32_t R_data[] = { 5, 4, 6, 5, 17, 16, 18, 17, 5, 4, 6, 5, 17, 16, 18, 17, 5, 4, 6, 5, 17, 16, 18, 17 };
	Tensor<qint32_t> R{ 1, 3, 2, 4 };
	R.allocateCustom(R_data, 24);

    return Equals(out, R);
}
inline int testConv2DCase39()
{
    qint32_t A_data[16] ;
	Tensor<qint32_t> A{ 1, 4, 4 };
	A.allocateCustom(A_data, 16);
	Arange(A,1);

    Conv2D<qint32_t> layer{ 1, 1, { 1, 1 }, { 1, 2 }, { 2, 2 }, false, PaddingMode::CIRCULAR };
    Fill(layer.weights, 1);
    
    const auto out = layer.forward(A);
    
    qint32_t R_data[] = { 11, 9, 11, 9, 15, 13, 15, 13, 3, 1, 3, 1, 7, 5, 7, 5, 11, 9, 11, 9, 15, 13, 15, 13, 3, 1, 3, 1, 7, 5, 7, 5 };
	Tensor<qint32_t> R{ 1, 1, 8, 4 };
	R.allocateCustom(R_data, 32);

    return Equals(out, R);
}
inline int testConv2DCase40()
{
    qint32_t A_data[16] ;
	Tensor<qint32_t> A{ 1, 4, 4 };
	A.allocateCustom(A_data, 16);
	Arange(A,1);

    Conv2D<qint32_t> layer{ 1, 3, { 1, 1 }, { 1, 2 }, { 0, 1 }, true, PaddingMode::CIRCULAR };
    Fill(layer.weights, 1);
    Fill(layer.biases, 1);
    const auto out = layer.forward(A);
    
    qint32_t R_data[] = { 5, 3, 5, 9, 7, 9, 13, 11, 13, 17, 15, 17, 5, 3, 5, 9, 7, 9, 13, 11, 13, 17, 15, 17, 5, 3, 5, 9, 7, 9, 13, 11, 13, 17, 15, 17 };
	Tensor<qint32_t> R{ 1, 3, 4, 3 };
	R.allocateCustom(R_data, 36);

    return Equals(out, R);
}
inline int testConv2DCase41()
{
    qint32_t A_data[16] ;
	Tensor<qint32_t> A{ 1, 4, 4 };
	A.allocateCustom(A_data, 16);
	Arange(A,1);

    Conv2D<qint32_t> layer{ 1, 1, { 2, 1 }, { 1, 2 }, { 2, 2 }, false, PaddingMode::REPLICATE };
    Fill(layer.weights, 1);
    
    const auto out = layer.forward(A);
    
    qint32_t R_data[] = { 2, 2, 6, 8, 2, 2, 6, 8, 6, 6, 10, 12, 14, 14, 18, 20, 22, 22, 26, 28, 26, 26, 30, 32, 26, 26, 30, 32 };
	Tensor<qint32_t> R{ 1, 1, 7, 4 };
	R.allocateCustom(R_data, 28);

    return Equals(out, R);
}
inline int testConv2DCase42()
{
    qint32_t A_data[36] ;
	Tensor<qint32_t> A{ 3, 3, 4 };
	A.allocateCustom(A_data, 36);
	Arange(A,1);

    Conv2D<qint32_t> layer{ 3, 1, { 1, 1 }, { 1, 1 }, { 0, 1 }, true, PaddingMode::REPLICATE };
    Fill(layer.weights, 1);
    Fill(layer.biases, 1);
    const auto out = layer.forward(A);
    
    qint32_t R_data[] = { 40, 40, 43, 46, 49, 49, 52, 52, 55, 58, 61, 61, 64, 64, 67, 70, 73, 73 };
	Tensor<qint32_t> R{ 1, 1, 3, 6 };
	R.allocateCustom(R_data, 18);

    return Equals(out, R);
}
inline int testConv2DCase43()
{
    qint32_t A_data[48] ;
	Tensor<qint32_t> A{ 3, 4, 4 };
	A.allocateCustom(A_data, 48);
	Arange(A,1);

    Conv2D<qint32_t> layer{ 3, 1, { 1, 1 }, { 2, 1 }, { 0, 0 }, false, PaddingMode::ZEROS };
    Fill(layer.weights, 1);
    
    const auto out = layer.forward(A);
    
    qint32_t R_data[] = { 51, 54, 57, 60, 75, 78, 81, 84 };
	Tensor<qint32_t> R{ 1, 1, 2, 4 };
	R.allocateCustom(R_data, 8);

    return Equals(out, R);
}
inline int testConv2DCase44()
{
    qint32_t A_data[16] ;
	Tensor<qint32_t> A{ 1, 4, 4 };
	A.allocateCustom(A_data, 16);
	Arange(A,1);

    Conv2D<qint32_t> layer{ 1, 1, { 1, 2 }, { 1, 2 }, { 2, 2 }, true, PaddingMode::REPLICATE };
    Fill(layer.weights, 1);
    Fill(layer.biases, 1);
    const auto out = layer.forward(A);
    
    qint32_t R_data[] = { 3, 4, 8, 9, 3, 4, 8, 9, 3, 4, 8, 9, 11, 12, 16, 17, 19, 20, 24, 25, 27, 28, 32, 33, 27, 28, 32, 33, 27, 28, 32, 33 };
	Tensor<qint32_t> R{ 1, 1, 8, 4 };
	R.allocateCustom(R_data, 32);

    return Equals(out, R);
}
inline int testConv2DCase45()
{
    qint32_t A_data[36] ;
	Tensor<qint32_t> A{ 3, 3, 4 };
	A.allocateCustom(A_data, 36);
	Arange(A,1);

    Conv2D<qint32_t> layer{ 3, 1, { 1, 1 }, { 1, 2 }, { 1, 0 }, false, PaddingMode::REPLICATE };
    Fill(layer.weights, 1);
    
    const auto out = layer.forward(A);
    
    qint32_t R_data[] = { 39, 45, 39, 45, 51, 57, 63, 69, 63, 69 };
	Tensor<qint32_t> R{ 1, 1, 5, 2 };
	R.allocateCustom(R_data, 10);

    return Equals(out, R);
}
inline int testConv2DCase46()
{
    qint32_t A_data[12] ;
	Tensor<qint32_t> A{ 1, 4, 3 };
	A.allocateCustom(A_data, 12);
	Arange(A,1);

    Conv2D<qint32_t> layer{ 1, 1, { 1, 2 }, { 2, 1 }, { 2, 2 }, true, PaddingMode::ZEROS };
    Fill(layer.weights, 1);
    Fill(layer.biases, 1);
    const auto out = layer.forward(A);
    
    qint32_t R_data[] = { 1, 1, 1, 1, 1, 1, 1, 2, 4, 6, 4, 1, 1, 8, 16, 18, 10, 1, 1, 1, 1, 1, 1, 1 };
	Tensor<qint32_t> R{ 1, 1, 4, 6 };
	R.allocateCustom(R_data, 24);

    return Equals(out, R);
}
inline int testConv2DCase47()
{
    qint32_t A_data[16] ;
	Tensor<qint32_t> A{ 1, 4, 4 };
	A.allocateCustom(A_data, 16);
	Arange(A,1);

    Conv2D<qint32_t> layer{ 1, 1, { 1, 1 }, { 2, 1 }, { 1, 0 }, false, PaddingMode::REPLICATE };
    Fill(layer.weights, 1);
    
    const auto out = layer.forward(A);
    
    qint32_t R_data[] = { 1, 2, 3, 4, 5, 6, 7, 8, 13, 14, 15, 16 };
	Tensor<qint32_t> R{ 1, 1, 3, 4 };
	R.allocateCustom(R_data, 12);

    return Equals(out, R);
}
inline int testConv2DCase48()
{
    qint32_t A_data[16] ;
	Tensor<qint32_t> A{ 1, 4, 4 };
	A.allocateCustom(A_data, 16);
	Arange(A,1);

    Conv2D<qint32_t> layer{ 1, 3, { 2, 1 }, { 1, 1 }, { 0, 0 }, true, PaddingMode::CIRCULAR };
    Fill(layer.weights, 1);
    Fill(layer.biases, 1);
    const auto out = layer.forward(A);
    
    qint32_t R_data[] = { 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29 };
	Tensor<qint32_t> R{ 1, 3, 3, 4 };
	R.allocateCustom(R_data, 36);

    return Equals(out, R);
}
inline int testConv2DCase49()
{
    qint32_t A_data[16] ;
	Tensor<qint32_t> A{ 1, 4, 4 };
	A.allocateCustom(A_data, 16);
	Arange(A,1);

    Conv2D<qint32_t> layer{ 1, 3, { 1, 2 }, { 1, 2 }, { 2, 2 }, false, PaddingMode::ZEROS };
    Fill(layer.weights, 1);
    
    const auto out = layer.forward(A);
    
    qint32_t R_data[] = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 7, 0, 0, 11, 15, 0, 0, 19, 23, 0, 0, 27, 31, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 7, 0, 0, 11, 15, 0, 0, 19, 23, 0, 0, 27, 31, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 7, 0, 0, 11, 15, 0, 0, 19, 23, 0, 0, 27, 31, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
	Tensor<qint32_t> R{ 1, 3, 8, 4 };
	R.allocateCustom(R_data, 96);

    return Equals(out, R);
}
inline int testConv2DCase50()
{
    qint32_t A_data[12] ;
	Tensor<qint32_t> A{ 1, 3, 4 };
	A.allocateCustom(A_data, 12);
	Arange(A,1);

    Conv2D<qint32_t> layer{ 1, 1, { 2, 1 }, { 2, 1 }, { 0, 0 }, true, PaddingMode::ZEROS };
    Fill(layer.weights, 1);
    Fill(layer.biases, 1);
    const auto out = layer.forward(A);
    
    qint32_t R_data[] = { 7, 9, 11, 13 };
	Tensor<qint32_t> R{ 1, 1, 1, 4 };
	R.allocateCustom(R_data, 4);

    return Equals(out, R);
}
inline int testConv2DCase51()
{
    qint32_t A_data[36] ;
	Tensor<qint32_t> A{ 3, 3, 4 };
	A.allocateCustom(A_data, 36);
	Arange(A,1);

    Conv2D<qint32_t> layer{ 3, 1, { 2, 1 }, { 1, 1 }, { 0, 1 }, false, PaddingMode::CIRCULAR };
    Fill(layer.weights, 1);
    
    const auto out = layer.forward(A);
    
    qint32_t R_data[] = { 108, 90, 96, 102, 108, 90, 132, 114, 120, 126, 132, 114 };
	Tensor<qint32_t> R{ 1, 1, 2, 6 };
	R.allocateCustom(R_data, 12);

    return Equals(out, R);
}
inline int testConv2DCase52()
{
    qint32_t A_data[12] ;
	Tensor<qint32_t> A{ 1, 3, 4 };
	A.allocateCustom(A_data, 12);
	Arange(A,1);

    Conv2D<qint32_t> layer{ 1, 1, { 1, 1 }, { 1, 1 }, { 0, 1 }, true, PaddingMode::ZEROS };
    Fill(layer.weights, 1);
    Fill(layer.biases, 1);
    const auto out = layer.forward(A);
    
    qint32_t R_data[] = { 1, 2, 3, 4, 5, 1, 1, 6, 7, 8, 9, 1, 1, 10, 11, 12, 13, 1 };
	Tensor<qint32_t> R{ 1, 1, 3, 6 };
	R.allocateCustom(R_data, 18);

    return Equals(out, R);
}
inline int testConv2DCase53()
{
    qint32_t A_data[12] ;
	Tensor<qint32_t> A{ 1, 4, 3 };
	A.allocateCustom(A_data, 12);
	Arange(A,1);

    Conv2D<qint32_t> layer{ 1, 3, { 1, 1 }, { 1, 1 }, { 2, 2 }, false, PaddingMode::CIRCULAR };
    Fill(layer.weights, 1);
    
    const auto out = layer.forward(A);
    
    qint32_t R_data[] = { 8, 9, 7, 8, 9, 7, 8, 11, 12, 10, 11, 12, 10, 11, 2, 3, 1, 2, 3, 1, 2, 5, 6, 4, 5, 6, 4, 5, 8, 9, 7, 8, 9, 7, 8, 11, 12, 10, 11, 12, 10, 11, 2, 3, 1, 2, 3, 1, 2, 5, 6, 4, 5, 6, 4, 5, 8, 9, 7, 8, 9, 7, 8, 11, 12, 10, 11, 12, 10, 11, 2, 3, 1, 2, 3, 1, 2, 5, 6, 4, 5, 6, 4, 5, 8, 9, 7, 8, 9, 7, 8, 11, 12, 10, 11, 12, 10, 11, 2, 3, 1, 2, 3, 1, 2, 5, 6, 4, 5, 6, 4, 5, 8, 9, 7, 8, 9, 7, 8, 11, 12, 10, 11, 12, 10, 11, 2, 3, 1, 2, 3, 1, 2, 5, 6, 4, 5, 6, 4, 5, 8, 9, 7, 8, 9, 7, 8, 11, 12, 10, 11, 12, 10, 11, 2, 3, 1, 2, 3, 1, 2, 5, 6, 4, 5, 6, 4, 5 };
	Tensor<qint32_t> R{ 1, 3, 8, 7 };
	R.allocateCustom(R_data, 168);

    return Equals(out, R);
}
inline int testConv2DCase54()
{
    qint32_t A_data[48] ;
	Tensor<qint32_t> A{ 3, 4, 4 };
	A.allocateCustom(A_data, 48);
	Arange(A,1);

    Conv2D<qint32_t> layer{ 3, 1, { 2, 1 }, { 1, 2 }, { 0, 0 }, true, PaddingMode::CIRCULAR };
    Fill(layer.weights, 1);
    Fill(layer.biases, 1);
    const auto out = layer.forward(A);
    
    qint32_t R_data[] = { 115, 127, 139, 151, 163, 175 };
	Tensor<qint32_t> R{ 1, 1, 3, 2 };
	R.allocateCustom(R_data, 6);

    return Equals(out, R);
}
inline int testConv2DCase55()
{
    qint32_t A_data[12] ;
	Tensor<qint32_t> A{ 1, 4, 3 };
	A.allocateCustom(A_data, 12);
	Arange(A,1);

    Conv2D<qint32_t> layer{ 1, 3, { 1, 2 }, { 2, 1 }, { 0, 0 }, false, PaddingMode::REPLICATE };
    Fill(layer.weights, 1);
    
    const auto out = layer.forward(A);
    
    qint32_t R_data[] = { 3, 5, 15, 17, 3, 5, 15, 17, 3, 5, 15, 17 };
	Tensor<qint32_t> R{ 1, 3, 2, 2 };
	R.allocateCustom(R_data, 12);

    return Equals(out, R);
}
inline int testConv2DCase56()
{
    qint32_t A_data[36] ;
	Tensor<qint32_t> A{ 3, 3, 4 };
	A.allocateCustom(A_data, 36);
	Arange(A,1);

    Conv2D<qint32_t> layer{ 3, 1, { 2, 1 }, { 1, 2 }, { 0, 1 }, true, PaddingMode::CIRCULAR };
    Fill(layer.weights, 1);
    Fill(layer.biases, 1);
    const auto out = layer.forward(A);
    
    qint32_t R_data[] = { 109, 97, 109, 133, 121, 133 };
	Tensor<qint32_t> R{ 1, 1, 2, 3 };
	R.allocateCustom(R_data, 6);

    return Equals(out, R);
}
inline int testConv2DCase57()
{
    qint32_t A_data[12] ;
	Tensor<qint32_t> A{ 1, 4, 3 };
	A.allocateCustom(A_data, 12);
	Arange(A,1);

    Conv2D<qint32_t> layer{ 1, 3, { 1, 1 }, { 1, 2 }, { 0, 1 }, false, PaddingMode::REFLECT };
    Fill(layer.weights, 1);
    
    const auto out = layer.forward(A);
    
    qint32_t R_data[] = { 2, 2, 2, 5, 5, 5, 8, 8, 8, 11, 11, 11, 2, 2, 2, 5, 5, 5, 8, 8, 8, 11, 11, 11, 2, 2, 2, 5, 5, 5, 8, 8, 8, 11, 11, 11 };
	Tensor<qint32_t> R{ 1, 3, 4, 3 };
	R.allocateCustom(R_data, 36);

    return Equals(out, R);
}
inline int testConv2DCase58()
{
    qint32_t A_data[16] ;
	Tensor<qint32_t> A{ 1, 4, 4 };
	A.allocateCustom(A_data, 16);
	Arange(A,1);

    Conv2D<qint32_t> layer{ 1, 1, { 2, 1 }, { 2, 1 }, { 2, 2 }, true, PaddingMode::REPLICATE };
    Fill(layer.weights, 1);
    Fill(layer.biases, 1);
    const auto out = layer.forward(A);
    
    qint32_t R_data[] = { 3, 3, 3, 5, 7, 9, 9, 9, 7, 7, 7, 9, 11, 13, 13, 13, 23, 23, 23, 25, 27, 29, 29, 29, 27, 27, 27, 29, 31, 33, 33, 33 };
	Tensor<qint32_t> R{ 1, 1, 4, 8 };
	R.allocateCustom(R_data, 32);

    return Equals(out, R);
}
inline int testConv2DCase59()
{
    qint32_t A_data[36] ;
	Tensor<qint32_t> A{ 3, 4, 3 };
	A.allocateCustom(A_data, 36);
	Arange(A,1);

    Conv2D<qint32_t> layer{ 3, 1, { 1, 1 }, { 1, 1 }, { 0, 0 }, false, PaddingMode::REFLECT };
    Fill(layer.weights, 1);
    
    const auto out = layer.forward(A);
    
    qint32_t R_data[] = { 39, 42, 45, 48, 51, 54, 57, 60, 63, 66, 69, 72 };
	Tensor<qint32_t> R{ 1, 1, 4, 3 };
	R.allocateCustom(R_data, 12);

    return Equals(out, R);
}
inline int testConv2DCase60()
{
    qint32_t A_data[12] ;
	Tensor<qint32_t> A{ 1, 3, 4 };
	A.allocateCustom(A_data, 12);
	Arange(A,1);

    Conv2D<qint32_t> layer{ 1, 3, { 2, 1 }, { 1, 2 }, { 2, 2 }, true, PaddingMode::REFLECT };
    Fill(layer.weights, 1);
    Fill(layer.biases, 1);
    const auto out = layer.forward(A);
    
    qint32_t R_data[] = { 19, 15, 19, 19, 11, 7, 11, 11, 11, 7, 11, 11, 19, 15, 19, 19, 19, 15, 19, 19, 11, 7, 11, 11, 19, 15, 19, 19, 11, 7, 11, 11, 11, 7, 11, 11, 19, 15, 19, 19, 19, 15, 19, 19, 11, 7, 11, 11, 19, 15, 19, 19, 11, 7, 11, 11, 11, 7, 11, 11, 19, 15, 19, 19, 19, 15, 19, 19, 11, 7, 11, 11 };
	Tensor<qint32_t> R{ 1, 3, 6, 4 };
	R.allocateCustom(R_data, 72);

    return Equals(out, R);
}
inline int testConv2DCase61()
{
    qint32_t A_data[36] ;
	Tensor<qint32_t> A{ 3, 3, 4 };
	A.allocateCustom(A_data, 36);
	Arange(A,1);

    Conv2D<qint32_t> layer{ 3, 1, { 1, 1 }, { 1, 2 }, { 0, 0 }, false, PaddingMode::ZEROS };
    Fill(layer.weights, 1);
    
    const auto out = layer.forward(A);
    
    qint32_t R_data[] = { 39, 45, 51, 57, 63, 69 };
	Tensor<qint32_t> R{ 1, 1, 3, 2 };
	R.allocateCustom(R_data, 6);

    return Equals(out, R);
}
inline int testConv2DCase62()
{
    qint32_t A_data[16] ;
	Tensor<qint32_t> A{ 1, 4, 4 };
	A.allocateCustom(A_data, 16);
	Arange(A,1);

    Conv2D<qint32_t> layer{ 1, 1, { 2, 1 }, { 1, 1 }, { 0, 0 }, true, PaddingMode::ZEROS };
    Fill(layer.weights, 1);
    Fill(layer.biases, 1);
    const auto out = layer.forward(A);
    
    qint32_t R_data[] = { 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29 };
	Tensor<qint32_t> R{ 1, 1, 3, 4 };
	R.allocateCustom(R_data, 12);

    return Equals(out, R);
}
inline int testConv2DCase63()
{
    qint32_t A_data[36] ;
	Tensor<qint32_t> A{ 3, 3, 4 };
	A.allocateCustom(A_data, 36);
	Arange(A,1);

    Conv2D<qint32_t> layer{ 3, 1, { 1, 1 }, { 1, 2 }, { 0, 1 }, false, PaddingMode::REPLICATE };
    Fill(layer.weights, 1);
    
    const auto out = layer.forward(A);
    
    qint32_t R_data[] = { 39, 42, 48, 51, 54, 60, 63, 66, 72 };
	Tensor<qint32_t> R{ 1, 1, 3, 3 };
	R.allocateCustom(R_data, 9);

    return Equals(out, R);
}
inline int testConv2DCase64()
{
    qint32_t A_data[12] ;
	Tensor<qint32_t> A{ 1, 3, 4 };
	A.allocateCustom(A_data, 12);
	Arange(A,1);

    Conv2D<qint32_t> layer{ 1, 1, { 1, 2 }, { 2, 1 }, { 2, 2 }, true, PaddingMode::CIRCULAR };
    Fill(layer.weights, 1);
    Fill(layer.biases, 1);
    const auto out = layer.forward(A);
    
    qint32_t R_data[] = { 16, 14, 12, 14, 16, 14, 12, 8, 6, 4, 6, 8, 6, 4, 24, 22, 20, 22, 24, 22, 20, 16, 14, 12, 14, 16, 14, 12 };
	Tensor<qint32_t> R{ 1, 1, 4, 7 };
	R.allocateCustom(R_data, 28);

    return Equals(out, R);
}
inline int testConv2DCase65()
{
    qint32_t A_data[16] ;
	Tensor<qint32_t> A{ 1, 4, 4 };
	A.allocateCustom(A_data, 16);
	Arange(A,1);

    Conv2D<qint32_t> layer{ 1, 3, { 1, 2 }, { 1, 1 }, { 0, 1 }, false, PaddingMode::CIRCULAR };
    Fill(layer.weights, 1);
    
    const auto out = layer.forward(A);
    
    qint32_t R_data[] = { 5, 3, 5, 7, 5, 13, 11, 13, 15, 13, 21, 19, 21, 23, 21, 29, 27, 29, 31, 29, 5, 3, 5, 7, 5, 13, 11, 13, 15, 13, 21, 19, 21, 23, 21, 29, 27, 29, 31, 29, 5, 3, 5, 7, 5, 13, 11, 13, 15, 13, 21, 19, 21, 23, 21, 29, 27, 29, 31, 29 };
	Tensor<qint32_t> R{ 1, 3, 4, 5 };
	R.allocateCustom(R_data, 60);

    return Equals(out, R);
}
inline int testConv2DCase66()
{
    qint32_t A_data[36] ;
	Tensor<qint32_t> A{ 3, 4, 3 };
	A.allocateCustom(A_data, 36);
	Arange(A,1);

    Conv2D<qint32_t> layer{ 3, 1, { 1, 2 }, { 2, 1 }, { 0, 1 }, true, PaddingMode::ZEROS };
    Fill(layer.weights, 1);
    Fill(layer.biases, 1);
    const auto out = layer.forward(A);
    
    qint32_t R_data[] = { 40, 82, 88, 46, 58, 118, 124, 64 };
	Tensor<qint32_t> R{ 1, 1, 2, 4 };
	R.allocateCustom(R_data, 8);

    return Equals(out, R);
}
inline int testConv2DCase67()
{
    qint32_t A_data[12] ;
	Tensor<qint32_t> A{ 1, 3, 4 };
	A.allocateCustom(A_data, 12);
	Arange(A,1);

    Conv2D<qint32_t> layer{ 1, 1, { 1, 1 }, { 1, 1 }, { 0, 0 }, false, PaddingMode::REPLICATE };
    Fill(layer.weights, 1);
    
    const auto out = layer.forward(A);
    
    qint32_t R_data[] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 };
	Tensor<qint32_t> R{ 1, 1, 3, 4 };
	R.allocateCustom(R_data, 12);

    return Equals(out, R);
}
inline int testConv2DCase68()
{
    qint32_t A_data[12] ;
	Tensor<qint32_t> A{ 1, 4, 3 };
	A.allocateCustom(A_data, 12);
	Arange(A,1);

    Conv2D<qint32_t> layer{ 1, 1, { 1, 1 }, { 1, 1 }, { 2, 2 }, true, PaddingMode::REFLECT };
    Fill(layer.weights, 1);
    Fill(layer.biases, 1);
    const auto out = layer.forward(A);
    
    qint32_t R_data[] = { 10, 9, 8, 9, 10, 9, 8, 7, 6, 5, 6, 7, 6, 5, 4, 3, 2, 3, 4, 3, 2, 7, 6, 5, 6, 7, 6, 5, 10, 9, 8, 9, 10, 9, 8, 13, 12, 11, 12, 13, 12, 11, 10, 9, 8, 9, 10, 9, 8, 7, 6, 5, 6, 7, 6, 5 };
	Tensor<qint32_t> R{ 1, 1, 8, 7 };
	R.allocateCustom(R_data, 56);

    return Equals(out, R);
}
inline int testConv2DCase69()
{
    qint32_t A_data[48] ;
	Tensor<qint32_t> A{ 3, 4, 4 };
	A.allocateCustom(A_data, 48);
	Arange(A,1);

    Conv2D<qint32_t> layer{ 3, 1, { 1, 1 }, { 2, 1 }, { 0, 0 }, false, PaddingMode::CIRCULAR };
    Fill(layer.weights, 1);
    
    const auto out = layer.forward(A);
    
    qint32_t R_data[] = { 51, 54, 57, 60, 75, 78, 81, 84 };
	Tensor<qint32_t> R{ 1, 1, 2, 4 };
	R.allocateCustom(R_data, 8);

    return Equals(out, R);
}
inline int testConv2DCase70()
{
    qint32_t A_data[12] ;
	Tensor<qint32_t> A{ 1, 3, 4 };
	A.allocateCustom(A_data, 12);
	Arange(A,1);

    Conv2D<qint32_t> layer{ 1, 1, { 1, 2 }, { 1, 2 }, { 2, 2 }, true, PaddingMode::REPLICATE };
    Fill(layer.weights, 1);
    Fill(layer.biases, 1);
    const auto out = layer.forward(A);
    
    qint32_t R_data[] = { 3, 4, 8, 9, 3, 4, 8, 9, 3, 4, 8, 9, 11, 12, 16, 17, 19, 20, 24, 25, 19, 20, 24, 25, 19, 20, 24, 25 };
	Tensor<qint32_t> R{ 1, 1, 7, 4 };
	R.allocateCustom(R_data, 28);

    return Equals(out, R);
}
inline int testConv2DCase71()
{
    qint32_t A_data[12] ;
	Tensor<qint32_t> A{ 1, 4, 3 };
	A.allocateCustom(A_data, 12);
	Arange(A,1);

    Conv2D<qint32_t> layer{ 1, 1, { 2, 1 }, { 2, 1 }, { 0, 1 }, false, PaddingMode::ZEROS };
    Fill(layer.weights, 1);
    
    const auto out = layer.forward(A);
    
    qint32_t R_data[] = { 0, 5, 7, 9, 0, 0, 17, 19, 21, 0 };
	Tensor<qint32_t> R{ 1, 1, 2, 5 };
	R.allocateCustom(R_data, 10);

    return Equals(out, R);
}
inline int testConv2DCase72()
{
    qint32_t A_data[16] ;
	Tensor<qint32_t> A{ 1, 4, 4 };
	A.allocateCustom(A_data, 16);
	Arange(A,1);

    Conv2D<qint32_t> layer{ 1, 1, { 1, 1 }, { 1, 2 }, { 1, 0 }, true, PaddingMode::ZEROS };
    Fill(layer.weights, 1);
    Fill(layer.biases, 1);
    const auto out = layer.forward(A);
    
    qint32_t R_data[] = { 1, 1, 2, 4, 6, 8, 10, 12, 14, 16, 1, 1 };
	Tensor<qint32_t> R{ 1, 1, 6, 2 };
	R.allocateCustom(R_data, 12);

    return Equals(out, R);
}
inline int testConv2DCase73()
{
    qint32_t A_data[12] ;
	Tensor<qint32_t> A{ 1, 3, 4 };
	A.allocateCustom(A_data, 12);
	Arange(A,1);

    Conv2D<qint32_t> layer{ 1, 1, { 1, 1 }, { 2, 1 }, { 2, 2 }, false, PaddingMode::REPLICATE };
    Fill(layer.weights, 1);
    
    const auto out = layer.forward(A);
    
    qint32_t R_data[] = { 1, 1, 1, 2, 3, 4, 4, 4, 1, 1, 1, 2, 3, 4, 4, 4, 9, 9, 9, 10, 11, 12, 12, 12, 9, 9, 9, 10, 11, 12, 12, 12 };
	Tensor<qint32_t> R{ 1, 1, 4, 8 };
	R.allocateCustom(R_data, 32);

    return Equals(out, R);
}
inline int testConv2DCase74()
{
    qint32_t A_data[48] ;
	Tensor<qint32_t> A{ 3, 4, 4 };
	A.allocateCustom(A_data, 48);
	Arange(A,1);

    Conv2D<qint32_t> layer{ 3, 1, { 1, 2 }, { 2, 1 }, { 0, 1 }, true, PaddingMode::REPLICATE };
    Fill(layer.weights, 1);
    Fill(layer.biases, 1);
    const auto out = layer.forward(A);
    
    qint32_t R_data[] = { 103, 106, 112, 118, 121, 151, 154, 160, 166, 169 };
	Tensor<qint32_t> R{ 1, 1, 2, 5 };
	R.allocateCustom(R_data, 10);

    return Equals(out, R);
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
    