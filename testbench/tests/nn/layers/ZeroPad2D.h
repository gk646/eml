
#ifndef EML_TEST_NN_LAYERS_ZEROPAD2D_H
#define EML_TEST_NN_LAYERS_ZEROPAD2D_H

inline int testZeroPad2DCase0()
{
    qint32_t A_data[20]{} ;
	Tensor<qint32_t> A{ 5, 4 };
	A.allocateCustom(A_data, 20);
	arange(A,  1);

    ZeroPad2D layer{ { 2,3 } };
    auto out = layer.forward(A);
    qint32_t R_data[] = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 0, 0, 0, 0, 0, 0, 5, 6, 7, 8, 0, 0, 0, 0, 0, 0, 9, 10, 11, 12, 0, 0, 0, 0, 0, 0, 13, 14, 15, 16, 0, 0, 0, 0, 0, 0, 17, 18, 19, 20, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
	Tensor<qint32_t> R{ 1, 1, 9, 10 };
	R.allocateCustom(R_data, 90);

    EML_ASSERT_TENSOR_EQUALS(R, out);
}
inline int testZeroPad2DCase1()
{
    qint32_t A_data[20]{} ;
	Tensor<qint32_t> A{ 4, 5 };
	A.allocateCustom(A_data, 20);
	arange(A,  1);

    ZeroPad2D layer{ { 1,1 } };
    auto out = layer.forward(A);
    qint32_t R_data[] = { 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 5, 0, 0, 6, 7, 8, 9, 10, 0, 0, 11, 12, 13, 14, 15, 0, 0, 16, 17, 18, 19, 20, 0, 0, 0, 0, 0, 0, 0, 0 };
	Tensor<qint32_t> R{ 1, 1, 6, 7 };
	R.allocateCustom(R_data, 42);

    EML_ASSERT_TENSOR_EQUALS(R, out);
}
inline int testZeroPad2DCase2()
{
    qint32_t A_data[56]{} ;
	Tensor<qint32_t> A{ 7, 8 };
	A.allocateCustom(A_data, 56);
	arange(A,  1);

    ZeroPad2D layer{ { 2,2 } };
    auto out = layer.forward(A);
    qint32_t R_data[] = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 0, 0, 0, 9, 10, 11, 12, 13, 14, 15, 16, 0, 0, 0, 0, 17, 18, 19, 20, 21, 22, 23, 24, 0, 0, 0, 0, 25, 26, 27, 28, 29, 30, 31, 32, 0, 0, 0, 0, 33, 34, 35, 36, 37, 38, 39, 40, 0, 0, 0, 0, 41, 42, 43, 44, 45, 46, 47, 48, 0, 0, 0, 0, 49, 50, 51, 52, 53, 54, 55, 56, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
	Tensor<qint32_t> R{ 1, 1, 11, 12 };
	R.allocateCustom(R_data, 132);

    EML_ASSERT_TENSOR_EQUALS(R, out);
}
inline int testZeroPad2DCase3()
{
    qint32_t A_data[20]{} ;
	Tensor<qint32_t> A{ 4, 5 };
	A.allocateCustom(A_data, 20);
	arange(A,  1);

    ZeroPad2D layer{ { 1,1 } };
    auto out = layer.forward(A);
    qint32_t R_data[] = { 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 5, 0, 0, 6, 7, 8, 9, 10, 0, 0, 11, 12, 13, 14, 15, 0, 0, 16, 17, 18, 19, 20, 0, 0, 0, 0, 0, 0, 0, 0 };
	Tensor<qint32_t> R{ 1, 1, 6, 7 };
	R.allocateCustom(R_data, 42);

    EML_ASSERT_TENSOR_EQUALS(R, out);
}
inline int testZeroPad2DCase4()
{
    qint32_t A_data[56]{} ;
	Tensor<qint32_t> A{ 7, 8 };
	A.allocateCustom(A_data, 56);
	arange(A,  1);

    ZeroPad2D layer{ { 2,3 } };
    auto out = layer.forward(A);
    qint32_t R_data[] = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 0, 0, 0, 0, 0, 9, 10, 11, 12, 13, 14, 15, 16, 0, 0, 0, 0, 0, 0, 17, 18, 19, 20, 21, 22, 23, 24, 0, 0, 0, 0, 0, 0, 25, 26, 27, 28, 29, 30, 31, 32, 0, 0, 0, 0, 0, 0, 33, 34, 35, 36, 37, 38, 39, 40, 0, 0, 0, 0, 0, 0, 41, 42, 43, 44, 45, 46, 47, 48, 0, 0, 0, 0, 0, 0, 49, 50, 51, 52, 53, 54, 55, 56, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
	Tensor<qint32_t> R{ 1, 1, 11, 14 };
	R.allocateCustom(R_data, 154);

    EML_ASSERT_TENSOR_EQUALS(R, out);
}
inline int testZeroPad2DCase5()
{
    qint32_t A_data[20]{} ;
	Tensor<qint32_t> A{ 5, 4 };
	A.allocateCustom(A_data, 20);
	arange(A,  1);

    ZeroPad2D layer{ { 2,3 } };
    auto out = layer.forward(A);
    qint32_t R_data[] = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 0, 0, 0, 0, 0, 0, 5, 6, 7, 8, 0, 0, 0, 0, 0, 0, 9, 10, 11, 12, 0, 0, 0, 0, 0, 0, 13, 14, 15, 16, 0, 0, 0, 0, 0, 0, 17, 18, 19, 20, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
	Tensor<qint32_t> R{ 1, 1, 9, 10 };
	R.allocateCustom(R_data, 90);

    EML_ASSERT_TENSOR_EQUALS(R, out);
}
inline int testZeroPad2DCase6()
{
    qint32_t A_data[36]{} ;
	Tensor<qint32_t> A{ 6, 6 };
	A.allocateCustom(A_data, 36);
	arange(A,  1);

    ZeroPad2D layer{ { 2,3 } };
    auto out = layer.forward(A);
    qint32_t R_data[] = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 0, 0, 0, 0, 0, 0, 7, 8, 9, 10, 11, 12, 0, 0, 0, 0, 0, 0, 13, 14, 15, 16, 17, 18, 0, 0, 0, 0, 0, 0, 19, 20, 21, 22, 23, 24, 0, 0, 0, 0, 0, 0, 25, 26, 27, 28, 29, 30, 0, 0, 0, 0, 0, 0, 31, 32, 33, 34, 35, 36, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
	Tensor<qint32_t> R{ 1, 1, 10, 12 };
	R.allocateCustom(R_data, 120);

    EML_ASSERT_TENSOR_EQUALS(R, out);
}
inline int testZeroPad2DCase7()
{
    qint32_t A_data[36]{} ;
	Tensor<qint32_t> A{ 6, 6 };
	A.allocateCustom(A_data, 36);
	arange(A,  1);

    ZeroPad2D layer{ { 2,3 } };
    auto out = layer.forward(A);
    qint32_t R_data[] = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 0, 0, 0, 0, 0, 0, 7, 8, 9, 10, 11, 12, 0, 0, 0, 0, 0, 0, 13, 14, 15, 16, 17, 18, 0, 0, 0, 0, 0, 0, 19, 20, 21, 22, 23, 24, 0, 0, 0, 0, 0, 0, 25, 26, 27, 28, 29, 30, 0, 0, 0, 0, 0, 0, 31, 32, 33, 34, 35, 36, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
	Tensor<qint32_t> R{ 1, 1, 10, 12 };
	R.allocateCustom(R_data, 120);

    EML_ASSERT_TENSOR_EQUALS(R, out);
}
inline int testZeroPad2DCase8()
{
    qint32_t A_data[20]{} ;
	Tensor<qint32_t> A{ 4, 5 };
	A.allocateCustom(A_data, 20);
	arange(A,  1);

    ZeroPad2D layer{ { 2,3 } };
    auto out = layer.forward(A);
    qint32_t R_data[] = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 5, 0, 0, 0, 0, 0, 0, 6, 7, 8, 9, 10, 0, 0, 0, 0, 0, 0, 11, 12, 13, 14, 15, 0, 0, 0, 0, 0, 0, 16, 17, 18, 19, 20, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
	Tensor<qint32_t> R{ 1, 1, 8, 11 };
	R.allocateCustom(R_data, 88);

    EML_ASSERT_TENSOR_EQUALS(R, out);
}
inline int testZeroPad2DCase9()
{
    qint32_t A_data[20]{} ;
	Tensor<qint32_t> A{ 4, 5 };
	A.allocateCustom(A_data, 20);
	arange(A,  1);

    ZeroPad2D layer{ { 2,2 } };
    auto out = layer.forward(A);
    qint32_t R_data[] = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 5, 0, 0, 0, 0, 6, 7, 8, 9, 10, 0, 0, 0, 0, 11, 12, 13, 14, 15, 0, 0, 0, 0, 16, 17, 18, 19, 20, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
	Tensor<qint32_t> R{ 1, 1, 8, 9 };
	R.allocateCustom(R_data, 72);

    EML_ASSERT_TENSOR_EQUALS(R, out);
}
inline int testZeroPad2DCase10()
{
    qint32_t A_data[56]{} ;
	Tensor<qint32_t> A{ 7, 8 };
	A.allocateCustom(A_data, 56);
	arange(A,  1);

    ZeroPad2D layer{ { 2,3 } };
    auto out = layer.forward(A);
    qint32_t R_data[] = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 0, 0, 0, 0, 0, 9, 10, 11, 12, 13, 14, 15, 16, 0, 0, 0, 0, 0, 0, 17, 18, 19, 20, 21, 22, 23, 24, 0, 0, 0, 0, 0, 0, 25, 26, 27, 28, 29, 30, 31, 32, 0, 0, 0, 0, 0, 0, 33, 34, 35, 36, 37, 38, 39, 40, 0, 0, 0, 0, 0, 0, 41, 42, 43, 44, 45, 46, 47, 48, 0, 0, 0, 0, 0, 0, 49, 50, 51, 52, 53, 54, 55, 56, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
	Tensor<qint32_t> R{ 1, 1, 11, 14 };
	R.allocateCustom(R_data, 154);

    EML_ASSERT_TENSOR_EQUALS(R, out);
}
inline int testZeroPad2DCase11()
{
    qint32_t A_data[56]{} ;
	Tensor<qint32_t> A{ 7, 8 };
	A.allocateCustom(A_data, 56);
	arange(A,  1);

    ZeroPad2D layer{ { 2,2 } };
    auto out = layer.forward(A);
    qint32_t R_data[] = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 0, 0, 0, 9, 10, 11, 12, 13, 14, 15, 16, 0, 0, 0, 0, 17, 18, 19, 20, 21, 22, 23, 24, 0, 0, 0, 0, 25, 26, 27, 28, 29, 30, 31, 32, 0, 0, 0, 0, 33, 34, 35, 36, 37, 38, 39, 40, 0, 0, 0, 0, 41, 42, 43, 44, 45, 46, 47, 48, 0, 0, 0, 0, 49, 50, 51, 52, 53, 54, 55, 56, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
	Tensor<qint32_t> R{ 1, 1, 11, 12 };
	R.allocateCustom(R_data, 132);

    EML_ASSERT_TENSOR_EQUALS(R, out);
}
inline int testZeroPad2DCase12()
{
    qint32_t A_data[36]{} ;
	Tensor<qint32_t> A{ 6, 6 };
	A.allocateCustom(A_data, 36);
	arange(A,  1);

    ZeroPad2D layer{ { 2,3 } };
    auto out = layer.forward(A);
    qint32_t R_data[] = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 0, 0, 0, 0, 0, 0, 7, 8, 9, 10, 11, 12, 0, 0, 0, 0, 0, 0, 13, 14, 15, 16, 17, 18, 0, 0, 0, 0, 0, 0, 19, 20, 21, 22, 23, 24, 0, 0, 0, 0, 0, 0, 25, 26, 27, 28, 29, 30, 0, 0, 0, 0, 0, 0, 31, 32, 33, 34, 35, 36, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
	Tensor<qint32_t> R{ 1, 1, 10, 12 };
	R.allocateCustom(R_data, 120);

    EML_ASSERT_TENSOR_EQUALS(R, out);
}
inline int testZeroPad2DCase13()
{
    qint32_t A_data[20]{} ;
	Tensor<qint32_t> A{ 5, 4 };
	A.allocateCustom(A_data, 20);
	arange(A,  1);

    ZeroPad2D layer{ { 2,2 } };
    auto out = layer.forward(A);
    qint32_t R_data[] = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 0, 0, 0, 0, 5, 6, 7, 8, 0, 0, 0, 0, 9, 10, 11, 12, 0, 0, 0, 0, 13, 14, 15, 16, 0, 0, 0, 0, 17, 18, 19, 20, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
	Tensor<qint32_t> R{ 1, 1, 9, 8 };
	R.allocateCustom(R_data, 72);

    EML_ASSERT_TENSOR_EQUALS(R, out);
}
inline int testZeroPad2DCase14()
{
    qint32_t A_data[56]{} ;
	Tensor<qint32_t> A{ 7, 8 };
	A.allocateCustom(A_data, 56);
	arange(A,  1);

    ZeroPad2D layer{ { 2,2 } };
    auto out = layer.forward(A);
    qint32_t R_data[] = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 0, 0, 0, 9, 10, 11, 12, 13, 14, 15, 16, 0, 0, 0, 0, 17, 18, 19, 20, 21, 22, 23, 24, 0, 0, 0, 0, 25, 26, 27, 28, 29, 30, 31, 32, 0, 0, 0, 0, 33, 34, 35, 36, 37, 38, 39, 40, 0, 0, 0, 0, 41, 42, 43, 44, 45, 46, 47, 48, 0, 0, 0, 0, 49, 50, 51, 52, 53, 54, 55, 56, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
	Tensor<qint32_t> R{ 1, 1, 11, 12 };
	R.allocateCustom(R_data, 132);

    EML_ASSERT_TENSOR_EQUALS(R, out);
}
inline int testZeroPad2DCase15()
{
    qint32_t A_data[36]{} ;
	Tensor<qint32_t> A{ 6, 6 };
	A.allocateCustom(A_data, 36);
	arange(A,  1);

    ZeroPad2D layer{ { 2,2 } };
    auto out = layer.forward(A);
    qint32_t R_data[] = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 0, 0, 0, 0, 7, 8, 9, 10, 11, 12, 0, 0, 0, 0, 13, 14, 15, 16, 17, 18, 0, 0, 0, 0, 19, 20, 21, 22, 23, 24, 0, 0, 0, 0, 25, 26, 27, 28, 29, 30, 0, 0, 0, 0, 31, 32, 33, 34, 35, 36, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
	Tensor<qint32_t> R{ 1, 1, 10, 10 };
	R.allocateCustom(R_data, 100);

    EML_ASSERT_TENSOR_EQUALS(R, out);
}

inline void TestNNZeroPad2D()
{	EML_RUN_TEST( testZeroPad2DCase0 );
	EML_RUN_TEST( testZeroPad2DCase1 );
	EML_RUN_TEST( testZeroPad2DCase2 );
	EML_RUN_TEST( testZeroPad2DCase3 );
	EML_RUN_TEST( testZeroPad2DCase4 );
	EML_RUN_TEST( testZeroPad2DCase5 );
	EML_RUN_TEST( testZeroPad2DCase6 );
	EML_RUN_TEST( testZeroPad2DCase7 );
	EML_RUN_TEST( testZeroPad2DCase8 );
	EML_RUN_TEST( testZeroPad2DCase9 );
	EML_RUN_TEST( testZeroPad2DCase10 );
	EML_RUN_TEST( testZeroPad2DCase11 );
	EML_RUN_TEST( testZeroPad2DCase12 );
	EML_RUN_TEST( testZeroPad2DCase13 );
	EML_RUN_TEST( testZeroPad2DCase14 );
	EML_RUN_TEST( testZeroPad2DCase15 );

}

#endif // EML_TEST_NN_LAYERS_ZEROPAD2D_H
    