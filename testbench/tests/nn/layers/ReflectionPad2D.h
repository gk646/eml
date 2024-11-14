
#ifndef EML_TEST_NN_LAYERS_REFLECTIONPAD2D_H
#define EML_TEST_NN_LAYERS_REFLECTIONPAD2D_H

inline int testReflectionPad2DCase0()
{
    qint32_t A_data[36]{} ;
	Tensor<qint32_t> A{ 6, 6 };
	A.allocateCustom(A_data, 36);
	arange(A,  1);

    ReflectionPad2D layer{ { 1,0 } };
    auto out = layer.forward(A);
    qint32_t R_data[] = { 7, 8, 9, 10, 11, 12, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 25, 26, 27, 28, 29, 30 };
	Tensor<qint32_t> R{ 1, 1, 8, 6 };
	R.allocateCustom(R_data, 48);

     EML_ASSERT_TENSOR_EQUALS(R, out);
}
inline int testReflectionPad2DCase1()
{
    qint32_t A_data[56]{} ;
	Tensor<qint32_t> A{ 7, 8 };
	A.allocateCustom(A_data, 56);
	arange(A,  1);

    ReflectionPad2D layer{ { 1,0 } };
    auto out = layer.forward(A);
    qint32_t R_data[] = { 9, 10, 11, 12, 13, 14, 15, 16, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 41, 42, 43, 44, 45, 46, 47, 48 };
	Tensor<qint32_t> R{ 1, 1, 9, 8 };
	R.allocateCustom(R_data, 72);

     EML_ASSERT_TENSOR_EQUALS(R, out);
}
inline int testReflectionPad2DCase2()
{
    qint32_t A_data[20]{} ;
	Tensor<qint32_t> A{ 5, 4 };
	A.allocateCustom(A_data, 20);
	arange(A,  1);

    ReflectionPad2D layer{ { 2,2 } };
    auto out = layer.forward(A);
    qint32_t R_data[] = { 11, 10, 9, 10, 11, 12, 11, 10, 7, 6, 5, 6, 7, 8, 7, 6, 3, 2, 1, 2, 3, 4, 3, 2, 7, 6, 5, 6, 7, 8, 7, 6, 11, 10, 9, 10, 11, 12, 11, 10, 15, 14, 13, 14, 15, 16, 15, 14, 19, 18, 17, 18, 19, 20, 19, 18, 15, 14, 13, 14, 15, 16, 15, 14, 11, 10, 9, 10, 11, 12, 11, 10 };
	Tensor<qint32_t> R{ 1, 1, 9, 8 };
	R.allocateCustom(R_data, 72);

     EML_ASSERT_TENSOR_EQUALS(R, out);
}
inline int testReflectionPad2DCase3()
{
    qint32_t A_data[20]{} ;
	Tensor<qint32_t> A{ 5, 4 };
	A.allocateCustom(A_data, 20);
	arange(A,  1);

    ReflectionPad2D layer{ { 2,3 } };
    auto out = layer.forward(A);
    qint32_t R_data[] = { 12, 11, 10, 9, 10, 11, 12, 11, 10, 9, 8, 7, 6, 5, 6, 7, 8, 7, 6, 5, 4, 3, 2, 1, 2, 3, 4, 3, 2, 1, 8, 7, 6, 5, 6, 7, 8, 7, 6, 5, 12, 11, 10, 9, 10, 11, 12, 11, 10, 9, 16, 15, 14, 13, 14, 15, 16, 15, 14, 13, 20, 19, 18, 17, 18, 19, 20, 19, 18, 17, 16, 15, 14, 13, 14, 15, 16, 15, 14, 13, 12, 11, 10, 9, 10, 11, 12, 11, 10, 9 };
	Tensor<qint32_t> R{ 1, 1, 9, 10 };
	R.allocateCustom(R_data, 90);

     EML_ASSERT_TENSOR_EQUALS(R, out);
}
inline int testReflectionPad2DCase4()
{
    qint32_t A_data[36]{} ;
	Tensor<qint32_t> A{ 6, 6 };
	A.allocateCustom(A_data, 36);
	arange(A,  1);

    ReflectionPad2D layer{ { 1,0 } };
    auto out = layer.forward(A);
    qint32_t R_data[] = { 7, 8, 9, 10, 11, 12, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 25, 26, 27, 28, 29, 30 };
	Tensor<qint32_t> R{ 1, 1, 8, 6 };
	R.allocateCustom(R_data, 48);

     EML_ASSERT_TENSOR_EQUALS(R, out);
}
inline int testReflectionPad2DCase5()
{
    qint32_t A_data[56]{} ;
	Tensor<qint32_t> A{ 7, 8 };
	A.allocateCustom(A_data, 56);
	arange(A,  1);

    ReflectionPad2D layer{ { 2,3 } };
    auto out = layer.forward(A);
    qint32_t R_data[] = { 20, 19, 18, 17, 18, 19, 20, 21, 22, 23, 24, 23, 22, 21, 12, 11, 10, 9, 10, 11, 12, 13, 14, 15, 16, 15, 14, 13, 4, 3, 2, 1, 2, 3, 4, 5, 6, 7, 8, 7, 6, 5, 12, 11, 10, 9, 10, 11, 12, 13, 14, 15, 16, 15, 14, 13, 20, 19, 18, 17, 18, 19, 20, 21, 22, 23, 24, 23, 22, 21, 28, 27, 26, 25, 26, 27, 28, 29, 30, 31, 32, 31, 30, 29, 36, 35, 34, 33, 34, 35, 36, 37, 38, 39, 40, 39, 38, 37, 44, 43, 42, 41, 42, 43, 44, 45, 46, 47, 48, 47, 46, 45, 52, 51, 50, 49, 50, 51, 52, 53, 54, 55, 56, 55, 54, 53, 44, 43, 42, 41, 42, 43, 44, 45, 46, 47, 48, 47, 46, 45, 36, 35, 34, 33, 34, 35, 36, 37, 38, 39, 40, 39, 38, 37 };
	Tensor<qint32_t> R{ 1, 1, 11, 14 };
	R.allocateCustom(R_data, 154);

     EML_ASSERT_TENSOR_EQUALS(R, out);
}
inline int testReflectionPad2DCase6()
{
    qint32_t A_data[20]{} ;
	Tensor<qint32_t> A{ 5, 4 };
	A.allocateCustom(A_data, 20);
	arange(A,  1);

    ReflectionPad2D layer{ { 1,0 } };
    auto out = layer.forward(A);
    qint32_t R_data[] = { 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 13, 14, 15, 16 };
	Tensor<qint32_t> R{ 1, 1, 7, 4 };
	R.allocateCustom(R_data, 28);

     EML_ASSERT_TENSOR_EQUALS(R, out);
}
inline int testReflectionPad2DCase7()
{
    qint32_t A_data[56]{} ;
	Tensor<qint32_t> A{ 7, 8 };
	A.allocateCustom(A_data, 56);
	arange(A,  1);

    ReflectionPad2D layer{ { 1,0 } };
    auto out = layer.forward(A);
    qint32_t R_data[] = { 9, 10, 11, 12, 13, 14, 15, 16, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 41, 42, 43, 44, 45, 46, 47, 48 };
	Tensor<qint32_t> R{ 1, 1, 9, 8 };
	R.allocateCustom(R_data, 72);

     EML_ASSERT_TENSOR_EQUALS(R, out);
}
inline int testReflectionPad2DCase8()
{
    qint32_t A_data[56]{} ;
	Tensor<qint32_t> A{ 7, 8 };
	A.allocateCustom(A_data, 56);
	arange(A,  1);

    ReflectionPad2D layer{ { 2,2 } };
    auto out = layer.forward(A);
    qint32_t R_data[] = { 19, 18, 17, 18, 19, 20, 21, 22, 23, 24, 23, 22, 11, 10, 9, 10, 11, 12, 13, 14, 15, 16, 15, 14, 3, 2, 1, 2, 3, 4, 5, 6, 7, 8, 7, 6, 11, 10, 9, 10, 11, 12, 13, 14, 15, 16, 15, 14, 19, 18, 17, 18, 19, 20, 21, 22, 23, 24, 23, 22, 27, 26, 25, 26, 27, 28, 29, 30, 31, 32, 31, 30, 35, 34, 33, 34, 35, 36, 37, 38, 39, 40, 39, 38, 43, 42, 41, 42, 43, 44, 45, 46, 47, 48, 47, 46, 51, 50, 49, 50, 51, 52, 53, 54, 55, 56, 55, 54, 43, 42, 41, 42, 43, 44, 45, 46, 47, 48, 47, 46, 35, 34, 33, 34, 35, 36, 37, 38, 39, 40, 39, 38 };
	Tensor<qint32_t> R{ 1, 1, 11, 12 };
	R.allocateCustom(R_data, 132);

     EML_ASSERT_TENSOR_EQUALS(R, out);
}
inline int testReflectionPad2DCase9()
{
    qint32_t A_data[20]{} ;
	Tensor<qint32_t> A{ 4, 5 };
	A.allocateCustom(A_data, 20);
	arange(A,  1);

    ReflectionPad2D layer{ { 2,2 } };
    auto out = layer.forward(A);
    qint32_t R_data[] = { 13, 12, 11, 12, 13, 14, 15, 14, 13, 8, 7, 6, 7, 8, 9, 10, 9, 8, 3, 2, 1, 2, 3, 4, 5, 4, 3, 8, 7, 6, 7, 8, 9, 10, 9, 8, 13, 12, 11, 12, 13, 14, 15, 14, 13, 18, 17, 16, 17, 18, 19, 20, 19, 18, 13, 12, 11, 12, 13, 14, 15, 14, 13, 8, 7, 6, 7, 8, 9, 10, 9, 8 };
	Tensor<qint32_t> R{ 1, 1, 8, 9 };
	R.allocateCustom(R_data, 72);

     EML_ASSERT_TENSOR_EQUALS(R, out);
}
inline int testReflectionPad2DCase10()
{
    qint32_t A_data[20]{} ;
	Tensor<qint32_t> A{ 5, 4 };
	A.allocateCustom(A_data, 20);
	arange(A,  1);

    ReflectionPad2D layer{ { 1,0 } };
    auto out = layer.forward(A);
    qint32_t R_data[] = { 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 13, 14, 15, 16 };
	Tensor<qint32_t> R{ 1, 1, 7, 4 };
	R.allocateCustom(R_data, 28);

     EML_ASSERT_TENSOR_EQUALS(R, out);
}
inline int testReflectionPad2DCase11()
{
    qint32_t A_data[36]{} ;
	Tensor<qint32_t> A{ 6, 6 };
	A.allocateCustom(A_data, 36);
	arange(A,  1);

    ReflectionPad2D layer{ { 1,0 } };
    auto out = layer.forward(A);
    qint32_t R_data[] = { 7, 8, 9, 10, 11, 12, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 25, 26, 27, 28, 29, 30 };
	Tensor<qint32_t> R{ 1, 1, 8, 6 };
	R.allocateCustom(R_data, 48);

     EML_ASSERT_TENSOR_EQUALS(R, out);
}
inline int testReflectionPad2DCase12()
{
    qint32_t A_data[20]{} ;
	Tensor<qint32_t> A{ 4, 5 };
	A.allocateCustom(A_data, 20);
	arange(A,  1);

    ReflectionPad2D layer{ { 1,0 } };
    auto out = layer.forward(A);
    qint32_t R_data[] = { 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 11, 12, 13, 14, 15 };
	Tensor<qint32_t> R{ 1, 1, 6, 5 };
	R.allocateCustom(R_data, 30);

     EML_ASSERT_TENSOR_EQUALS(R, out);
}
inline int testReflectionPad2DCase13()
{
    qint32_t A_data[20]{} ;
	Tensor<qint32_t> A{ 5, 4 };
	A.allocateCustom(A_data, 20);
	arange(A,  1);

    ReflectionPad2D layer{ { 2,3 } };
    auto out = layer.forward(A);
    qint32_t R_data[] = { 12, 11, 10, 9, 10, 11, 12, 11, 10, 9, 8, 7, 6, 5, 6, 7, 8, 7, 6, 5, 4, 3, 2, 1, 2, 3, 4, 3, 2, 1, 8, 7, 6, 5, 6, 7, 8, 7, 6, 5, 12, 11, 10, 9, 10, 11, 12, 11, 10, 9, 16, 15, 14, 13, 14, 15, 16, 15, 14, 13, 20, 19, 18, 17, 18, 19, 20, 19, 18, 17, 16, 15, 14, 13, 14, 15, 16, 15, 14, 13, 12, 11, 10, 9, 10, 11, 12, 11, 10, 9 };
	Tensor<qint32_t> R{ 1, 1, 9, 10 };
	R.allocateCustom(R_data, 90);

     EML_ASSERT_TENSOR_EQUALS(R, out);
}
inline int testReflectionPad2DCase14()
{
    qint32_t A_data[36]{} ;
	Tensor<qint32_t> A{ 6, 6 };
	A.allocateCustom(A_data, 36);
	arange(A,  1);

    ReflectionPad2D layer{ { 2,3 } };
    auto out = layer.forward(A);
    qint32_t R_data[] = { 16, 15, 14, 13, 14, 15, 16, 17, 18, 17, 16, 15, 10, 9, 8, 7, 8, 9, 10, 11, 12, 11, 10, 9, 4, 3, 2, 1, 2, 3, 4, 5, 6, 5, 4, 3, 10, 9, 8, 7, 8, 9, 10, 11, 12, 11, 10, 9, 16, 15, 14, 13, 14, 15, 16, 17, 18, 17, 16, 15, 22, 21, 20, 19, 20, 21, 22, 23, 24, 23, 22, 21, 28, 27, 26, 25, 26, 27, 28, 29, 30, 29, 28, 27, 34, 33, 32, 31, 32, 33, 34, 35, 36, 35, 34, 33, 28, 27, 26, 25, 26, 27, 28, 29, 30, 29, 28, 27, 22, 21, 20, 19, 20, 21, 22, 23, 24, 23, 22, 21 };
	Tensor<qint32_t> R{ 1, 1, 10, 12 };
	R.allocateCustom(R_data, 120);

     EML_ASSERT_TENSOR_EQUALS(R, out);
}
inline int testReflectionPad2DCase15()
{
    qint32_t A_data[36]{} ;
	Tensor<qint32_t> A{ 6, 6 };
	A.allocateCustom(A_data, 36);
	arange(A,  1);

    ReflectionPad2D layer{ { 1,1 } };
    auto out = layer.forward(A);
    qint32_t R_data[] = { 8, 7, 8, 9, 10, 11, 12, 11, 2, 1, 2, 3, 4, 5, 6, 5, 8, 7, 8, 9, 10, 11, 12, 11, 14, 13, 14, 15, 16, 17, 18, 17, 20, 19, 20, 21, 22, 23, 24, 23, 26, 25, 26, 27, 28, 29, 30, 29, 32, 31, 32, 33, 34, 35, 36, 35, 26, 25, 26, 27, 28, 29, 30, 29 };
	Tensor<qint32_t> R{ 1, 1, 8, 8 };
	R.allocateCustom(R_data, 64);

     EML_ASSERT_TENSOR_EQUALS(R, out);
}

inline void TestNNReflectionPad2D()
{	EML_RUN_TEST( testReflectionPad2DCase0 );
	EML_RUN_TEST( testReflectionPad2DCase1 );
	EML_RUN_TEST( testReflectionPad2DCase2 );
	EML_RUN_TEST( testReflectionPad2DCase3 );
	EML_RUN_TEST( testReflectionPad2DCase4 );
	EML_RUN_TEST( testReflectionPad2DCase5 );
	EML_RUN_TEST( testReflectionPad2DCase6 );
	EML_RUN_TEST( testReflectionPad2DCase7 );
	EML_RUN_TEST( testReflectionPad2DCase8 );
	EML_RUN_TEST( testReflectionPad2DCase9 );
	EML_RUN_TEST( testReflectionPad2DCase10 );
	EML_RUN_TEST( testReflectionPad2DCase11 );
	EML_RUN_TEST( testReflectionPad2DCase12 );
	EML_RUN_TEST( testReflectionPad2DCase13 );
	EML_RUN_TEST( testReflectionPad2DCase14 );
	EML_RUN_TEST( testReflectionPad2DCase15 );

}

#endif // EML_TEST_NN_LAYERS_REFLECTIONPAD2D_H