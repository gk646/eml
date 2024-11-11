
#ifndef EML_TEST_NN_LAYERS_REFLECTIONPAD2D_H
#define EML_TEST_NN_LAYERS_REFLECTIONPAD2D_H

inline int testReflectionPad2DCase0()
{
    qint32_t A_data[20] ;
	Tensor<qint32_t> A{ 5, 4 };
	A.allocateCustom(A_data, 20);
	Arange(A,1);

    ReflectionPad2D layer{ { 2,2 } };
    auto out = layer.forward(A);
    qint32_t R_data[] = { 11, 10, 9, 10, 11, 12, 11, 10, 7, 6, 5, 6, 7, 8, 7, 6, 3, 2, 1, 2, 3, 4, 3, 2, 7, 6, 5, 6, 7, 8, 7, 6, 11, 10, 9, 10, 11, 12, 11, 10, 15, 14, 13, 14, 15, 16, 15, 14, 19, 18, 17, 18, 19, 20, 19, 18, 15, 14, 13, 14, 15, 16, 15, 14, 11, 10, 9, 10, 11, 12, 11, 10 };
	Tensor<qint32_t> R{ 1, 1, 9, 8 };
	R.allocateCustom(R_data, 72);

     EML_ASSERT_TENSOR_EQUALS(R, out);
}
inline int testReflectionPad2DCase1()
{
    qint32_t A_data[36] ;
	Tensor<qint32_t> A{ 6, 6 };
	A.allocateCustom(A_data, 36);
	Arange(A,1);

    ReflectionPad2D layer{ { 2,3 } };
    auto out = layer.forward(A);
    qint32_t R_data[] = { 16, 15, 14, 13, 14, 15, 16, 17, 18, 17, 16, 15, 10, 9, 8, 7, 8, 9, 10, 11, 12, 11, 10, 9, 4, 3, 2, 1, 2, 3, 4, 5, 6, 5, 4, 3, 10, 9, 8, 7, 8, 9, 10, 11, 12, 11, 10, 9, 16, 15, 14, 13, 14, 15, 16, 17, 18, 17, 16, 15, 22, 21, 20, 19, 20, 21, 22, 23, 24, 23, 22, 21, 28, 27, 26, 25, 26, 27, 28, 29, 30, 29, 28, 27, 34, 33, 32, 31, 32, 33, 34, 35, 36, 35, 34, 33, 28, 27, 26, 25, 26, 27, 28, 29, 30, 29, 28, 27, 22, 21, 20, 19, 20, 21, 22, 23, 24, 23, 22, 21 };
	Tensor<qint32_t> R{ 1, 1, 10, 12 };
	R.allocateCustom(R_data, 120);

     EML_ASSERT_TENSOR_EQUALS(R, out);
}
inline int testReflectionPad2DCase2()
{
    qint32_t A_data[36] ;
	Tensor<qint32_t> A{ 6, 6 };
	A.allocateCustom(A_data, 36);
	Arange(A,1);

    ReflectionPad2D layer{ { 2,2 } };
    auto out = layer.forward(A);
    qint32_t R_data[] = { 15, 14, 13, 14, 15, 16, 17, 18, 17, 16, 9, 8, 7, 8, 9, 10, 11, 12, 11, 10, 3, 2, 1, 2, 3, 4, 5, 6, 5, 4, 9, 8, 7, 8, 9, 10, 11, 12, 11, 10, 15, 14, 13, 14, 15, 16, 17, 18, 17, 16, 21, 20, 19, 20, 21, 22, 23, 24, 23, 22, 27, 26, 25, 26, 27, 28, 29, 30, 29, 28, 33, 32, 31, 32, 33, 34, 35, 36, 35, 34, 27, 26, 25, 26, 27, 28, 29, 30, 29, 28, 21, 20, 19, 20, 21, 22, 23, 24, 23, 22 };
	Tensor<qint32_t> R{ 1, 1, 10, 10 };
	R.allocateCustom(R_data, 100);

     EML_ASSERT_TENSOR_EQUALS(R, out);
}
inline int testReflectionPad2DCase3()
{
    qint32_t A_data[20] ;
	Tensor<qint32_t> A{ 4, 5 };
	A.allocateCustom(A_data, 20);
	Arange(A,1);

    ReflectionPad2D layer{ { 2,3 } };
    auto out = layer.forward(A);
    qint32_t R_data[] = { 14, 13, 12, 11, 12, 13, 14, 15, 14, 13, 12, 9, 8, 7, 6, 7, 8, 9, 10, 9, 8, 7, 4, 3, 2, 1, 2, 3, 4, 5, 4, 3, 2, 9, 8, 7, 6, 7, 8, 9, 10, 9, 8, 7, 14, 13, 12, 11, 12, 13, 14, 15, 14, 13, 12, 19, 18, 17, 16, 17, 18, 19, 20, 19, 18, 17, 14, 13, 12, 11, 12, 13, 14, 15, 14, 13, 12, 9, 8, 7, 6, 7, 8, 9, 10, 9, 8, 7 };
	Tensor<qint32_t> R{ 1, 1, 8, 11 };
	R.allocateCustom(R_data, 88);

     EML_ASSERT_TENSOR_EQUALS(R, out);
}
inline int testReflectionPad2DCase4()
{
    qint32_t A_data[56] ;
	Tensor<qint32_t> A{ 7, 8 };
	A.allocateCustom(A_data, 56);
	Arange(A,1);

    ReflectionPad2D layer{ { 1,0 } };
    auto out = layer.forward(A);
    qint32_t R_data[] = { 9, 10, 11, 12, 13, 14, 15, 16, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 41, 42, 43, 44, 45, 46, 47, 48 };
	Tensor<qint32_t> R{ 1, 1, 9, 8 };
	R.allocateCustom(R_data, 72);

     EML_ASSERT_TENSOR_EQUALS(R, out);
}
inline int testReflectionPad2DCase5()
{
    qint32_t A_data[20] ;
	Tensor<qint32_t> A{ 4, 5 };
	A.allocateCustom(A_data, 20);
	Arange(A,1);

    ReflectionPad2D layer{ { 2,3 } };
    auto out = layer.forward(A);
    qint32_t R_data[] = { 14, 13, 12, 11, 12, 13, 14, 15, 14, 13, 12, 9, 8, 7, 6, 7, 8, 9, 10, 9, 8, 7, 4, 3, 2, 1, 2, 3, 4, 5, 4, 3, 2, 9, 8, 7, 6, 7, 8, 9, 10, 9, 8, 7, 14, 13, 12, 11, 12, 13, 14, 15, 14, 13, 12, 19, 18, 17, 16, 17, 18, 19, 20, 19, 18, 17, 14, 13, 12, 11, 12, 13, 14, 15, 14, 13, 12, 9, 8, 7, 6, 7, 8, 9, 10, 9, 8, 7 };
	Tensor<qint32_t> R{ 1, 1, 8, 11 };
	R.allocateCustom(R_data, 88);

     EML_ASSERT_TENSOR_EQUALS(R, out);
}
inline int testReflectionPad2DCase6()
{
    qint32_t A_data[20] ;
	Tensor<qint32_t> A{ 4, 5 };
	A.allocateCustom(A_data, 20);
	Arange(A,1);

    ReflectionPad2D layer{ { 1,0 } };
    auto out = layer.forward(A);
    qint32_t R_data[] = { 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 11, 12, 13, 14, 15 };
	Tensor<qint32_t> R{ 1, 1, 6, 5 };
	R.allocateCustom(R_data, 30);

     EML_ASSERT_TENSOR_EQUALS(R, out);
}
inline int testReflectionPad2DCase7()
{
    qint32_t A_data[20] ;
	Tensor<qint32_t> A{ 5, 4 };
	A.allocateCustom(A_data, 20);
	Arange(A,1);

    ReflectionPad2D layer{ { 2,2 } };
    auto out = layer.forward(A);
    qint32_t R_data[] = { 11, 10, 9, 10, 11, 12, 11, 10, 7, 6, 5, 6, 7, 8, 7, 6, 3, 2, 1, 2, 3, 4, 3, 2, 7, 6, 5, 6, 7, 8, 7, 6, 11, 10, 9, 10, 11, 12, 11, 10, 15, 14, 13, 14, 15, 16, 15, 14, 19, 18, 17, 18, 19, 20, 19, 18, 15, 14, 13, 14, 15, 16, 15, 14, 11, 10, 9, 10, 11, 12, 11, 10 };
	Tensor<qint32_t> R{ 1, 1, 9, 8 };
	R.allocateCustom(R_data, 72);

     EML_ASSERT_TENSOR_EQUALS(R, out);
}
inline int testReflectionPad2DCase8()
{
    qint32_t A_data[36] ;
	Tensor<qint32_t> A{ 6, 6 };
	A.allocateCustom(A_data, 36);
	Arange(A,1);

    ReflectionPad2D layer{ { 2,3 } };
    auto out = layer.forward(A);
    qint32_t R_data[] = { 16, 15, 14, 13, 14, 15, 16, 17, 18, 17, 16, 15, 10, 9, 8, 7, 8, 9, 10, 11, 12, 11, 10, 9, 4, 3, 2, 1, 2, 3, 4, 5, 6, 5, 4, 3, 10, 9, 8, 7, 8, 9, 10, 11, 12, 11, 10, 9, 16, 15, 14, 13, 14, 15, 16, 17, 18, 17, 16, 15, 22, 21, 20, 19, 20, 21, 22, 23, 24, 23, 22, 21, 28, 27, 26, 25, 26, 27, 28, 29, 30, 29, 28, 27, 34, 33, 32, 31, 32, 33, 34, 35, 36, 35, 34, 33, 28, 27, 26, 25, 26, 27, 28, 29, 30, 29, 28, 27, 22, 21, 20, 19, 20, 21, 22, 23, 24, 23, 22, 21 };
	Tensor<qint32_t> R{ 1, 1, 10, 12 };
	R.allocateCustom(R_data, 120);

     EML_ASSERT_TENSOR_EQUALS(R, out);
}
inline int testReflectionPad2DCase9()
{
    qint32_t A_data[36] ;
	Tensor<qint32_t> A{ 6, 6 };
	A.allocateCustom(A_data, 36);
	Arange(A,1);

    ReflectionPad2D layer{ { 2,3 } };
    auto out = layer.forward(A);
    qint32_t R_data[] = { 16, 15, 14, 13, 14, 15, 16, 17, 18, 17, 16, 15, 10, 9, 8, 7, 8, 9, 10, 11, 12, 11, 10, 9, 4, 3, 2, 1, 2, 3, 4, 5, 6, 5, 4, 3, 10, 9, 8, 7, 8, 9, 10, 11, 12, 11, 10, 9, 16, 15, 14, 13, 14, 15, 16, 17, 18, 17, 16, 15, 22, 21, 20, 19, 20, 21, 22, 23, 24, 23, 22, 21, 28, 27, 26, 25, 26, 27, 28, 29, 30, 29, 28, 27, 34, 33, 32, 31, 32, 33, 34, 35, 36, 35, 34, 33, 28, 27, 26, 25, 26, 27, 28, 29, 30, 29, 28, 27, 22, 21, 20, 19, 20, 21, 22, 23, 24, 23, 22, 21 };
	Tensor<qint32_t> R{ 1, 1, 10, 12 };
	R.allocateCustom(R_data, 120);

     EML_ASSERT_TENSOR_EQUALS(R, out);
}
inline int testReflectionPad2DCase10()
{
    qint32_t A_data[36] ;
	Tensor<qint32_t> A{ 6, 6 };
	A.allocateCustom(A_data, 36);
	Arange(A,1);

    ReflectionPad2D layer{ { 1,0 } };
    auto out = layer.forward(A);
    qint32_t R_data[] = { 7, 8, 9, 10, 11, 12, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 25, 26, 27, 28, 29, 30 };
	Tensor<qint32_t> R{ 1, 1, 8, 6 };
	R.allocateCustom(R_data, 48);

     EML_ASSERT_TENSOR_EQUALS(R, out);
}
inline int testReflectionPad2DCase11()
{
    qint32_t A_data[20] ;
	Tensor<qint32_t> A{ 4, 5 };
	A.allocateCustom(A_data, 20);
	Arange(A,1);

    ReflectionPad2D layer{ { 1,1 } };
    auto out = layer.forward(A);
    qint32_t R_data[] = { 7, 6, 7, 8, 9, 10, 9, 2, 1, 2, 3, 4, 5, 4, 7, 6, 7, 8, 9, 10, 9, 12, 11, 12, 13, 14, 15, 14, 17, 16, 17, 18, 19, 20, 19, 12, 11, 12, 13, 14, 15, 14 };
	Tensor<qint32_t> R{ 1, 1, 6, 7 };
	R.allocateCustom(R_data, 42);

     EML_ASSERT_TENSOR_EQUALS(R, out);
}
inline int testReflectionPad2DCase12()
{
    qint32_t A_data[20] ;
	Tensor<qint32_t> A{ 5, 4 };
	A.allocateCustom(A_data, 20);
	Arange(A,1);

    ReflectionPad2D layer{ { 1,1 } };
    auto out = layer.forward(A);
    qint32_t R_data[] = { 6, 5, 6, 7, 8, 7, 2, 1, 2, 3, 4, 3, 6, 5, 6, 7, 8, 7, 10, 9, 10, 11, 12, 11, 14, 13, 14, 15, 16, 15, 18, 17, 18, 19, 20, 19, 14, 13, 14, 15, 16, 15 };
	Tensor<qint32_t> R{ 1, 1, 7, 6 };
	R.allocateCustom(R_data, 42);

     EML_ASSERT_TENSOR_EQUALS(R, out);
}
inline int testReflectionPad2DCase13()
{
    qint32_t A_data[20] ;
	Tensor<qint32_t> A{ 4, 5 };
	A.allocateCustom(A_data, 20);
	Arange(A,1);

    ReflectionPad2D layer{ { 1,0 } };
    auto out = layer.forward(A);
    qint32_t R_data[] = { 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 11, 12, 13, 14, 15 };
	Tensor<qint32_t> R{ 1, 1, 6, 5 };
	R.allocateCustom(R_data, 30);

     EML_ASSERT_TENSOR_EQUALS(R, out);
}
inline int testReflectionPad2DCase14()
{
    qint32_t A_data[20] ;
	Tensor<qint32_t> A{ 4, 5 };
	A.allocateCustom(A_data, 20);
	Arange(A,1);

    ReflectionPad2D layer{ { 1,1 } };
    auto out = layer.forward(A);
    qint32_t R_data[] = { 7, 6, 7, 8, 9, 10, 9, 2, 1, 2, 3, 4, 5, 4, 7, 6, 7, 8, 9, 10, 9, 12, 11, 12, 13, 14, 15, 14, 17, 16, 17, 18, 19, 20, 19, 12, 11, 12, 13, 14, 15, 14 };
	Tensor<qint32_t> R{ 1, 1, 6, 7 };
	R.allocateCustom(R_data, 42);

     EML_ASSERT_TENSOR_EQUALS(R, out);
}
inline int testReflectionPad2DCase15()
{
    qint32_t A_data[20] ;
	Tensor<qint32_t> A{ 4, 5 };
	A.allocateCustom(A_data, 20);
	Arange(A,1);

    ReflectionPad2D layer{ { 1,1 } };
    auto out = layer.forward(A);
    qint32_t R_data[] = { 7, 6, 7, 8, 9, 10, 9, 2, 1, 2, 3, 4, 5, 4, 7, 6, 7, 8, 9, 10, 9, 12, 11, 12, 13, 14, 15, 14, 17, 16, 17, 18, 19, 20, 19, 12, 11, 12, 13, 14, 15, 14 };
	Tensor<qint32_t> R{ 1, 1, 6, 7 };
	R.allocateCustom(R_data, 42);

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
    