
#ifndef EML_TEST_NN_LAYERS_LINEAR_H
#define EML_TEST_NN_LAYERS_LINEAR_H

inline int testLinearCase0()
{
    qint32_t A_data[4] ;
	Tensor<qint32_t> A{ 1, 4 };
	A.allocateCustom(A_data, 4);
	Arange(A,1);

    Linear<qint32_t> layer{ 4, 2, true };
    Fill(layer.weights, 1);
    Fill(layer.biases, 1);
    const auto out = layer.forward(A);
    
    qint32_t R_data[] = { 11, 11 };
	Tensor<qint32_t> R{ 2 };
	R.allocateCustom(R_data, 2);

    return Equals(out, R);
}
inline int testLinearCase1()
{
    qint32_t A_data[8] ;
	Tensor<qint32_t> A{ 1, 8 };
	A.allocateCustom(A_data, 8);
	Arange(A,1);

    Linear<qint32_t> layer{ 8, 8, false };
    Fill(layer.weights, 1);
    
    const auto out = layer.forward(A);
    
    qint32_t R_data[] = { 36, 36, 36, 36, 36, 36, 36, 36 };
	Tensor<qint32_t> R{ 8 };
	R.allocateCustom(R_data, 8);

    return Equals(out, R);
}
inline int testLinearCase2()
{
    qint32_t A_data[16] ;
	Tensor<qint32_t> A{ 1, 16 };
	A.allocateCustom(A_data, 16);
	Arange(A,1);

    Linear<qint32_t> layer{ 16, 20, true };
    Fill(layer.weights, 1);
    Fill(layer.biases, 1);
    const auto out = layer.forward(A);
    
    qint32_t R_data[] = { 137, 137, 137, 137, 137, 137, 137, 137, 137, 137, 137, 137, 137, 137, 137, 137, 137, 137, 137, 137 };
	Tensor<qint32_t> R{ 20 };
	R.allocateCustom(R_data, 20);

    return Equals(out, R);
}
inline int testLinearCase3()
{
    qint32_t A_data[4] ;
	Tensor<qint32_t> A{ 1, 4 };
	A.allocateCustom(A_data, 4);
	Arange(A,1);

    Linear<qint32_t> layer{ 4, 2, false };
    Fill(layer.weights, 1);
    
    const auto out = layer.forward(A);
    
    qint32_t R_data[] = { 10, 10 };
	Tensor<qint32_t> R{ 2 };
	R.allocateCustom(R_data, 2);

    return Equals(out, R);
}
inline int testLinearCase4()
{
    qint32_t A_data[8] ;
	Tensor<qint32_t> A{ 1, 8 };
	A.allocateCustom(A_data, 8);
	Arange(A,1);

    Linear<qint32_t> layer{ 8, 8, true };
    Fill(layer.weights, 1);
    Fill(layer.biases, 1);
    const auto out = layer.forward(A);
    
    qint32_t R_data[] = { 37, 37, 37, 37, 37, 37, 37, 37 };
	Tensor<qint32_t> R{ 8 };
	R.allocateCustom(R_data, 8);

    return Equals(out, R);
}
inline int testLinearCase5()
{
    qint32_t A_data[16] ;
	Tensor<qint32_t> A{ 1, 16 };
	A.allocateCustom(A_data, 16);
	Arange(A,1);

    Linear<qint32_t> layer{ 16, 20, false };
    Fill(layer.weights, 1);
    
    const auto out = layer.forward(A);
    
    qint32_t R_data[] = { 136, 136, 136, 136, 136, 136, 136, 136, 136, 136, 136, 136, 136, 136, 136, 136, 136, 136, 136, 136 };
	Tensor<qint32_t> R{ 20 };
	R.allocateCustom(R_data, 20);

    return Equals(out, R);
}
inline int testLinearCase6()
{
    qint32_t A_data[4] ;
	Tensor<qint32_t> A{ 1, 4 };
	A.allocateCustom(A_data, 4);
	Arange(A,1);

    Linear<qint32_t> layer{ 4, 2, true };
    Fill(layer.weights, 1);
    Fill(layer.biases, 1);
    const auto out = layer.forward(A);
    
    qint32_t R_data[] = { 11, 11 };
	Tensor<qint32_t> R{ 2 };
	R.allocateCustom(R_data, 2);

    return Equals(out, R);
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
    