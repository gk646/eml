
#ifndef EML_TEST_MATH_MATMUL_H
#define EML_TEST_MATH_MATMUL_H

inline int testMatMulCase0()
{
    qint32_t A_data[100]{} ;
	Tensor<qint32_t> A{ 10, 10 };
	A.allocateCustom(A_data, 100);
	arange(A,  1);

    qint32_t B_data[100]{} ;
	Tensor<qint32_t> B{ 10, 10 };
	B.allocateCustom(B_data, 100);
	arange(B,  1);

    qint32_t C_data[100]{} ;
	Tensor<qint32_t> C{ 10, 10 };
	C.allocateCustom(C_data, 100);

    matmul(A, B,C);
    qint32_t expected[] = { 3355, 3410, 3465, 3520, 3575, 3630, 3685, 3740, 3795, 3850, 7955, 8110, 8265, 8420, 8575, 8730, 8885, 9040, 9195, 9350, 12555, 12810, 13065, 13320, 13575, 13830, 14085, 14340, 14595, 14850, 17155, 17510, 17865, 18220, 18575, 18930, 19285, 19640, 19995, 20350, 21755, 22210, 22665, 23120, 23575, 24030, 24485, 24940, 25395, 25850, 26355, 26910, 27465, 28020, 28575, 29130, 29685, 30240, 30795, 31350, 30955, 31610, 32265, 32920, 33575, 34230, 34885, 35540, 36195, 36850, 35555, 36310, 37065, 37820, 38575, 39330, 40085, 40840, 41595, 42350, 40155, 41010, 41865, 42720, 43575, 44430, 45285, 46140, 46995, 47850, 44755, 45710, 46665, 47620, 48575, 49530, 50485, 51440, 52395, 53350 };

    Tensor<qint32_t> R{C.shape()};
    R.allocateCustom(expected, 100);
    EML_ASSERT_TENSOR_EQUALS(R, C);
}
inline int testMatMulCase1()
{
    qint32_t A_data[15]{} ;
	Tensor<qint32_t> A{ 3, 5 };
	A.allocateCustom(A_data, 15);
	arange(A,  1);

    qint32_t B_data[15]{} ;
	Tensor<qint32_t> B{ 3, 5 };
	B.allocateCustom(B_data, 15);
	arange(B,  1);

    qint32_t C_data[9]{} ;
	Tensor<qint32_t> C{ 3, 3 };
	C.allocateCustom(C_data, 9);

    matmulBTrans(A, B,C);
    qint32_t expected[] = { 55, 130, 205, 130, 330, 530, 205, 530, 855 };

    Tensor<qint32_t> R{C.shape()};
    R.allocateCustom(expected, 9);
    EML_ASSERT_TENSOR_EQUALS(R, C);
}
inline int testMatMulCase2()
{
    qint32_t A_data[15]{} ;
	Tensor<qint32_t> A{ 5, 3 };
	A.allocateCustom(A_data, 15);
	arange(A,  1);

    qint32_t B_data[15]{} ;
	Tensor<qint32_t> B{ 5, 3 };
	B.allocateCustom(B_data, 15);
	arange(B,  1);

    qint32_t C_data[9]{} ;
	Tensor<qint32_t> C{ 3, 3 };
	C.allocateCustom(C_data, 9);

    matmulATrans(A, B,C);
    qint32_t expected[] = { 335, 370, 405, 370, 410, 450, 405, 450, 495 };

    Tensor<qint32_t> R{C.shape()};
    R.allocateCustom(expected, 9);
    EML_ASSERT_TENSOR_EQUALS(R, C);
}
inline int testMatMulCase3()
{
    qint32_t A_data[64]{} ;
	Tensor<qint32_t> A{ 8, 8 };
	A.allocateCustom(A_data, 64);
	arange(A,  1);

    qint32_t B_data[64]{} ;
	Tensor<qint32_t> B{ 8, 8 };
	B.allocateCustom(B_data, 64);
	arange(B,  1);

    qint32_t C_data[64]{} ;
	Tensor<qint32_t> C{ 8, 8 };
	C.allocateCustom(C_data, 64);

    matmul(A, B,C);
    qint32_t expected[] = { 1380, 1416, 1452, 1488, 1524, 1560, 1596, 1632, 3236, 3336, 3436, 3536, 3636, 3736, 3836, 3936, 5092, 5256, 5420, 5584, 5748, 5912, 6076, 6240, 6948, 7176, 7404, 7632, 7860, 8088, 8316, 8544, 8804, 9096, 9388, 9680, 9972, 10264, 10556, 10848, 10660, 11016, 11372, 11728, 12084, 12440, 12796, 13152, 12516, 12936, 13356, 13776, 14196, 14616, 15036, 15456, 14372, 14856, 15340, 15824, 16308, 16792, 17276, 17760 };

    Tensor<qint32_t> R{C.shape()};
    R.allocateCustom(expected, 64);
    EML_ASSERT_TENSOR_EQUALS(R, C);
}
inline int testMatMulCase4()
{
    qint32_t A_data[100]{} ;
	Tensor<qint32_t> A{ 10, 10 };
	A.allocateCustom(A_data, 100);
	arange(A,  1);

    qint32_t B_data[100]{} ;
	Tensor<qint32_t> B{ 10, 10 };
	B.allocateCustom(B_data, 100);
	arange(B,  1);

    qint32_t C_data[100]{} ;
	Tensor<qint32_t> C{ 10, 10 };
	C.allocateCustom(C_data, 100);

    matmulBTrans(A, B,C);
    qint32_t expected[] = { 385, 935, 1485, 2035, 2585, 3135, 3685, 4235, 4785, 5335, 935, 2485, 4035, 5585, 7135, 8685, 10235, 11785, 13335, 14885, 1485, 4035, 6585, 9135, 11685, 14235, 16785, 19335, 21885, 24435, 2035, 5585, 9135, 12685, 16235, 19785, 23335, 26885, 30435, 33985, 2585, 7135, 11685, 16235, 20785, 25335, 29885, 34435, 38985, 43535, 3135, 8685, 14235, 19785, 25335, 30885, 36435, 41985, 47535, 53085, 3685, 10235, 16785, 23335, 29885, 36435, 42985, 49535, 56085, 62635, 4235, 11785, 19335, 26885, 34435, 41985, 49535, 57085, 64635, 72185, 4785, 13335, 21885, 30435, 38985, 47535, 56085, 64635, 73185, 81735, 5335, 14885, 24435, 33985, 43535, 53085, 62635, 72185, 81735, 91285 };

    Tensor<qint32_t> R{C.shape()};
    R.allocateCustom(expected, 100);
    EML_ASSERT_TENSOR_EQUALS(R, C);
}
inline int testMatMulCase5()
{
    qint32_t A_data[15]{} ;
	Tensor<qint32_t> A{ 3, 5 };
	A.allocateCustom(A_data, 15);
	arange(A,  1);

    qint32_t B_data[15]{} ;
	Tensor<qint32_t> B{ 3, 5 };
	B.allocateCustom(B_data, 15);
	arange(B,  1);

    qint32_t C_data[25]{} ;
	Tensor<qint32_t> C{ 5, 5 };
	C.allocateCustom(C_data, 25);

    matmulATrans(A, B,C);
    qint32_t expected[] = { 158, 176, 194, 212, 230, 176, 197, 218, 239, 260, 194, 218, 242, 266, 290, 212, 239, 266, 293, 320, 230, 260, 290, 320, 350 };

    Tensor<qint32_t> R{C.shape()};
    R.allocateCustom(expected, 25);
    EML_ASSERT_TENSOR_EQUALS(R, C);
}
inline int testMatMulCase6()
{
    qint32_t A_data[15]{} ;
	Tensor<qint32_t> A{ 5, 3 };
	A.allocateCustom(A_data, 15);
	arange(A,  1);

    qint32_t B_data[15]{} ;
	Tensor<qint32_t> B{ 3, 5 };
	B.allocateCustom(B_data, 15);
	arange(B,  1);

    qint32_t C_data[25]{} ;
	Tensor<qint32_t> C{ 5, 5 };
	C.allocateCustom(C_data, 25);

    matmul(A, B,C);
    qint32_t expected[] = { 46, 52, 58, 64, 70, 100, 115, 130, 145, 160, 154, 178, 202, 226, 250, 208, 241, 274, 307, 340, 262, 304, 346, 388, 430 };

    Tensor<qint32_t> R{C.shape()};
    R.allocateCustom(expected, 25);
    EML_ASSERT_TENSOR_EQUALS(R, C);
}
inline int testMatMulCase7()
{
    qint32_t A_data[64]{} ;
	Tensor<qint32_t> A{ 8, 8 };
	A.allocateCustom(A_data, 64);
	arange(A,  1);

    qint32_t B_data[64]{} ;
	Tensor<qint32_t> B{ 8, 8 };
	B.allocateCustom(B_data, 64);
	arange(B,  1);

    qint32_t C_data[64]{} ;
	Tensor<qint32_t> C{ 8, 8 };
	C.allocateCustom(C_data, 64);

    matmulBTrans(A, B,C);
    qint32_t expected[] = { 204, 492, 780, 1068, 1356, 1644, 1932, 2220, 492, 1292, 2092, 2892, 3692, 4492, 5292, 6092, 780, 2092, 3404, 4716, 6028, 7340, 8652, 9964, 1068, 2892, 4716, 6540, 8364, 10188, 12012, 13836, 1356, 3692, 6028, 8364, 10700, 13036, 15372, 17708, 1644, 4492, 7340, 10188, 13036, 15884, 18732, 21580, 1932, 5292, 8652, 12012, 15372, 18732, 22092, 25452, 2220, 6092, 9964, 13836, 17708, 21580, 25452, 29324 };

    Tensor<qint32_t> R{C.shape()};
    R.allocateCustom(expected, 64);
    EML_ASSERT_TENSOR_EQUALS(R, C);
}
inline int testMatMulCase8()
{
    qint32_t A_data[100]{} ;
	Tensor<qint32_t> A{ 10, 10 };
	A.allocateCustom(A_data, 100);
	arange(A,  1);

    qint32_t B_data[100]{} ;
	Tensor<qint32_t> B{ 10, 10 };
	B.allocateCustom(B_data, 100);
	arange(B,  1);

    qint32_t C_data[100]{} ;
	Tensor<qint32_t> C{ 10, 10 };
	C.allocateCustom(C_data, 100);

    matmulATrans(A, B,C);
    qint32_t expected[] = { 29410, 29870, 30330, 30790, 31250, 31710, 32170, 32630, 33090, 33550, 29870, 30340, 30810, 31280, 31750, 32220, 32690, 33160, 33630, 34100, 30330, 30810, 31290, 31770, 32250, 32730, 33210, 33690, 34170, 34650, 30790, 31280, 31770, 32260, 32750, 33240, 33730, 34220, 34710, 35200, 31250, 31750, 32250, 32750, 33250, 33750, 34250, 34750, 35250, 35750, 31710, 32220, 32730, 33240, 33750, 34260, 34770, 35280, 35790, 36300, 32170, 32690, 33210, 33730, 34250, 34770, 35290, 35810, 36330, 36850, 32630, 33160, 33690, 34220, 34750, 35280, 35810, 36340, 36870, 37400, 33090, 33630, 34170, 34710, 35250, 35790, 36330, 36870, 37410, 37950, 33550, 34100, 34650, 35200, 35750, 36300, 36850, 37400, 37950, 38500 };

    Tensor<qint32_t> R{C.shape()};
    R.allocateCustom(expected, 100);
    EML_ASSERT_TENSOR_EQUALS(R, C);
}
inline int testMatMulCase9()
{
    qint32_t A_data[15]{} ;
	Tensor<qint32_t> A{ 3, 5 };
	A.allocateCustom(A_data, 15);
	arange(A,  1);

    qint32_t B_data[15]{} ;
	Tensor<qint32_t> B{ 5, 3 };
	B.allocateCustom(B_data, 15);
	arange(B,  1);

    qint32_t C_data[9]{} ;
	Tensor<qint32_t> C{ 3, 3 };
	C.allocateCustom(C_data, 9);

    matmul(A, B,C);
    qint32_t expected[] = { 135, 150, 165, 310, 350, 390, 485, 550, 615 };

    Tensor<qint32_t> R{C.shape()};
    R.allocateCustom(expected, 9);
    EML_ASSERT_TENSOR_EQUALS(R, C);
}
inline int testMatMulCase10()
{
    qint32_t A_data[64]{} ;
	Tensor<qint32_t> A{ 8, 8 };
	A.allocateCustom(A_data, 64);
	arange(A,  1);

    qint32_t B_data[64]{} ;
	Tensor<qint32_t> B{ 8, 8 };
	B.allocateCustom(B_data, 64);
	arange(B,  1);

    qint32_t C_data[64]{} ;
	Tensor<qint32_t> C{ 8, 8 };
	C.allocateCustom(C_data, 64);

    matmulATrans(A, B,C);
    qint32_t expected[] = { 9416, 9648, 9880, 10112, 10344, 10576, 10808, 11040, 9648, 9888, 10128, 10368, 10608, 10848, 11088, 11328, 9880, 10128, 10376, 10624, 10872, 11120, 11368, 11616, 10112, 10368, 10624, 10880, 11136, 11392, 11648, 11904, 10344, 10608, 10872, 11136, 11400, 11664, 11928, 12192, 10576, 10848, 11120, 11392, 11664, 11936, 12208, 12480, 10808, 11088, 11368, 11648, 11928, 12208, 12488, 12768, 11040, 11328, 11616, 11904, 12192, 12480, 12768, 13056 };

    Tensor<qint32_t> R{C.shape()};
    R.allocateCustom(expected, 64);
    EML_ASSERT_TENSOR_EQUALS(R, C);
}
inline int testMatMulCase11()
{
    qint32_t A_data[100]{} ;
	Tensor<qint32_t> A{ 10, 10 };
	A.allocateCustom(A_data, 100);
	arange(A,  1);

    qint32_t B_data[100]{} ;
	Tensor<qint32_t> B{ 10, 10 };
	B.allocateCustom(B_data, 100);
	arange(B,  1);

    qint32_t C_data[100]{} ;
	Tensor<qint32_t> C{ 10, 10 };
	C.allocateCustom(C_data, 100);

    matmulBTrans(A, B,C);
    qint32_t expected[] = { 385, 935, 1485, 2035, 2585, 3135, 3685, 4235, 4785, 5335, 935, 2485, 4035, 5585, 7135, 8685, 10235, 11785, 13335, 14885, 1485, 4035, 6585, 9135, 11685, 14235, 16785, 19335, 21885, 24435, 2035, 5585, 9135, 12685, 16235, 19785, 23335, 26885, 30435, 33985, 2585, 7135, 11685, 16235, 20785, 25335, 29885, 34435, 38985, 43535, 3135, 8685, 14235, 19785, 25335, 30885, 36435, 41985, 47535, 53085, 3685, 10235, 16785, 23335, 29885, 36435, 42985, 49535, 56085, 62635, 4235, 11785, 19335, 26885, 34435, 41985, 49535, 57085, 64635, 72185, 4785, 13335, 21885, 30435, 38985, 47535, 56085, 64635, 73185, 81735, 5335, 14885, 24435, 33985, 43535, 53085, 62635, 72185, 81735, 91285 };

    Tensor<qint32_t> R{C.shape()};
    R.allocateCustom(expected, 100);
    EML_ASSERT_TENSOR_EQUALS(R, C);
}
inline int testMatMulCase12()
{
    qint32_t A_data[15]{} ;
	Tensor<qint32_t> A{ 5, 3 };
	A.allocateCustom(A_data, 15);
	arange(A,  1);

    qint32_t B_data[15]{} ;
	Tensor<qint32_t> B{ 3, 5 };
	B.allocateCustom(B_data, 15);
	arange(B,  1);

    qint32_t C_data[25]{} ;
	Tensor<qint32_t> C{ 5, 5 };
	C.allocateCustom(C_data, 25);

    matmul(A, B,C);
    qint32_t expected[] = { 46, 52, 58, 64, 70, 100, 115, 130, 145, 160, 154, 178, 202, 226, 250, 208, 241, 274, 307, 340, 262, 304, 346, 388, 430 };

    Tensor<qint32_t> R{C.shape()};
    R.allocateCustom(expected, 25);
    EML_ASSERT_TENSOR_EQUALS(R, C);
}
inline int testMatMulCase13()
{
    qint32_t A_data[15]{} ;
	Tensor<qint32_t> A{ 3, 5 };
	A.allocateCustom(A_data, 15);
	arange(A,  1);

    qint32_t B_data[15]{} ;
	Tensor<qint32_t> B{ 3, 5 };
	B.allocateCustom(B_data, 15);
	arange(B,  1);

    qint32_t C_data[25]{} ;
	Tensor<qint32_t> C{ 5, 5 };
	C.allocateCustom(C_data, 25);

    matmulATrans(A, B,C);
    qint32_t expected[] = { 158, 176, 194, 212, 230, 176, 197, 218, 239, 260, 194, 218, 242, 266, 290, 212, 239, 266, 293, 320, 230, 260, 290, 320, 350 };

    Tensor<qint32_t> R{C.shape()};
    R.allocateCustom(expected, 25);
    EML_ASSERT_TENSOR_EQUALS(R, C);
}
inline int testMatMulCase14()
{
    qint32_t A_data[15]{} ;
	Tensor<qint32_t> A{ 3, 5 };
	A.allocateCustom(A_data, 15);
	arange(A,  1);

    qint32_t B_data[15]{} ;
	Tensor<qint32_t> B{ 3, 5 };
	B.allocateCustom(B_data, 15);
	arange(B,  1);

    qint32_t C_data[9]{} ;
	Tensor<qint32_t> C{ 3, 3 };
	C.allocateCustom(C_data, 9);

    matmulBTrans(A, B,C);
    qint32_t expected[] = { 55, 130, 205, 130, 330, 530, 205, 530, 855 };

    Tensor<qint32_t> R{C.shape()};
    R.allocateCustom(expected, 9);
    EML_ASSERT_TENSOR_EQUALS(R, C);
}
inline int testMatMulCase15()
{
    qint32_t A_data[100]{} ;
	Tensor<qint32_t> A{ 10, 10 };
	A.allocateCustom(A_data, 100);
	arange(A,  1);

    qint32_t B_data[100]{} ;
	Tensor<qint32_t> B{ 10, 10 };
	B.allocateCustom(B_data, 100);
	arange(B,  1);

    qint32_t C_data[100]{} ;
	Tensor<qint32_t> C{ 10, 10 };
	C.allocateCustom(C_data, 100);

    matmul(A, B,C);
    qint32_t expected[] = { 3355, 3410, 3465, 3520, 3575, 3630, 3685, 3740, 3795, 3850, 7955, 8110, 8265, 8420, 8575, 8730, 8885, 9040, 9195, 9350, 12555, 12810, 13065, 13320, 13575, 13830, 14085, 14340, 14595, 14850, 17155, 17510, 17865, 18220, 18575, 18930, 19285, 19640, 19995, 20350, 21755, 22210, 22665, 23120, 23575, 24030, 24485, 24940, 25395, 25850, 26355, 26910, 27465, 28020, 28575, 29130, 29685, 30240, 30795, 31350, 30955, 31610, 32265, 32920, 33575, 34230, 34885, 35540, 36195, 36850, 35555, 36310, 37065, 37820, 38575, 39330, 40085, 40840, 41595, 42350, 40155, 41010, 41865, 42720, 43575, 44430, 45285, 46140, 46995, 47850, 44755, 45710, 46665, 47620, 48575, 49530, 50485, 51440, 52395, 53350 };

    Tensor<qint32_t> R{C.shape()};
    R.allocateCustom(expected, 100);
    EML_ASSERT_TENSOR_EQUALS(R, C);
}
inline int testMatMulCase16()
{
    qint32_t A_data[15]{} ;
	Tensor<qint32_t> A{ 5, 3 };
	A.allocateCustom(A_data, 15);
	arange(A,  1);

    qint32_t B_data[15]{} ;
	Tensor<qint32_t> B{ 5, 3 };
	B.allocateCustom(B_data, 15);
	arange(B,  1);

    qint32_t C_data[9]{} ;
	Tensor<qint32_t> C{ 3, 3 };
	C.allocateCustom(C_data, 9);

    matmulATrans(A, B,C);
    qint32_t expected[] = { 335, 370, 405, 370, 410, 450, 405, 450, 495 };

    Tensor<qint32_t> R{C.shape()};
    R.allocateCustom(expected, 9);
    EML_ASSERT_TENSOR_EQUALS(R, C);
}
inline int testMatMulCase17()
{
    qint32_t A_data[64]{} ;
	Tensor<qint32_t> A{ 8, 8 };
	A.allocateCustom(A_data, 64);
	arange(A,  1);

    qint32_t B_data[64]{} ;
	Tensor<qint32_t> B{ 8, 8 };
	B.allocateCustom(B_data, 64);
	arange(B,  1);

    qint32_t C_data[64]{} ;
	Tensor<qint32_t> C{ 8, 8 };
	C.allocateCustom(C_data, 64);

    matmulBTrans(A, B,C);
    qint32_t expected[] = { 204, 492, 780, 1068, 1356, 1644, 1932, 2220, 492, 1292, 2092, 2892, 3692, 4492, 5292, 6092, 780, 2092, 3404, 4716, 6028, 7340, 8652, 9964, 1068, 2892, 4716, 6540, 8364, 10188, 12012, 13836, 1356, 3692, 6028, 8364, 10700, 13036, 15372, 17708, 1644, 4492, 7340, 10188, 13036, 15884, 18732, 21580, 1932, 5292, 8652, 12012, 15372, 18732, 22092, 25452, 2220, 6092, 9964, 13836, 17708, 21580, 25452, 29324 };

    Tensor<qint32_t> R{C.shape()};
    R.allocateCustom(expected, 64);
    EML_ASSERT_TENSOR_EQUALS(R, C);
}
inline int testMatMulCase18()
{
    qint32_t A_data[64]{} ;
	Tensor<qint32_t> A{ 8, 8 };
	A.allocateCustom(A_data, 64);
	arange(A,  1);

    qint32_t B_data[64]{} ;
	Tensor<qint32_t> B{ 8, 8 };
	B.allocateCustom(B_data, 64);
	arange(B,  1);

    qint32_t C_data[64]{} ;
	Tensor<qint32_t> C{ 8, 8 };
	C.allocateCustom(C_data, 64);

    matmul(A, B,C);
    qint32_t expected[] = { 1380, 1416, 1452, 1488, 1524, 1560, 1596, 1632, 3236, 3336, 3436, 3536, 3636, 3736, 3836, 3936, 5092, 5256, 5420, 5584, 5748, 5912, 6076, 6240, 6948, 7176, 7404, 7632, 7860, 8088, 8316, 8544, 8804, 9096, 9388, 9680, 9972, 10264, 10556, 10848, 10660, 11016, 11372, 11728, 12084, 12440, 12796, 13152, 12516, 12936, 13356, 13776, 14196, 14616, 15036, 15456, 14372, 14856, 15340, 15824, 16308, 16792, 17276, 17760 };

    Tensor<qint32_t> R{C.shape()};
    R.allocateCustom(expected, 64);
    EML_ASSERT_TENSOR_EQUALS(R, C);
}
inline int testMatMulCase19()
{
    qint32_t A_data[100]{} ;
	Tensor<qint32_t> A{ 10, 10 };
	A.allocateCustom(A_data, 100);
	arange(A,  1);

    qint32_t B_data[100]{} ;
	Tensor<qint32_t> B{ 10, 10 };
	B.allocateCustom(B_data, 100);
	arange(B,  1);

    qint32_t C_data[100]{} ;
	Tensor<qint32_t> C{ 10, 10 };
	C.allocateCustom(C_data, 100);

    matmulATrans(A, B,C);
    qint32_t expected[] = { 29410, 29870, 30330, 30790, 31250, 31710, 32170, 32630, 33090, 33550, 29870, 30340, 30810, 31280, 31750, 32220, 32690, 33160, 33630, 34100, 30330, 30810, 31290, 31770, 32250, 32730, 33210, 33690, 34170, 34650, 30790, 31280, 31770, 32260, 32750, 33240, 33730, 34220, 34710, 35200, 31250, 31750, 32250, 32750, 33250, 33750, 34250, 34750, 35250, 35750, 31710, 32220, 32730, 33240, 33750, 34260, 34770, 35280, 35790, 36300, 32170, 32690, 33210, 33730, 34250, 34770, 35290, 35810, 36330, 36850, 32630, 33160, 33690, 34220, 34750, 35280, 35810, 36340, 36870, 37400, 33090, 33630, 34170, 34710, 35250, 35790, 36330, 36870, 37410, 37950, 33550, 34100, 34650, 35200, 35750, 36300, 36850, 37400, 37950, 38500 };

    Tensor<qint32_t> R{C.shape()};
    R.allocateCustom(expected, 100);
    EML_ASSERT_TENSOR_EQUALS(R, C);
}
inline int testMatMulCase20()
{
    qint32_t A_data[15]{} ;
	Tensor<qint32_t> A{ 5, 3 };
	A.allocateCustom(A_data, 15);
	arange(A,  1);

    qint32_t B_data[15]{} ;
	Tensor<qint32_t> B{ 5, 3 };
	B.allocateCustom(B_data, 15);
	arange(B,  1);

    qint32_t C_data[25]{} ;
	Tensor<qint32_t> C{ 5, 5 };
	C.allocateCustom(C_data, 25);

    matmulBTrans(A, B,C);
    qint32_t expected[] = { 14, 32, 50, 68, 86, 32, 77, 122, 167, 212, 50, 122, 194, 266, 338, 68, 167, 266, 365, 464, 86, 212, 338, 464, 590 };

    Tensor<qint32_t> R{C.shape()};
    R.allocateCustom(expected, 25);
    EML_ASSERT_TENSOR_EQUALS(R, C);
}
inline int testMatMulCase21()
{
    qint32_t A_data[15]{} ;
	Tensor<qint32_t> A{ 3, 5 };
	A.allocateCustom(A_data, 15);
	arange(A,  1);

    qint32_t B_data[15]{} ;
	Tensor<qint32_t> B{ 5, 3 };
	B.allocateCustom(B_data, 15);
	arange(B,  1);

    qint32_t C_data[9]{} ;
	Tensor<qint32_t> C{ 3, 3 };
	C.allocateCustom(C_data, 9);

    matmul(A, B,C);
    qint32_t expected[] = { 135, 150, 165, 310, 350, 390, 485, 550, 615 };

    Tensor<qint32_t> R{C.shape()};
    R.allocateCustom(expected, 9);
    EML_ASSERT_TENSOR_EQUALS(R, C);
}
inline int testMatMulCase22()
{
    qint32_t A_data[64]{} ;
	Tensor<qint32_t> A{ 8, 8 };
	A.allocateCustom(A_data, 64);
	arange(A,  1);

    qint32_t B_data[64]{} ;
	Tensor<qint32_t> B{ 8, 8 };
	B.allocateCustom(B_data, 64);
	arange(B,  1);

    qint32_t C_data[64]{} ;
	Tensor<qint32_t> C{ 8, 8 };
	C.allocateCustom(C_data, 64);

    matmulATrans(A, B,C);
    qint32_t expected[] = { 9416, 9648, 9880, 10112, 10344, 10576, 10808, 11040, 9648, 9888, 10128, 10368, 10608, 10848, 11088, 11328, 9880, 10128, 10376, 10624, 10872, 11120, 11368, 11616, 10112, 10368, 10624, 10880, 11136, 11392, 11648, 11904, 10344, 10608, 10872, 11136, 11400, 11664, 11928, 12192, 10576, 10848, 11120, 11392, 11664, 11936, 12208, 12480, 10808, 11088, 11368, 11648, 11928, 12208, 12488, 12768, 11040, 11328, 11616, 11904, 12192, 12480, 12768, 13056 };

    Tensor<qint32_t> R{C.shape()};
    R.allocateCustom(expected, 64);
    EML_ASSERT_TENSOR_EQUALS(R, C);
}
inline int testMatMulCase23()
{
    qint32_t A_data[100]{} ;
	Tensor<qint32_t> A{ 10, 10 };
	A.allocateCustom(A_data, 100);
	arange(A,  1);

    qint32_t B_data[100]{} ;
	Tensor<qint32_t> B{ 10, 10 };
	B.allocateCustom(B_data, 100);
	arange(B,  1);

    qint32_t C_data[100]{} ;
	Tensor<qint32_t> C{ 10, 10 };
	C.allocateCustom(C_data, 100);

    matmulBTrans(A, B,C);
    qint32_t expected[] = { 385, 935, 1485, 2035, 2585, 3135, 3685, 4235, 4785, 5335, 935, 2485, 4035, 5585, 7135, 8685, 10235, 11785, 13335, 14885, 1485, 4035, 6585, 9135, 11685, 14235, 16785, 19335, 21885, 24435, 2035, 5585, 9135, 12685, 16235, 19785, 23335, 26885, 30435, 33985, 2585, 7135, 11685, 16235, 20785, 25335, 29885, 34435, 38985, 43535, 3135, 8685, 14235, 19785, 25335, 30885, 36435, 41985, 47535, 53085, 3685, 10235, 16785, 23335, 29885, 36435, 42985, 49535, 56085, 62635, 4235, 11785, 19335, 26885, 34435, 41985, 49535, 57085, 64635, 72185, 4785, 13335, 21885, 30435, 38985, 47535, 56085, 64635, 73185, 81735, 5335, 14885, 24435, 33985, 43535, 53085, 62635, 72185, 81735, 91285 };

    Tensor<qint32_t> R{C.shape()};
    R.allocateCustom(expected, 100);
    EML_ASSERT_TENSOR_EQUALS(R, C);
}
inline int testMatMulCase24()
{
    qint32_t A_data[15]{} ;
	Tensor<qint32_t> A{ 5, 3 };
	A.allocateCustom(A_data, 15);
	arange(A,  1);

    qint32_t B_data[15]{} ;
	Tensor<qint32_t> B{ 3, 5 };
	B.allocateCustom(B_data, 15);
	arange(B,  1);

    qint32_t C_data[25]{} ;
	Tensor<qint32_t> C{ 5, 5 };
	C.allocateCustom(C_data, 25);

    matmul(A, B,C);
    qint32_t expected[] = { 46, 52, 58, 64, 70, 100, 115, 130, 145, 160, 154, 178, 202, 226, 250, 208, 241, 274, 307, 340, 262, 304, 346, 388, 430 };

    Tensor<qint32_t> R{C.shape()};
    R.allocateCustom(expected, 25);
    EML_ASSERT_TENSOR_EQUALS(R, C);
}
inline int testMatMulCase25()
{
    qint32_t A_data[15]{} ;
	Tensor<qint32_t> A{ 3, 5 };
	A.allocateCustom(A_data, 15);
	arange(A,  1);

    qint32_t B_data[15]{} ;
	Tensor<qint32_t> B{ 3, 5 };
	B.allocateCustom(B_data, 15);
	arange(B,  1);

    qint32_t C_data[25]{} ;
	Tensor<qint32_t> C{ 5, 5 };
	C.allocateCustom(C_data, 25);

    matmulATrans(A, B,C);
    qint32_t expected[] = { 158, 176, 194, 212, 230, 176, 197, 218, 239, 260, 194, 218, 242, 266, 290, 212, 239, 266, 293, 320, 230, 260, 290, 320, 350 };

    Tensor<qint32_t> R{C.shape()};
    R.allocateCustom(expected, 25);
    EML_ASSERT_TENSOR_EQUALS(R, C);
}
inline int testMatMulCase26()
{
    qint32_t A_data[15]{} ;
	Tensor<qint32_t> A{ 3, 5 };
	A.allocateCustom(A_data, 15);
	arange(A,  1);

    qint32_t B_data[15]{} ;
	Tensor<qint32_t> B{ 3, 5 };
	B.allocateCustom(B_data, 15);
	arange(B,  1);

    qint32_t C_data[9]{} ;
	Tensor<qint32_t> C{ 3, 3 };
	C.allocateCustom(C_data, 9);

    matmulBTrans(A, B,C);
    qint32_t expected[] = { 55, 130, 205, 130, 330, 530, 205, 530, 855 };

    Tensor<qint32_t> R{C.shape()};
    R.allocateCustom(expected, 9);
    EML_ASSERT_TENSOR_EQUALS(R, C);
}
inline int testMatMulCase27()
{
    qint32_t A_data[100]{} ;
	Tensor<qint32_t> A{ 10, 10 };
	A.allocateCustom(A_data, 100);
	arange(A,  1);

    qint32_t B_data[100]{} ;
	Tensor<qint32_t> B{ 10, 10 };
	B.allocateCustom(B_data, 100);
	arange(B,  1);

    qint32_t C_data[100]{} ;
	Tensor<qint32_t> C{ 10, 10 };
	C.allocateCustom(C_data, 100);

    matmul(A, B,C);
    qint32_t expected[] = { 3355, 3410, 3465, 3520, 3575, 3630, 3685, 3740, 3795, 3850, 7955, 8110, 8265, 8420, 8575, 8730, 8885, 9040, 9195, 9350, 12555, 12810, 13065, 13320, 13575, 13830, 14085, 14340, 14595, 14850, 17155, 17510, 17865, 18220, 18575, 18930, 19285, 19640, 19995, 20350, 21755, 22210, 22665, 23120, 23575, 24030, 24485, 24940, 25395, 25850, 26355, 26910, 27465, 28020, 28575, 29130, 29685, 30240, 30795, 31350, 30955, 31610, 32265, 32920, 33575, 34230, 34885, 35540, 36195, 36850, 35555, 36310, 37065, 37820, 38575, 39330, 40085, 40840, 41595, 42350, 40155, 41010, 41865, 42720, 43575, 44430, 45285, 46140, 46995, 47850, 44755, 45710, 46665, 47620, 48575, 49530, 50485, 51440, 52395, 53350 };

    Tensor<qint32_t> R{C.shape()};
    R.allocateCustom(expected, 100);
    EML_ASSERT_TENSOR_EQUALS(R, C);
}
inline int testMatMulCase28()
{
    qint32_t A_data[15]{} ;
	Tensor<qint32_t> A{ 5, 3 };
	A.allocateCustom(A_data, 15);
	arange(A,  1);

    qint32_t B_data[15]{} ;
	Tensor<qint32_t> B{ 5, 3 };
	B.allocateCustom(B_data, 15);
	arange(B,  1);

    qint32_t C_data[9]{} ;
	Tensor<qint32_t> C{ 3, 3 };
	C.allocateCustom(C_data, 9);

    matmulATrans(A, B,C);
    qint32_t expected[] = { 335, 370, 405, 370, 410, 450, 405, 450, 495 };

    Tensor<qint32_t> R{C.shape()};
    R.allocateCustom(expected, 9);
    EML_ASSERT_TENSOR_EQUALS(R, C);
}
inline int testMatMulCase29()
{
    qint32_t A_data[64]{} ;
	Tensor<qint32_t> A{ 8, 8 };
	A.allocateCustom(A_data, 64);
	arange(A,  1);

    qint32_t B_data[64]{} ;
	Tensor<qint32_t> B{ 8, 8 };
	B.allocateCustom(B_data, 64);
	arange(B,  1);

    qint32_t C_data[64]{} ;
	Tensor<qint32_t> C{ 8, 8 };
	C.allocateCustom(C_data, 64);

    matmulBTrans(A, B,C);
    qint32_t expected[] = { 204, 492, 780, 1068, 1356, 1644, 1932, 2220, 492, 1292, 2092, 2892, 3692, 4492, 5292, 6092, 780, 2092, 3404, 4716, 6028, 7340, 8652, 9964, 1068, 2892, 4716, 6540, 8364, 10188, 12012, 13836, 1356, 3692, 6028, 8364, 10700, 13036, 15372, 17708, 1644, 4492, 7340, 10188, 13036, 15884, 18732, 21580, 1932, 5292, 8652, 12012, 15372, 18732, 22092, 25452, 2220, 6092, 9964, 13836, 17708, 21580, 25452, 29324 };

    Tensor<qint32_t> R{C.shape()};
    R.allocateCustom(expected, 64);
    EML_ASSERT_TENSOR_EQUALS(R, C);
}
inline int testMatMulCase30()
{
    qint32_t A_data[64]{} ;
	Tensor<qint32_t> A{ 8, 8 };
	A.allocateCustom(A_data, 64);
	arange(A,  1);

    qint32_t B_data[64]{} ;
	Tensor<qint32_t> B{ 8, 8 };
	B.allocateCustom(B_data, 64);
	arange(B,  1);

    qint32_t C_data[64]{} ;
	Tensor<qint32_t> C{ 8, 8 };
	C.allocateCustom(C_data, 64);

    matmul(A, B,C);
    qint32_t expected[] = { 1380, 1416, 1452, 1488, 1524, 1560, 1596, 1632, 3236, 3336, 3436, 3536, 3636, 3736, 3836, 3936, 5092, 5256, 5420, 5584, 5748, 5912, 6076, 6240, 6948, 7176, 7404, 7632, 7860, 8088, 8316, 8544, 8804, 9096, 9388, 9680, 9972, 10264, 10556, 10848, 10660, 11016, 11372, 11728, 12084, 12440, 12796, 13152, 12516, 12936, 13356, 13776, 14196, 14616, 15036, 15456, 14372, 14856, 15340, 15824, 16308, 16792, 17276, 17760 };

    Tensor<qint32_t> R{C.shape()};
    R.allocateCustom(expected, 64);
    EML_ASSERT_TENSOR_EQUALS(R, C);
}
inline int testMatMulCase31()
{
    qint32_t A_data[100]{} ;
	Tensor<qint32_t> A{ 10, 10 };
	A.allocateCustom(A_data, 100);
	arange(A,  1);

    qint32_t B_data[100]{} ;
	Tensor<qint32_t> B{ 10, 10 };
	B.allocateCustom(B_data, 100);
	arange(B,  1);

    qint32_t C_data[100]{} ;
	Tensor<qint32_t> C{ 10, 10 };
	C.allocateCustom(C_data, 100);

    matmulATrans(A, B,C);
    qint32_t expected[] = { 29410, 29870, 30330, 30790, 31250, 31710, 32170, 32630, 33090, 33550, 29870, 30340, 30810, 31280, 31750, 32220, 32690, 33160, 33630, 34100, 30330, 30810, 31290, 31770, 32250, 32730, 33210, 33690, 34170, 34650, 30790, 31280, 31770, 32260, 32750, 33240, 33730, 34220, 34710, 35200, 31250, 31750, 32250, 32750, 33250, 33750, 34250, 34750, 35250, 35750, 31710, 32220, 32730, 33240, 33750, 34260, 34770, 35280, 35790, 36300, 32170, 32690, 33210, 33730, 34250, 34770, 35290, 35810, 36330, 36850, 32630, 33160, 33690, 34220, 34750, 35280, 35810, 36340, 36870, 37400, 33090, 33630, 34170, 34710, 35250, 35790, 36330, 36870, 37410, 37950, 33550, 34100, 34650, 35200, 35750, 36300, 36850, 37400, 37950, 38500 };

    Tensor<qint32_t> R{C.shape()};
    R.allocateCustom(expected, 100);
    EML_ASSERT_TENSOR_EQUALS(R, C);
}
inline int testMatMulCase32()
{
    qint32_t A_data[15]{} ;
	Tensor<qint32_t> A{ 5, 3 };
	A.allocateCustom(A_data, 15);
	arange(A,  1);

    qint32_t B_data[15]{} ;
	Tensor<qint32_t> B{ 5, 3 };
	B.allocateCustom(B_data, 15);
	arange(B,  1);

    qint32_t C_data[25]{} ;
	Tensor<qint32_t> C{ 5, 5 };
	C.allocateCustom(C_data, 25);

    matmulBTrans(A, B,C);
    qint32_t expected[] = { 14, 32, 50, 68, 86, 32, 77, 122, 167, 212, 50, 122, 194, 266, 338, 68, 167, 266, 365, 464, 86, 212, 338, 464, 590 };

    Tensor<qint32_t> R{C.shape()};
    R.allocateCustom(expected, 25);
    EML_ASSERT_TENSOR_EQUALS(R, C);
}
inline int testMatMulCase33()
{
    qint32_t A_data[15]{} ;
	Tensor<qint32_t> A{ 3, 5 };
	A.allocateCustom(A_data, 15);
	arange(A,  1);

    qint32_t B_data[15]{} ;
	Tensor<qint32_t> B{ 5, 3 };
	B.allocateCustom(B_data, 15);
	arange(B,  1);

    qint32_t C_data[9]{} ;
	Tensor<qint32_t> C{ 3, 3 };
	C.allocateCustom(C_data, 9);

    matmul(A, B,C);
    qint32_t expected[] = { 135, 150, 165, 310, 350, 390, 485, 550, 615 };

    Tensor<qint32_t> R{C.shape()};
    R.allocateCustom(expected, 9);
    EML_ASSERT_TENSOR_EQUALS(R, C);
}
inline int testMatMulCase34()
{
    qint32_t A_data[64]{} ;
	Tensor<qint32_t> A{ 8, 8 };
	A.allocateCustom(A_data, 64);
	arange(A,  1);

    qint32_t B_data[64]{} ;
	Tensor<qint32_t> B{ 8, 8 };
	B.allocateCustom(B_data, 64);
	arange(B,  1);

    qint32_t C_data[64]{} ;
	Tensor<qint32_t> C{ 8, 8 };
	C.allocateCustom(C_data, 64);

    matmulATrans(A, B,C);
    qint32_t expected[] = { 9416, 9648, 9880, 10112, 10344, 10576, 10808, 11040, 9648, 9888, 10128, 10368, 10608, 10848, 11088, 11328, 9880, 10128, 10376, 10624, 10872, 11120, 11368, 11616, 10112, 10368, 10624, 10880, 11136, 11392, 11648, 11904, 10344, 10608, 10872, 11136, 11400, 11664, 11928, 12192, 10576, 10848, 11120, 11392, 11664, 11936, 12208, 12480, 10808, 11088, 11368, 11648, 11928, 12208, 12488, 12768, 11040, 11328, 11616, 11904, 12192, 12480, 12768, 13056 };

    Tensor<qint32_t> R{C.shape()};
    R.allocateCustom(expected, 64);
    EML_ASSERT_TENSOR_EQUALS(R, C);
}
inline int testMatMulCase35()
{
    qint32_t A_data[100]{} ;
	Tensor<qint32_t> A{ 10, 10 };
	A.allocateCustom(A_data, 100);
	arange(A,  1);

    qint32_t B_data[100]{} ;
	Tensor<qint32_t> B{ 10, 10 };
	B.allocateCustom(B_data, 100);
	arange(B,  1);

    qint32_t C_data[100]{} ;
	Tensor<qint32_t> C{ 10, 10 };
	C.allocateCustom(C_data, 100);

    matmulBTrans(A, B,C);
    qint32_t expected[] = { 385, 935, 1485, 2035, 2585, 3135, 3685, 4235, 4785, 5335, 935, 2485, 4035, 5585, 7135, 8685, 10235, 11785, 13335, 14885, 1485, 4035, 6585, 9135, 11685, 14235, 16785, 19335, 21885, 24435, 2035, 5585, 9135, 12685, 16235, 19785, 23335, 26885, 30435, 33985, 2585, 7135, 11685, 16235, 20785, 25335, 29885, 34435, 38985, 43535, 3135, 8685, 14235, 19785, 25335, 30885, 36435, 41985, 47535, 53085, 3685, 10235, 16785, 23335, 29885, 36435, 42985, 49535, 56085, 62635, 4235, 11785, 19335, 26885, 34435, 41985, 49535, 57085, 64635, 72185, 4785, 13335, 21885, 30435, 38985, 47535, 56085, 64635, 73185, 81735, 5335, 14885, 24435, 33985, 43535, 53085, 62635, 72185, 81735, 91285 };

    Tensor<qint32_t> R{C.shape()};
    R.allocateCustom(expected, 100);
    EML_ASSERT_TENSOR_EQUALS(R, C);
}
inline int testMatMulCase36()
{
    qint32_t A_data[15]{} ;
	Tensor<qint32_t> A{ 5, 3 };
	A.allocateCustom(A_data, 15);
	arange(A,  1);

    qint32_t B_data[15]{} ;
	Tensor<qint32_t> B{ 3, 5 };
	B.allocateCustom(B_data, 15);
	arange(B,  1);

    qint32_t C_data[25]{} ;
	Tensor<qint32_t> C{ 5, 5 };
	C.allocateCustom(C_data, 25);

    matmul(A, B,C);
    qint32_t expected[] = { 46, 52, 58, 64, 70, 100, 115, 130, 145, 160, 154, 178, 202, 226, 250, 208, 241, 274, 307, 340, 262, 304, 346, 388, 430 };

    Tensor<qint32_t> R{C.shape()};
    R.allocateCustom(expected, 25);
    EML_ASSERT_TENSOR_EQUALS(R, C);
}
inline int testMatMulCase37()
{
    qint32_t A_data[15]{} ;
	Tensor<qint32_t> A{ 3, 5 };
	A.allocateCustom(A_data, 15);
	arange(A,  1);

    qint32_t B_data[15]{} ;
	Tensor<qint32_t> B{ 3, 5 };
	B.allocateCustom(B_data, 15);
	arange(B,  1);

    qint32_t C_data[25]{} ;
	Tensor<qint32_t> C{ 5, 5 };
	C.allocateCustom(C_data, 25);

    matmulATrans(A, B,C);
    qint32_t expected[] = { 158, 176, 194, 212, 230, 176, 197, 218, 239, 260, 194, 218, 242, 266, 290, 212, 239, 266, 293, 320, 230, 260, 290, 320, 350 };

    Tensor<qint32_t> R{C.shape()};
    R.allocateCustom(expected, 25);
    EML_ASSERT_TENSOR_EQUALS(R, C);
}
inline int testMatMulCase38()
{
    qint32_t A_data[15]{} ;
	Tensor<qint32_t> A{ 3, 5 };
	A.allocateCustom(A_data, 15);
	arange(A,  1);

    qint32_t B_data[15]{} ;
	Tensor<qint32_t> B{ 3, 5 };
	B.allocateCustom(B_data, 15);
	arange(B,  1);

    qint32_t C_data[9]{} ;
	Tensor<qint32_t> C{ 3, 3 };
	C.allocateCustom(C_data, 9);

    matmulBTrans(A, B,C);
    qint32_t expected[] = { 55, 130, 205, 130, 330, 530, 205, 530, 855 };

    Tensor<qint32_t> R{C.shape()};
    R.allocateCustom(expected, 9);
    EML_ASSERT_TENSOR_EQUALS(R, C);
}
inline int testMatMulCase39()
{
    qint32_t A_data[100]{} ;
	Tensor<qint32_t> A{ 10, 10 };
	A.allocateCustom(A_data, 100);
	arange(A,  1);

    qint32_t B_data[100]{} ;
	Tensor<qint32_t> B{ 10, 10 };
	B.allocateCustom(B_data, 100);
	arange(B,  1);

    qint32_t C_data[100]{} ;
	Tensor<qint32_t> C{ 10, 10 };
	C.allocateCustom(C_data, 100);

    matmul(A, B,C);
    qint32_t expected[] = { 3355, 3410, 3465, 3520, 3575, 3630, 3685, 3740, 3795, 3850, 7955, 8110, 8265, 8420, 8575, 8730, 8885, 9040, 9195, 9350, 12555, 12810, 13065, 13320, 13575, 13830, 14085, 14340, 14595, 14850, 17155, 17510, 17865, 18220, 18575, 18930, 19285, 19640, 19995, 20350, 21755, 22210, 22665, 23120, 23575, 24030, 24485, 24940, 25395, 25850, 26355, 26910, 27465, 28020, 28575, 29130, 29685, 30240, 30795, 31350, 30955, 31610, 32265, 32920, 33575, 34230, 34885, 35540, 36195, 36850, 35555, 36310, 37065, 37820, 38575, 39330, 40085, 40840, 41595, 42350, 40155, 41010, 41865, 42720, 43575, 44430, 45285, 46140, 46995, 47850, 44755, 45710, 46665, 47620, 48575, 49530, 50485, 51440, 52395, 53350 };

    Tensor<qint32_t> R{C.shape()};
    R.allocateCustom(expected, 100);
    EML_ASSERT_TENSOR_EQUALS(R, C);
}
inline int testMatMulCase40()
{
    qint32_t A_data[15]{} ;
	Tensor<qint32_t> A{ 5, 3 };
	A.allocateCustom(A_data, 15);
	arange(A,  1);

    qint32_t B_data[15]{} ;
	Tensor<qint32_t> B{ 5, 3 };
	B.allocateCustom(B_data, 15);
	arange(B,  1);

    qint32_t C_data[9]{} ;
	Tensor<qint32_t> C{ 3, 3 };
	C.allocateCustom(C_data, 9);

    matmulATrans(A, B,C);
    qint32_t expected[] = { 335, 370, 405, 370, 410, 450, 405, 450, 495 };

    Tensor<qint32_t> R{C.shape()};
    R.allocateCustom(expected, 9);
    EML_ASSERT_TENSOR_EQUALS(R, C);
}
inline int testMatMulCase41()
{
    qint32_t A_data[64]{} ;
	Tensor<qint32_t> A{ 8, 8 };
	A.allocateCustom(A_data, 64);
	arange(A,  1);

    qint32_t B_data[64]{} ;
	Tensor<qint32_t> B{ 8, 8 };
	B.allocateCustom(B_data, 64);
	arange(B,  1);

    qint32_t C_data[64]{} ;
	Tensor<qint32_t> C{ 8, 8 };
	C.allocateCustom(C_data, 64);

    matmulBTrans(A, B,C);
    qint32_t expected[] = { 204, 492, 780, 1068, 1356, 1644, 1932, 2220, 492, 1292, 2092, 2892, 3692, 4492, 5292, 6092, 780, 2092, 3404, 4716, 6028, 7340, 8652, 9964, 1068, 2892, 4716, 6540, 8364, 10188, 12012, 13836, 1356, 3692, 6028, 8364, 10700, 13036, 15372, 17708, 1644, 4492, 7340, 10188, 13036, 15884, 18732, 21580, 1932, 5292, 8652, 12012, 15372, 18732, 22092, 25452, 2220, 6092, 9964, 13836, 17708, 21580, 25452, 29324 };

    Tensor<qint32_t> R{C.shape()};
    R.allocateCustom(expected, 64);
    EML_ASSERT_TENSOR_EQUALS(R, C);
}
inline int testMatMulCase42()
{
    qint32_t A_data[64]{} ;
	Tensor<qint32_t> A{ 8, 8 };
	A.allocateCustom(A_data, 64);
	arange(A,  1);

    qint32_t B_data[64]{} ;
	Tensor<qint32_t> B{ 8, 8 };
	B.allocateCustom(B_data, 64);
	arange(B,  1);

    qint32_t C_data[64]{} ;
	Tensor<qint32_t> C{ 8, 8 };
	C.allocateCustom(C_data, 64);

    matmul(A, B,C);
    qint32_t expected[] = { 1380, 1416, 1452, 1488, 1524, 1560, 1596, 1632, 3236, 3336, 3436, 3536, 3636, 3736, 3836, 3936, 5092, 5256, 5420, 5584, 5748, 5912, 6076, 6240, 6948, 7176, 7404, 7632, 7860, 8088, 8316, 8544, 8804, 9096, 9388, 9680, 9972, 10264, 10556, 10848, 10660, 11016, 11372, 11728, 12084, 12440, 12796, 13152, 12516, 12936, 13356, 13776, 14196, 14616, 15036, 15456, 14372, 14856, 15340, 15824, 16308, 16792, 17276, 17760 };

    Tensor<qint32_t> R{C.shape()};
    R.allocateCustom(expected, 64);
    EML_ASSERT_TENSOR_EQUALS(R, C);
}
inline int testMatMulCase43()
{
    qint32_t A_data[100]{} ;
	Tensor<qint32_t> A{ 10, 10 };
	A.allocateCustom(A_data, 100);
	arange(A,  1);

    qint32_t B_data[100]{} ;
	Tensor<qint32_t> B{ 10, 10 };
	B.allocateCustom(B_data, 100);
	arange(B,  1);

    qint32_t C_data[100]{} ;
	Tensor<qint32_t> C{ 10, 10 };
	C.allocateCustom(C_data, 100);

    matmulATrans(A, B,C);
    qint32_t expected[] = { 29410, 29870, 30330, 30790, 31250, 31710, 32170, 32630, 33090, 33550, 29870, 30340, 30810, 31280, 31750, 32220, 32690, 33160, 33630, 34100, 30330, 30810, 31290, 31770, 32250, 32730, 33210, 33690, 34170, 34650, 30790, 31280, 31770, 32260, 32750, 33240, 33730, 34220, 34710, 35200, 31250, 31750, 32250, 32750, 33250, 33750, 34250, 34750, 35250, 35750, 31710, 32220, 32730, 33240, 33750, 34260, 34770, 35280, 35790, 36300, 32170, 32690, 33210, 33730, 34250, 34770, 35290, 35810, 36330, 36850, 32630, 33160, 33690, 34220, 34750, 35280, 35810, 36340, 36870, 37400, 33090, 33630, 34170, 34710, 35250, 35790, 36330, 36870, 37410, 37950, 33550, 34100, 34650, 35200, 35750, 36300, 36850, 37400, 37950, 38500 };

    Tensor<qint32_t> R{C.shape()};
    R.allocateCustom(expected, 100);
    EML_ASSERT_TENSOR_EQUALS(R, C);
}
inline int testMatMulCase44()
{
    qint32_t A_data[15]{} ;
	Tensor<qint32_t> A{ 5, 3 };
	A.allocateCustom(A_data, 15);
	arange(A,  1);

    qint32_t B_data[15]{} ;
	Tensor<qint32_t> B{ 5, 3 };
	B.allocateCustom(B_data, 15);
	arange(B,  1);

    qint32_t C_data[25]{} ;
	Tensor<qint32_t> C{ 5, 5 };
	C.allocateCustom(C_data, 25);

    matmulBTrans(A, B,C);
    qint32_t expected[] = { 14, 32, 50, 68, 86, 32, 77, 122, 167, 212, 50, 122, 194, 266, 338, 68, 167, 266, 365, 464, 86, 212, 338, 464, 590 };

    Tensor<qint32_t> R{C.shape()};
    R.allocateCustom(expected, 25);
    EML_ASSERT_TENSOR_EQUALS(R, C);
}

inline void TestMATHMatMul()
{	EML_RUN_TEST( testMatMulCase0 );
	EML_RUN_TEST( testMatMulCase1 );
	EML_RUN_TEST( testMatMulCase2 );
	EML_RUN_TEST( testMatMulCase3 );
	EML_RUN_TEST( testMatMulCase4 );
	EML_RUN_TEST( testMatMulCase5 );
	EML_RUN_TEST( testMatMulCase6 );
	EML_RUN_TEST( testMatMulCase7 );
	EML_RUN_TEST( testMatMulCase8 );
	EML_RUN_TEST( testMatMulCase9 );
	EML_RUN_TEST( testMatMulCase10 );
	EML_RUN_TEST( testMatMulCase11 );
	EML_RUN_TEST( testMatMulCase12 );
	EML_RUN_TEST( testMatMulCase13 );
	EML_RUN_TEST( testMatMulCase14 );
	EML_RUN_TEST( testMatMulCase15 );
	EML_RUN_TEST( testMatMulCase16 );
	EML_RUN_TEST( testMatMulCase17 );
	EML_RUN_TEST( testMatMulCase18 );
	EML_RUN_TEST( testMatMulCase19 );
	EML_RUN_TEST( testMatMulCase20 );
	EML_RUN_TEST( testMatMulCase21 );
	EML_RUN_TEST( testMatMulCase22 );
	EML_RUN_TEST( testMatMulCase23 );
	EML_RUN_TEST( testMatMulCase24 );
	EML_RUN_TEST( testMatMulCase25 );
#ifndef EML_TINY_TESTBENCH
	EML_RUN_TEST( testMatMulCase26 );
	EML_RUN_TEST( testMatMulCase27 );
	EML_RUN_TEST( testMatMulCase28 );
	EML_RUN_TEST( testMatMulCase29 );
	EML_RUN_TEST( testMatMulCase30 );
	EML_RUN_TEST( testMatMulCase31 );
	EML_RUN_TEST( testMatMulCase32 );
	EML_RUN_TEST( testMatMulCase33 );
	EML_RUN_TEST( testMatMulCase34 );
	EML_RUN_TEST( testMatMulCase35 );
	EML_RUN_TEST( testMatMulCase36 );
	EML_RUN_TEST( testMatMulCase37 );
	EML_RUN_TEST( testMatMulCase38 );
	EML_RUN_TEST( testMatMulCase39 );
	EML_RUN_TEST( testMatMulCase40 );
	EML_RUN_TEST( testMatMulCase41 );
	EML_RUN_TEST( testMatMulCase42 );
	EML_RUN_TEST( testMatMulCase43 );
	EML_RUN_TEST( testMatMulCase44 );
#endif
}

#endif // EML_TEST_MATH_MATMUL_H
    