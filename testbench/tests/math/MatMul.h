
#ifndef EML_TEST_MATH_MATMUL_H
#define EML_TEST_MATH_MATMUL_H

inline int testMatMulCase0()
{
    qint32_t A_data[16] ;
	Tensor<qint32_t> A{ 4, 4 };
	A.allocateCustom(A_data, 16);
	Arange(A,1);

    qint32_t B_data[16] ;
	Tensor<qint32_t> B{ 4, 4 };
	B.allocateCustom(B_data, 16);
	Arange(B,1);

    qint32_t C_data[16] ;
	Tensor<qint32_t> C{ 4, 4 };
	C.allocateCustom(C_data, 16);

    MatmulBTrans(A, B,C);
    qint32_t expected[] = { 30, 70, 110, 150, 70, 174, 278, 382, 110, 278, 446, 614, 150, 382, 614, 846 };

    Tensor<qint32_t> R{C.shape()};
    R.allocateCustom(expected, 16);
    EML_ASSERT_TENSOR_EQUALS(R, C);
}
inline int testMatMulCase1()
{
    qint32_t A_data[100] ;
	Tensor<qint32_t> A{ 10, 10 };
	A.allocateCustom(A_data, 100);
	Arange(A,1);

    qint32_t B_data[100] ;
	Tensor<qint32_t> B{ 10, 10 };
	B.allocateCustom(B_data, 100);
	Arange(B,1);

    qint32_t C_data[100] ;
	Tensor<qint32_t> C{ 10, 10 };
	C.allocateCustom(C_data, 100);

    Matmul(A, B,C);
    qint32_t expected[] = { 3355, 3410, 3465, 3520, 3575, 3630, 3685, 3740, 3795, 3850, 7955, 8110, 8265, 8420, 8575, 8730, 8885, 9040, 9195, 9350, 12555, 12810, 13065, 13320, 13575, 13830, 14085, 14340, 14595, 14850, 17155, 17510, 17865, 18220, 18575, 18930, 19285, 19640, 19995, 20350, 21755, 22210, 22665, 23120, 23575, 24030, 24485, 24940, 25395, 25850, 26355, 26910, 27465, 28020, 28575, 29130, 29685, 30240, 30795, 31350, 30955, 31610, 32265, 32920, 33575, 34230, 34885, 35540, 36195, 36850, 35555, 36310, 37065, 37820, 38575, 39330, 40085, 40840, 41595, 42350, 40155, 41010, 41865, 42720, 43575, 44430, 45285, 46140, 46995, 47850, 44755, 45710, 46665, 47620, 48575, 49530, 50485, 51440, 52395, 53350 };

    Tensor<qint32_t> R{C.shape()};
    R.allocateCustom(expected, 100);
    EML_ASSERT_TENSOR_EQUALS(R, C);
}
inline int testMatMulCase2()
{
    qint32_t A_data[15] ;
	Tensor<qint32_t> A{ 3, 5 };
	A.allocateCustom(A_data, 15);
	Arange(A,1);

    qint32_t B_data[15] ;
	Tensor<qint32_t> B{ 3, 5 };
	B.allocateCustom(B_data, 15);
	Arange(B,1);

    qint32_t C_data[9] ;
	Tensor<qint32_t> C{ 3, 3 };
	C.allocateCustom(C_data, 9);

    MatmulBTrans(A, B,C);
    qint32_t expected[] = { 55, 130, 205, 130, 330, 530, 205, 530, 855 };

    Tensor<qint32_t> R{C.shape()};
    R.allocateCustom(expected, 9);
    EML_ASSERT_TENSOR_EQUALS(R, C);
}
inline int testMatMulCase3()
{
    qint32_t A_data[15] ;
	Tensor<qint32_t> A{ 3, 5 };
	A.allocateCustom(A_data, 15);
	Arange(A,1);

    qint32_t B_data[15] ;
	Tensor<qint32_t> B{ 3, 5 };
	B.allocateCustom(B_data, 15);
	Arange(B,1);

    qint32_t C_data[9] ;
	Tensor<qint32_t> C{ 3, 3 };
	C.allocateCustom(C_data, 9);

    MatmulBTrans(A, B,C);
    qint32_t expected[] = { 55, 130, 205, 130, 330, 530, 205, 530, 855 };

    Tensor<qint32_t> R{C.shape()};
    R.allocateCustom(expected, 9);
    EML_ASSERT_TENSOR_EQUALS(R, C);
}
inline int testMatMulCase4()
{
    qint32_t A_data[25] ;
	Tensor<qint32_t> A{ 5, 5 };
	A.allocateCustom(A_data, 25);
	Arange(A,1);

    qint32_t B_data[25] ;
	Tensor<qint32_t> B{ 5, 5 };
	B.allocateCustom(B_data, 25);
	Arange(B,1);

    qint32_t C_data[25] ;
	Tensor<qint32_t> C{ 5, 5 };
	C.allocateCustom(C_data, 25);

    MatmulATrans(A, B,C);
    qint32_t expected[] = { 855, 910, 965, 1020, 1075, 910, 970, 1030, 1090, 1150, 965, 1030, 1095, 1160, 1225, 1020, 1090, 1160, 1230, 1300, 1075, 1150, 1225, 1300, 1375 };

    Tensor<qint32_t> R{C.shape()};
    R.allocateCustom(expected, 25);
    EML_ASSERT_TENSOR_EQUALS(R, C);
}
inline int testMatMulCase5()
{
    qint32_t A_data[100] ;
	Tensor<qint32_t> A{ 10, 10 };
	A.allocateCustom(A_data, 100);
	Arange(A,1);

    qint32_t B_data[100] ;
	Tensor<qint32_t> B{ 10, 10 };
	B.allocateCustom(B_data, 100);
	Arange(B,1);

    qint32_t C_data[100] ;
	Tensor<qint32_t> C{ 10, 10 };
	C.allocateCustom(C_data, 100);

    Matmul(A, B,C);
    qint32_t expected[] = { 3355, 3410, 3465, 3520, 3575, 3630, 3685, 3740, 3795, 3850, 7955, 8110, 8265, 8420, 8575, 8730, 8885, 9040, 9195, 9350, 12555, 12810, 13065, 13320, 13575, 13830, 14085, 14340, 14595, 14850, 17155, 17510, 17865, 18220, 18575, 18930, 19285, 19640, 19995, 20350, 21755, 22210, 22665, 23120, 23575, 24030, 24485, 24940, 25395, 25850, 26355, 26910, 27465, 28020, 28575, 29130, 29685, 30240, 30795, 31350, 30955, 31610, 32265, 32920, 33575, 34230, 34885, 35540, 36195, 36850, 35555, 36310, 37065, 37820, 38575, 39330, 40085, 40840, 41595, 42350, 40155, 41010, 41865, 42720, 43575, 44430, 45285, 46140, 46995, 47850, 44755, 45710, 46665, 47620, 48575, 49530, 50485, 51440, 52395, 53350 };

    Tensor<qint32_t> R{C.shape()};
    R.allocateCustom(expected, 100);
    EML_ASSERT_TENSOR_EQUALS(R, C);
}
inline int testMatMulCase6()
{
    qint32_t A_data[15] ;
	Tensor<qint32_t> A{ 3, 5 };
	A.allocateCustom(A_data, 15);
	Arange(A,1);

    qint32_t B_data[15] ;
	Tensor<qint32_t> B{ 3, 5 };
	B.allocateCustom(B_data, 15);
	Arange(B,1);

    qint32_t C_data[25] ;
	Tensor<qint32_t> C{ 5, 5 };
	C.allocateCustom(C_data, 25);

    MatmulATrans(A, B,C);
    qint32_t expected[] = { 158, 176, 194, 212, 230, 176, 197, 218, 239, 260, 194, 218, 242, 266, 290, 212, 239, 266, 293, 320, 230, 260, 290, 320, 350 };

    Tensor<qint32_t> R{C.shape()};
    R.allocateCustom(expected, 25);
    EML_ASSERT_TENSOR_EQUALS(R, C);
}
inline int testMatMulCase7()
{
    qint32_t A_data[100] ;
	Tensor<qint32_t> A{ 10, 10 };
	A.allocateCustom(A_data, 100);
	Arange(A,1);

    qint32_t B_data[100] ;
	Tensor<qint32_t> B{ 10, 10 };
	B.allocateCustom(B_data, 100);
	Arange(B,1);

    qint32_t C_data[100] ;
	Tensor<qint32_t> C{ 10, 10 };
	C.allocateCustom(C_data, 100);

    Matmul(A, B,C);
    qint32_t expected[] = { 3355, 3410, 3465, 3520, 3575, 3630, 3685, 3740, 3795, 3850, 7955, 8110, 8265, 8420, 8575, 8730, 8885, 9040, 9195, 9350, 12555, 12810, 13065, 13320, 13575, 13830, 14085, 14340, 14595, 14850, 17155, 17510, 17865, 18220, 18575, 18930, 19285, 19640, 19995, 20350, 21755, 22210, 22665, 23120, 23575, 24030, 24485, 24940, 25395, 25850, 26355, 26910, 27465, 28020, 28575, 29130, 29685, 30240, 30795, 31350, 30955, 31610, 32265, 32920, 33575, 34230, 34885, 35540, 36195, 36850, 35555, 36310, 37065, 37820, 38575, 39330, 40085, 40840, 41595, 42350, 40155, 41010, 41865, 42720, 43575, 44430, 45285, 46140, 46995, 47850, 44755, 45710, 46665, 47620, 48575, 49530, 50485, 51440, 52395, 53350 };

    Tensor<qint32_t> R{C.shape()};
    R.allocateCustom(expected, 100);
    EML_ASSERT_TENSOR_EQUALS(R, C);
}
inline int testMatMulCase8()
{
    qint32_t A_data[16] ;
	Tensor<qint32_t> A{ 4, 4 };
	A.allocateCustom(A_data, 16);
	Arange(A,1);

    qint32_t B_data[16] ;
	Tensor<qint32_t> B{ 4, 4 };
	B.allocateCustom(B_data, 16);
	Arange(B,1);

    qint32_t C_data[16] ;
	Tensor<qint32_t> C{ 4, 4 };
	C.allocateCustom(C_data, 16);

    MatmulBTrans(A, B,C);
    qint32_t expected[] = { 30, 70, 110, 150, 70, 174, 278, 382, 110, 278, 446, 614, 150, 382, 614, 846 };

    Tensor<qint32_t> R{C.shape()};
    R.allocateCustom(expected, 16);
    EML_ASSERT_TENSOR_EQUALS(R, C);
}
inline int testMatMulCase9()
{
    qint32_t A_data[100] ;
	Tensor<qint32_t> A{ 10, 10 };
	A.allocateCustom(A_data, 100);
	Arange(A,1);

    qint32_t B_data[100] ;
	Tensor<qint32_t> B{ 10, 10 };
	B.allocateCustom(B_data, 100);
	Arange(B,1);

    qint32_t C_data[100] ;
	Tensor<qint32_t> C{ 10, 10 };
	C.allocateCustom(C_data, 100);

    Matmul(A, B,C);
    qint32_t expected[] = { 3355, 3410, 3465, 3520, 3575, 3630, 3685, 3740, 3795, 3850, 7955, 8110, 8265, 8420, 8575, 8730, 8885, 9040, 9195, 9350, 12555, 12810, 13065, 13320, 13575, 13830, 14085, 14340, 14595, 14850, 17155, 17510, 17865, 18220, 18575, 18930, 19285, 19640, 19995, 20350, 21755, 22210, 22665, 23120, 23575, 24030, 24485, 24940, 25395, 25850, 26355, 26910, 27465, 28020, 28575, 29130, 29685, 30240, 30795, 31350, 30955, 31610, 32265, 32920, 33575, 34230, 34885, 35540, 36195, 36850, 35555, 36310, 37065, 37820, 38575, 39330, 40085, 40840, 41595, 42350, 40155, 41010, 41865, 42720, 43575, 44430, 45285, 46140, 46995, 47850, 44755, 45710, 46665, 47620, 48575, 49530, 50485, 51440, 52395, 53350 };

    Tensor<qint32_t> R{C.shape()};
    R.allocateCustom(expected, 100);
    EML_ASSERT_TENSOR_EQUALS(R, C);
}
inline int testMatMulCase10()
{
    qint32_t A_data[16] ;
	Tensor<qint32_t> A{ 4, 4 };
	A.allocateCustom(A_data, 16);
	Arange(A,1);

    qint32_t B_data[16] ;
	Tensor<qint32_t> B{ 4, 4 };
	B.allocateCustom(B_data, 16);
	Arange(B,1);

    qint32_t C_data[16] ;
	Tensor<qint32_t> C{ 4, 4 };
	C.allocateCustom(C_data, 16);

    Matmul(A, B,C);
    qint32_t expected[] = { 90, 100, 110, 120, 202, 228, 254, 280, 314, 356, 398, 440, 426, 484, 542, 600 };

    Tensor<qint32_t> R{C.shape()};
    R.allocateCustom(expected, 16);
    EML_ASSERT_TENSOR_EQUALS(R, C);
}
inline int testMatMulCase11()
{
    qint32_t A_data[16] ;
	Tensor<qint32_t> A{ 4, 4 };
	A.allocateCustom(A_data, 16);
	Arange(A,1);

    qint32_t B_data[16] ;
	Tensor<qint32_t> B{ 4, 4 };
	B.allocateCustom(B_data, 16);
	Arange(B,1);

    qint32_t C_data[16] ;
	Tensor<qint32_t> C{ 4, 4 };
	C.allocateCustom(C_data, 16);

    Matmul(A, B,C);
    qint32_t expected[] = { 90, 100, 110, 120, 202, 228, 254, 280, 314, 356, 398, 440, 426, 484, 542, 600 };

    Tensor<qint32_t> R{C.shape()};
    R.allocateCustom(expected, 16);
    EML_ASSERT_TENSOR_EQUALS(R, C);
}
inline int testMatMulCase12()
{
    qint32_t A_data[100] ;
	Tensor<qint32_t> A{ 10, 10 };
	A.allocateCustom(A_data, 100);
	Arange(A,1);

    qint32_t B_data[100] ;
	Tensor<qint32_t> B{ 10, 10 };
	B.allocateCustom(B_data, 100);
	Arange(B,1);

    qint32_t C_data[100] ;
	Tensor<qint32_t> C{ 10, 10 };
	C.allocateCustom(C_data, 100);

    Matmul(A, B,C);
    qint32_t expected[] = { 3355, 3410, 3465, 3520, 3575, 3630, 3685, 3740, 3795, 3850, 7955, 8110, 8265, 8420, 8575, 8730, 8885, 9040, 9195, 9350, 12555, 12810, 13065, 13320, 13575, 13830, 14085, 14340, 14595, 14850, 17155, 17510, 17865, 18220, 18575, 18930, 19285, 19640, 19995, 20350, 21755, 22210, 22665, 23120, 23575, 24030, 24485, 24940, 25395, 25850, 26355, 26910, 27465, 28020, 28575, 29130, 29685, 30240, 30795, 31350, 30955, 31610, 32265, 32920, 33575, 34230, 34885, 35540, 36195, 36850, 35555, 36310, 37065, 37820, 38575, 39330, 40085, 40840, 41595, 42350, 40155, 41010, 41865, 42720, 43575, 44430, 45285, 46140, 46995, 47850, 44755, 45710, 46665, 47620, 48575, 49530, 50485, 51440, 52395, 53350 };

    Tensor<qint32_t> R{C.shape()};
    R.allocateCustom(expected, 100);
    EML_ASSERT_TENSOR_EQUALS(R, C);
}
inline int testMatMulCase13()
{
    qint32_t A_data[16] ;
	Tensor<qint32_t> A{ 4, 4 };
	A.allocateCustom(A_data, 16);
	Arange(A,1);

    qint32_t B_data[16] ;
	Tensor<qint32_t> B{ 4, 4 };
	B.allocateCustom(B_data, 16);
	Arange(B,1);

    qint32_t C_data[16] ;
	Tensor<qint32_t> C{ 4, 4 };
	C.allocateCustom(C_data, 16);

    Matmul(A, B,C);
    qint32_t expected[] = { 90, 100, 110, 120, 202, 228, 254, 280, 314, 356, 398, 440, 426, 484, 542, 600 };

    Tensor<qint32_t> R{C.shape()};
    R.allocateCustom(expected, 16);
    EML_ASSERT_TENSOR_EQUALS(R, C);
}
inline int testMatMulCase14()
{
    qint32_t A_data[15] ;
	Tensor<qint32_t> A{ 5, 3 };
	A.allocateCustom(A_data, 15);
	Arange(A,1);

    qint32_t B_data[15] ;
	Tensor<qint32_t> B{ 5, 3 };
	B.allocateCustom(B_data, 15);
	Arange(B,1);

    qint32_t C_data[9] ;
	Tensor<qint32_t> C{ 3, 3 };
	C.allocateCustom(C_data, 9);

    MatmulATrans(A, B,C);
    qint32_t expected[] = { 335, 370, 405, 370, 410, 450, 405, 450, 495 };

    Tensor<qint32_t> R{C.shape()};
    R.allocateCustom(expected, 9);
    EML_ASSERT_TENSOR_EQUALS(R, C);
}
inline int testMatMulCase15()
{
    qint32_t A_data[25] ;
	Tensor<qint32_t> A{ 5, 5 };
	A.allocateCustom(A_data, 25);
	Arange(A,1);

    qint32_t B_data[25] ;
	Tensor<qint32_t> B{ 5, 5 };
	B.allocateCustom(B_data, 25);
	Arange(B,1);

    qint32_t C_data[25] ;
	Tensor<qint32_t> C{ 5, 5 };
	C.allocateCustom(C_data, 25);

    MatmulBTrans(A, B,C);
    qint32_t expected[] = { 55, 130, 205, 280, 355, 130, 330, 530, 730, 930, 205, 530, 855, 1180, 1505, 280, 730, 1180, 1630, 2080, 355, 930, 1505, 2080, 2655 };

    Tensor<qint32_t> R{C.shape()};
    R.allocateCustom(expected, 25);
    EML_ASSERT_TENSOR_EQUALS(R, C);
}
inline int testMatMulCase16()
{
    qint32_t A_data[16] ;
	Tensor<qint32_t> A{ 4, 4 };
	A.allocateCustom(A_data, 16);
	Arange(A,1);

    qint32_t B_data[16] ;
	Tensor<qint32_t> B{ 4, 4 };
	B.allocateCustom(B_data, 16);
	Arange(B,1);

    qint32_t C_data[16] ;
	Tensor<qint32_t> C{ 4, 4 };
	C.allocateCustom(C_data, 16);

    Matmul(A, B,C);
    qint32_t expected[] = { 90, 100, 110, 120, 202, 228, 254, 280, 314, 356, 398, 440, 426, 484, 542, 600 };

    Tensor<qint32_t> R{C.shape()};
    R.allocateCustom(expected, 16);
    EML_ASSERT_TENSOR_EQUALS(R, C);
}
inline int testMatMulCase17()
{
    qint32_t A_data[16] ;
	Tensor<qint32_t> A{ 4, 4 };
	A.allocateCustom(A_data, 16);
	Arange(A,1);

    qint32_t B_data[16] ;
	Tensor<qint32_t> B{ 4, 4 };
	B.allocateCustom(B_data, 16);
	Arange(B,1);

    qint32_t C_data[16] ;
	Tensor<qint32_t> C{ 4, 4 };
	C.allocateCustom(C_data, 16);

    Matmul(A, B,C);
    qint32_t expected[] = { 90, 100, 110, 120, 202, 228, 254, 280, 314, 356, 398, 440, 426, 484, 542, 600 };

    Tensor<qint32_t> R{C.shape()};
    R.allocateCustom(expected, 16);
    EML_ASSERT_TENSOR_EQUALS(R, C);
}
inline int testMatMulCase18()
{
    qint32_t A_data[16] ;
	Tensor<qint32_t> A{ 4, 4 };
	A.allocateCustom(A_data, 16);
	Arange(A,1);

    qint32_t B_data[16] ;
	Tensor<qint32_t> B{ 4, 4 };
	B.allocateCustom(B_data, 16);
	Arange(B,1);

    qint32_t C_data[16] ;
	Tensor<qint32_t> C{ 4, 4 };
	C.allocateCustom(C_data, 16);

    Matmul(A, B,C);
    qint32_t expected[] = { 90, 100, 110, 120, 202, 228, 254, 280, 314, 356, 398, 440, 426, 484, 542, 600 };

    Tensor<qint32_t> R{C.shape()};
    R.allocateCustom(expected, 16);
    EML_ASSERT_TENSOR_EQUALS(R, C);
}
inline int testMatMulCase19()
{
    qint32_t A_data[15] ;
	Tensor<qint32_t> A{ 5, 3 };
	A.allocateCustom(A_data, 15);
	Arange(A,1);

    qint32_t B_data[15] ;
	Tensor<qint32_t> B{ 3, 5 };
	B.allocateCustom(B_data, 15);
	Arange(B,1);

    qint32_t C_data[25] ;
	Tensor<qint32_t> C{ 5, 5 };
	C.allocateCustom(C_data, 25);

    Matmul(A, B,C);
    qint32_t expected[] = { 46, 52, 58, 64, 70, 100, 115, 130, 145, 160, 154, 178, 202, 226, 250, 208, 241, 274, 307, 340, 262, 304, 346, 388, 430 };

    Tensor<qint32_t> R{C.shape()};
    R.allocateCustom(expected, 25);
    EML_ASSERT_TENSOR_EQUALS(R, C);
}
inline int testMatMulCase20()
{
    qint32_t A_data[100] ;
	Tensor<qint32_t> A{ 10, 10 };
	A.allocateCustom(A_data, 100);
	Arange(A,1);

    qint32_t B_data[100] ;
	Tensor<qint32_t> B{ 10, 10 };
	B.allocateCustom(B_data, 100);
	Arange(B,1);

    qint32_t C_data[100] ;
	Tensor<qint32_t> C{ 10, 10 };
	C.allocateCustom(C_data, 100);

    Matmul(A, B,C);
    qint32_t expected[] = { 3355, 3410, 3465, 3520, 3575, 3630, 3685, 3740, 3795, 3850, 7955, 8110, 8265, 8420, 8575, 8730, 8885, 9040, 9195, 9350, 12555, 12810, 13065, 13320, 13575, 13830, 14085, 14340, 14595, 14850, 17155, 17510, 17865, 18220, 18575, 18930, 19285, 19640, 19995, 20350, 21755, 22210, 22665, 23120, 23575, 24030, 24485, 24940, 25395, 25850, 26355, 26910, 27465, 28020, 28575, 29130, 29685, 30240, 30795, 31350, 30955, 31610, 32265, 32920, 33575, 34230, 34885, 35540, 36195, 36850, 35555, 36310, 37065, 37820, 38575, 39330, 40085, 40840, 41595, 42350, 40155, 41010, 41865, 42720, 43575, 44430, 45285, 46140, 46995, 47850, 44755, 45710, 46665, 47620, 48575, 49530, 50485, 51440, 52395, 53350 };

    Tensor<qint32_t> R{C.shape()};
    R.allocateCustom(expected, 100);
    EML_ASSERT_TENSOR_EQUALS(R, C);
}
inline int testMatMulCase21()
{
    qint32_t A_data[15] ;
	Tensor<qint32_t> A{ 3, 5 };
	A.allocateCustom(A_data, 15);
	Arange(A,1);

    qint32_t B_data[15] ;
	Tensor<qint32_t> B{ 5, 3 };
	B.allocateCustom(B_data, 15);
	Arange(B,1);

    qint32_t C_data[9] ;
	Tensor<qint32_t> C{ 3, 3 };
	C.allocateCustom(C_data, 9);

    Matmul(A, B,C);
    qint32_t expected[] = { 135, 150, 165, 310, 350, 390, 485, 550, 615 };

    Tensor<qint32_t> R{C.shape()};
    R.allocateCustom(expected, 9);
    EML_ASSERT_TENSOR_EQUALS(R, C);
}
inline int testMatMulCase22()
{
    qint32_t A_data[16] ;
	Tensor<qint32_t> A{ 4, 4 };
	A.allocateCustom(A_data, 16);
	Arange(A,1);

    qint32_t B_data[16] ;
	Tensor<qint32_t> B{ 4, 4 };
	B.allocateCustom(B_data, 16);
	Arange(B,1);

    qint32_t C_data[16] ;
	Tensor<qint32_t> C{ 4, 4 };
	C.allocateCustom(C_data, 16);

    MatmulBTrans(A, B,C);
    qint32_t expected[] = { 30, 70, 110, 150, 70, 174, 278, 382, 110, 278, 446, 614, 150, 382, 614, 846 };

    Tensor<qint32_t> R{C.shape()};
    R.allocateCustom(expected, 16);
    EML_ASSERT_TENSOR_EQUALS(R, C);
}
inline int testMatMulCase23()
{
    qint32_t A_data[15] ;
	Tensor<qint32_t> A{ 3, 5 };
	A.allocateCustom(A_data, 15);
	Arange(A,1);

    qint32_t B_data[15] ;
	Tensor<qint32_t> B{ 5, 3 };
	B.allocateCustom(B_data, 15);
	Arange(B,1);

    qint32_t C_data[9] ;
	Tensor<qint32_t> C{ 3, 3 };
	C.allocateCustom(C_data, 9);

    Matmul(A, B,C);
    qint32_t expected[] = { 135, 150, 165, 310, 350, 390, 485, 550, 615 };

    Tensor<qint32_t> R{C.shape()};
    R.allocateCustom(expected, 9);
    EML_ASSERT_TENSOR_EQUALS(R, C);
}
inline int testMatMulCase24()
{
    qint32_t A_data[25] ;
	Tensor<qint32_t> A{ 5, 5 };
	A.allocateCustom(A_data, 25);
	Arange(A,1);

    qint32_t B_data[25] ;
	Tensor<qint32_t> B{ 5, 5 };
	B.allocateCustom(B_data, 25);
	Arange(B,1);

    qint32_t C_data[25] ;
	Tensor<qint32_t> C{ 5, 5 };
	C.allocateCustom(C_data, 25);

    MatmulBTrans(A, B,C);
    qint32_t expected[] = { 55, 130, 205, 280, 355, 130, 330, 530, 730, 930, 205, 530, 855, 1180, 1505, 280, 730, 1180, 1630, 2080, 355, 930, 1505, 2080, 2655 };

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

}

#endif // EML_TEST_MATH_MATMUL_H
    