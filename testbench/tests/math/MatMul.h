
#ifndef EML_TEST_MATH_MATMUL_H
#define EML_TEST_MATH_MATMUL_H

inline int testMatMulCase0()
{
    qint32_t A_data[] = { 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15, 4, 8, 12, 16 };
    qint32_t B_data[] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 };
    qint32_t C_data[16] ;

    Tensor<qint32_t> A{ 4, 4 };
A.allocateCustom(A_data, 16);
    Tensor<qint32_t> B{ 4, 4 };
B.allocateCustom(B_data, 16);
    Tensor<qint32_t> C{ 4, 4 };
C.allocateCustom(C_data, 16);
    MatmulATrans(A, B,C);
    C.print();
    qint32_t expected[] = { 276, 304, 332, 360, 304, 336, 368, 400, 332, 368, 404, 440, 360, 400, 440, 480 };
    Tensor<qint32_t> R{C.shape()};
    R.allocateCustom(expected, 16);
    R.print();
    return Equals(C, R);

}
inline int testMatMulCase1()
{
    qint32_t A_data[] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 };
    qint32_t B_data[] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 };
    qint32_t C_data[9] ;

    Tensor<qint32_t> A{ 3, 5 };
A.allocateCustom(A_data, 15);
    Tensor<qint32_t> B{ 5, 3 };
B.allocateCustom(B_data, 15);
    Tensor<qint32_t> C{ 3, 3 };
C.allocateCustom(C_data, 9);
    Matmul(A, B,C);
    qint32_t expected[] = { 135, 150, 165, 310, 350, 390, 485, 550, 615 };
    Tensor<qint32_t> R{C.shape()};
    R.allocateCustom(expected, 9);
    return Equals(C, R);

}
inline int testMatMulCase2()
{
    qint32_t A_data[] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 };
    qint32_t B_data[] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 };
    qint32_t C_data[25] ;

    Tensor<qint32_t> A{ 5, 3 };
A.allocateCustom(A_data, 15);
    Tensor<qint32_t> B{ 3, 5 };
B.allocateCustom(B_data, 15);
    Tensor<qint32_t> C{ 5, 5 };
C.allocateCustom(C_data, 25);
    Matmul(A, B,C);
    qint32_t expected[] = { 46, 52, 58, 64, 70, 100, 115, 130, 145, 160, 154, 178, 202, 226, 250, 208, 241, 274, 307, 340, 262, 304, 346, 388, 430 };
    Tensor<qint32_t> R{C.shape()};
    R.allocateCustom(expected, 25);
    return Equals(C, R);

}
inline int testMatMulCase3()
{
    qint32_t A_data[] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25 };
    qint32_t B_data[] = { 1, 6, 11, 16, 21, 2, 7, 12, 17, 22, 3, 8, 13, 18, 23, 4, 9, 14, 19, 24, 5, 10, 15, 20, 25 };
    qint32_t C_data[25] ;

    Tensor<qint32_t> A{ 5, 5 };
A.allocateCustom(A_data, 25);
    Tensor<qint32_t> B{ 5, 5 };
B.allocateCustom(B_data, 25);
    Tensor<qint32_t> C{ 5, 5 };
C.allocateCustom(C_data, 25);
    MatmulBTrans(A, B,C);
    qint32_t expected[] = { 55, 130, 205, 280, 355, 130, 330, 530, 730, 930, 205, 530, 855, 1180, 1505, 280, 730, 1180, 1630, 2080, 355, 930, 1505, 2080, 2655 };
    Tensor<qint32_t> R{C.shape()};
    R.allocateCustom(expected, 25);
    return Equals(C, R);

}
inline int testMatMulCase4()
{
    qint32_t A_data[] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 };
    qint32_t B_data[] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 };
    qint32_t C_data[16] ;

    Tensor<qint32_t> A{ 4, 4 };
A.allocateCustom(A_data, 16);
    Tensor<qint32_t> B{ 4, 4 };
B.allocateCustom(B_data, 16);
    Tensor<qint32_t> C{ 4, 4 };
C.allocateCustom(C_data, 16);
    Matmul(A, B,C);
    qint32_t expected[] = { 90, 100, 110, 120, 202, 228, 254, 280, 314, 356, 398, 440, 426, 484, 542, 600 };
    Tensor<qint32_t> R{C.shape()};
    R.allocateCustom(expected, 16);
    return Equals(C, R);

}
inline int testMatMulCase5()
{
    qint32_t A_data[] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 };
    qint32_t B_data[] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 };
    qint32_t C_data[9] ;

    Tensor<qint32_t> A{ 3, 5 };
A.allocateCustom(A_data, 15);
    Tensor<qint32_t> B{ 5, 3 };
B.allocateCustom(B_data, 15);
    Tensor<qint32_t> C{ 3, 3 };
C.allocateCustom(C_data, 9);
    Matmul(A, B,C);
    qint32_t expected[] = { 135, 150, 165, 310, 350, 390, 485, 550, 615 };
    Tensor<qint32_t> R{C.shape()};
    R.allocateCustom(expected, 9);
    return Equals(C, R);

}
inline int testMatMulCase6()
{
    qint32_t A_data[] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 };
    qint32_t B_data[] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 };
    qint32_t C_data[25] ;

    Tensor<qint32_t> A{ 5, 3 };
A.allocateCustom(A_data, 15);
    Tensor<qint32_t> B{ 3, 5 };
B.allocateCustom(B_data, 15);
    Tensor<qint32_t> C{ 5, 5 };
C.allocateCustom(C_data, 25);
    Matmul(A, B,C);
    qint32_t expected[] = { 46, 52, 58, 64, 70, 100, 115, 130, 145, 160, 154, 178, 202, 226, 250, 208, 241, 274, 307, 340, 262, 304, 346, 388, 430 };
    Tensor<qint32_t> R{C.shape()};
    R.allocateCustom(expected, 25);
    return Equals(C, R);

}
inline int testMatMulCase7()
{
    qint32_t A_data[] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25 };
    qint32_t B_data[] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25 };
    qint32_t C_data[25] ;

    Tensor<qint32_t> A{ 5, 5 };
A.allocateCustom(A_data, 25);
    Tensor<qint32_t> B{ 5, 5 };
B.allocateCustom(B_data, 25);
    Tensor<qint32_t> C{ 5, 5 };
C.allocateCustom(C_data, 25);
    Matmul(A, B,C);
    qint32_t expected[] = { 215, 230, 245, 260, 275, 490, 530, 570, 610, 650, 765, 830, 895, 960, 1025, 1040, 1130, 1220, 1310, 1400, 1315, 1430, 1545, 1660, 1775 };
    Tensor<qint32_t> R{C.shape()};
    R.allocateCustom(expected, 25);
    return Equals(C, R);

}
inline int testMatMulCase8()
{
    qint32_t A_data[] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 };
    qint32_t B_data[] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 };
    qint32_t C_data[16] ;

    Tensor<qint32_t> A{ 4, 4 };
A.allocateCustom(A_data, 16);
    Tensor<qint32_t> B{ 4, 4 };
B.allocateCustom(B_data, 16);
    Tensor<qint32_t> C{ 4, 4 };
C.allocateCustom(C_data, 16);
    Matmul(A, B,C);
    qint32_t expected[] = { 90, 100, 110, 120, 202, 228, 254, 280, 314, 356, 398, 440, 426, 484, 542, 600 };
    Tensor<qint32_t> R{C.shape()};
    R.allocateCustom(expected, 16);
    return Equals(C, R);

}
inline int testMatMulCase9()
{
    qint32_t A_data[] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 };
    qint32_t B_data[] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 };
    qint32_t C_data[9] ;

    Tensor<qint32_t> A{ 3, 5 };
A.allocateCustom(A_data, 15);
    Tensor<qint32_t> B{ 5, 3 };
B.allocateCustom(B_data, 15);
    Tensor<qint32_t> C{ 3, 3 };
C.allocateCustom(C_data, 9);
    Matmul(A, B,C);
    qint32_t expected[] = { 135, 150, 165, 310, 350, 390, 485, 550, 615 };
    Tensor<qint32_t> R{C.shape()};
    R.allocateCustom(expected, 9);
    return Equals(C, R);

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

}

#endif // EML_TEST_MATH_MATMUL_H