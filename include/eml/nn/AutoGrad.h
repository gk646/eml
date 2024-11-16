#ifndef EML_NN_AUTOGRAD_H
#define EML_NN_AUTOGRAD_H

#include <eml/fwd.h>

// ================================================================
// AutoGrad
// ================================================================
// ................................................................
// ................................................................

namespace eml
{

enum Operation : uint8_t
{
    MATMUL,
    MATMUL_A_TRANS,
    MATMUL_B_TRANS,
    E_ADD,
};

template <typename T>
struct AutoGradNode final
{
    Operation op; // Op this tensor was created with
    Tensor<T>& a; // input 1
    Tensor<T>& b; // input 2
    Tensor<T>& c; // output
};

} // namespace eml

#endif // EML_NN_AUTOGRAD_H