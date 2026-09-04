#ifndef EML_NN_AUTOGRAD_H
#define EML_NN_AUTOGRAD_H

#include <array>
#include <variant>
#include <vector>
#include "eml/nn/Optim.h"
#include "eml/util/Memory.h"
#include "eml/nn/Layer.h"

// ================================================================
// AutoGrad
// ================================================================
// ................................................................
// ................................................................

namespace eml
{

enum class Operation : uint8_t
{
    // Tensor
    TEN_MATMUL,
    TEN_MATMUL_A_TRANS,
    TEN_MATMUL_B_TRANS,
    TEN_E_ADD,

    // Activation Function
    ACTF_RELU,

    // Loss function
    LOSS_MSE,
};


struct AutoGradNode final
{
    static constexpr int8_t MAX_INPUTS = 3;
    static constexpr int8_t MAX_OUTPUTS = 2;

    Operation op; // operation this tensor was created with

    std::array<Tensor<float>*, MAX_INPUTS> inputs{};
    std::array<Tensor<float>*, MAX_OUTPUTS> outputs{};
};

struct AutoGradContext final
{
    std::pmr::vector<AutoGradNode> nodes;

    void reset()
    {
        nodes.clear();
    }

    AutoGradNode& addOp( Operation op )
    {
        return nodes.emplace_back( AutoGradNode{ op } );
    }
};

using GradIndex = uint8_t;

struct ModelContext
{
    int32_t batches = 0;
    AutoGradContext autograd;
    nn::Optimizer optim;
};

} // namespace eml

#endif // EML_NN_AUTOGRAD_H