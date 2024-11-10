#ifndef EML_LAYERS_REFLECTIONPAD1D_H
#define EML_LAYERS_REFLECTIONPAD1D_H

#include <eml/nn/Layer.h>


// ----------------------------------------------------------------
// ReflectionPad1D
// ----------------------------------------------------------------
// ................................................................
// ................................................................
// Doc: https://pytorch.org/docs/stable/generated/torch.nn.ReflectionPad1d.html
// ................................................................


namespace eml::nn
{

struct ReflectionPad1D final : Layer
{

    /// @brief Applies padding by using the reflection of the input on the input boundaries
    ///        E.g [1,2,3] -> [2 | 1,2,3 | 2]
    /// @param input the input tensor modified in place
    template <typename T>
    void forward( Tensor<T>& input );

    /// @brief Applies padding by using the reflection of the input on the input boundaries
    ///        E.g [1,2,3] -> [2 | 1,2,3 | 2]
    /// @param input the input tensor
    /// @return a tensor with the dimensions
    template <typename T>
    Tensor<T> forward( const Tensor<T>& input );

};


// IMPLEMENTATION
//
//
//
//
//
//
//
//
//
//
//
//
//
//
//

namespace eml::nn
{

}

} // namespace eml::nn
#endif // EML_LAYERS_REFLECTIONPAD1D_H