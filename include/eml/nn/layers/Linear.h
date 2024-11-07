#ifndef EML_LAYER_LINEAR_H
#define EML_LAYER_LINEAR_H

#include <eml/math/Tensor.h>
#include <eml/math/TensorOps.h>

namespace eml::nn
{
template <typename T>
struct Linear final : Layer
{
    Linear( int32_t in, int32_t out, bool bias = true );

    // Returns an allocated tensor with shape (in, out)
    Tensor<T> forward( Tensor<T>& input );

    // Expects a correctly shaped and sufficiently allocated output tensor (in, out)
    void forward( Tensor<T>& input, Tensor<T>& output );

    // ------------ Info ------------

    // see nn/Layer.h
    [[nodiscard]] int32_t getWeights() const override;

    // see nn/Layer.h
    [[nodiscard]] int32_t getMults( const Tuple& inputShape ) const override;

  private:
    int32_t inputSize;
    int32_t outputSize;
    Tensor<T> weights; // Learnable weights of shape (out, in)
    Tensor<T> biases; // Learnable bias of the layer (out)
    bool useBias = true;
};

} // namespace eml::nn

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
template <typename T>
Linear<T>::Linear( const int32_t in, const int32_t out, const bool bias )
    : inputSize( in ), outputSize( out ), weights( out, in ), biases( out ), useBias( bias )
{
    weights.allocate();
    biases.allocate();
}

template <typename T>
Tensor<T> Linear<T>::forward( Tensor<T>& input )
{
    Tensor<T> output{ input.h, weights.w };
    output.allocate();
    ops::Matmul( input, weights, output );
    if( useBias )
        ops::Matmul( output, biases, output );
    return output;
}

template <typename T>
void Linear<T>::forward( Tensor<T>& input, Tensor<T>& output )
{
    Matmul( input, weights, output );
    if( useBias )
        Matmul( output, biases, output );
}

template <typename T>
int32_t Linear<T>::getWeights() const
{
    return useBias ? weights.size + biases.size : weights.size;
}

template <typename T>
int32_t Linear<T>::getMults( const Tuple& inputShape ) const
{
    return inputShape.third * inputShape.fourth * weights.h;
}

} // namespace eml::nn

#endif // EML_LAYER_LINEAR_H