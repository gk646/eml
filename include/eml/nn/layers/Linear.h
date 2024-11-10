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
    // Weights are in shape (out,in) so whe need weights.h to get out
    Tensor<T> output{ input.h, weights.h };
    output.allocate();
    // Multiplied as if b is transposed to match the dims (1, in), (out,in)
    ops::MatmulBTrans( input, weights, output );

    if( useBias )
    {
        constexpr int32_t simdSize = xsimd::batch<T>::size;
        if constexpr( simdSize >= 4 )
        {
            for( int32_t row = 0; row < output.h; ++row )
            {
                T* outputRow = &output[ +row * outputSize ];

                int32_t i = 0;
                for( ; i <= outputSize - simdSize; i += simdSize )
                {
                    xsimd::batch<T> outputBatch = xsimd::load_unaligned( outputRow + i );
                    xsimd::batch<T> biasBatch = xsimd::load_unaligned( &biases[ i ] );
                    outputBatch += biasBatch;
                    outputBatch.store_unaligned( outputRow + i );
                }

                for( ; i < outputSize; ++i )
                {
                    outputRow[ i ] += biases[ i ];
                }
            }
        }
        else
        {
            int32_t rowOff = 0;
            for( int32_t row = 0; row < output.h; ++row )
            {
                for( int32_t i = 0; i < outputSize; ++i )
                {
                    output[ rowOff + i ] = biases[ i ];
                }
                rowOff += output.w;
            }
        }
    }

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
    // Inputs for linear layer are at the simplest form a vector
    // But you can have a matrix where each vector is an input vector
    // For each input vector you do a matmul with dims (MxN): (1xinput) * (input,output)
    // For a matmul (MxN) the multiplications are: m1 * n1 * n2
    return inputShape.third * ( 1 * inputShape.fourth * weights.h );
}

} // namespace eml::nn

#endif // EML_LAYER_LINEAR_H