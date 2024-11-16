#ifndef EML_LAYER_LINEAR_H
#define EML_LAYER_LINEAR_H

#include <eml/math/Tensor.h>
#include <eml/math/TensorOps.h>

// ================================================================
// Linear
// ================================================================
// ................................................................
// Linear or dense layer with an optional bias vector
// ................................................................
// Doc: https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
// ................................................................

namespace eml::nn
{
template <typename T>
struct Linear final
{
    Linear( int32_t in, int32_t out, bool bias = true, Model* model = nullptr );

    // Returns an allocated tensor with shape (input.nch, out)
    Tensor<T> forward( Tensor<T>& input );

    // Expects a correctly shaped and sufficiently allocated output tensor (input.nch, out)
    void forwardI( Tensor<T>& input, Tensor<T>& output );

    // ============ Info ============

    Tensor<T> weights; // Learnable weights of shape (out, in)
    Tensor<T> biases;  // Learnable bias of the layer (out)
    Model* model;
    int32_t inputSize;
    int32_t outputSize;
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
Linear<T>::Linear( const int32_t in, const int32_t out, const bool bias, Model* model )
    : weights( out, in ), biases( out ), model( model ), inputSize( in ), outputSize( out ), useBias( bias )
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
    forwardI( input, output );
    return output;
}

template <typename T>
void Linear<T>::forwardI( Tensor<T>& input, Tensor<T>& output )
{
    EML_ASSERT( input.w == inputSize, "Invalid input shape" );
    EML_ASSERT( output.isAllocated() || output.isAllocatedCustom(), "Output Tensor is not allocated!" );
    EML_ASSERT( input.isAllocated() || input.isAllocatedCustom(), "Input Tensor is not allocated!" );
    EML_ASSERT( ( !input.requiresGrad ) || ( model != nullptr ), "Layer must be part of a model to enable autograd!" );

    // Multiplied as if b is transposed to match the dims (1, in), (out,in)
    matmulBTrans( input, weights, output );

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
}

} // namespace eml::nn

#endif // EML_LAYER_LINEAR_H