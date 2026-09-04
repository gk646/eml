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
    Linear( int32_t in, int32_t out, bool bias = true, Model<T>* model = nullptr );

    // Returns an allocated tensor with shape (*,*,*, out)
    Tensor<T> forward( const Tensor<T>& input );

    // Expects a correctly shaped and sufficiently allocated output tensor (*,*,*, out)
    void forwardI( const Tensor<T>& input, Tensor<T>& output );

    void backwardI( AutoGradNode& node, const Tensor<T>& delta );

    // ============ Info ============

    static void forwardImpl( const Tensor<T>& restrict input, Tensor<T>& restrict output,
                             const Tensor<T>& restrict weights, const Tensor<T>& restrict bias, bool useBias );

    Tensor<T> weights; // Learnable weights of shape (out, in)
    Tensor<T> biases;  // Learnable bias of the layer (out)
    Model<T>* model = nullptr;

    int32_t inputSize;
    int32_t outputSize;
    bool useBias = true;
};

} // namespace eml::nn

// IMPLEMENTATION

namespace eml::nn
{

template <typename T>
Linear<T>::Linear( const int32_t in, const int32_t out, const bool bias, Model<T>* model )
    : weights( out, in ), biases( out ), model( model ), inputSize( in ), outputSize( out ), useBias( bias )
{
    if( model ) // Part of a model
    {
        EML_MODEL_ALLOCATE( weights );
        if( bias )
        {
            EML_MODEL_ALLOCATE( biases );
        }
    }
    else
    {
        weights.allocate();
        biases.allocate();
    }
}

template <typename T>
Tensor<T> Linear<T>::forward( const Tensor<T>& input )
{
    EML_ASSERT( input.w == inputSize, "Invalid input shape" );
    // Weights are in shape (out,in) so whe need weights.h to get out
    Tensor<T> output{ input.n, input.c, input.h, weights.h };
    output.allocate();

    forwardI( input, output );
    return output;
}

template <typename T>
void Linear<T>::forwardI( const Tensor<T>& input, Tensor<T>& output )
{
    EML_ASSERT( input.w == inputSize, "Invalid input shape" );
    EML_ASSERT( output.isAllocated() || output.isAllocatedCustom(), "Output Tensor is not allocated!" );
    EML_ASSERT( input.isAllocated() || input.isAllocatedCustom(), "Input Tensor is not allocated!" );
    EML_ASSERT( ( !input.requiresGrad ) || ( model != nullptr ), "Layer must be part of a model to enable autograd!" );

    forwardImpl( input, output, weights, biases, useBias );
}

template <typename T>
void Linear<T>::backwardI( AutoGradNode& node, const Tensor<T>& delta )
{
    // Gradient for weights: delta * input^T
    // Gradient for bias: delta
    // Gradient for input: weights^T * delta

    // // Update weight gradients
    // for( size_t i = 0; i < saved_weights.data.size(); ++i )
    // {
    //     saved_weights.grad.data[ i ] += output_grad.grad.data[ i % output_grad.grad.data.size() ] *
    //                                     saved_input.data[ i / output_grad.grad.data.size() ];
    // }
    //
    // // Accumulate bias gradients
    // for( size_t i = 0; i < bias->grad.data.size(); ++i )
    // {
    //     bias->grad.data[ i ] += output_grad.grad.data[ i ];
    // }
    //
    // // Compute input gradient: weights^T * output_grad
    // for( size_t i = 0; i < input->grad.data.size(); ++i )
    // {
    //     input->grad.data[ i ] += 0.0f; // Initialize
    //     for( size_t j = 0; j < weights->data.size() / input->grad.data.size(); ++j )
    //     {
    //         input->grad.data[ i ] += weights->data[ j * input->grad.data.size() + i ] * output_grad.grad.data[ j ];
    //     }
    // }
}

template <typename T>
void Linear<T>::forwardImpl( const Tensor<T>& input, Tensor<T>& output, const Tensor<T>& weights,
                             const Tensor<T>& biases, bool useBias )
{
    EML_ASSERT( output.isAllocated() || output.isAllocatedCustom(), "Output Tensor is not allocated!" );
    EML_ASSERT( input.isAllocated() || input.isAllocatedCustom(), "Input Tensor is not allocated!" );

    // Multiplied as if b is transposed to match the dims (1, in), (out,in)
    matmulBTrans( input, weights, output );

    const auto outputSize = weights.h;
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