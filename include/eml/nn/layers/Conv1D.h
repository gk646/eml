#ifndef EML_LAYERS_CONV1D_H
#define EML_LAYERS_CONV1D_H

#include <eml/math/MathUtil.h>
#include <eml/math/TensorOps.h>
#include <eml/nn/LayerUtil.h>
#include <eml/nn/layers/ReflectionPad2D.h>
#include <eml/nn/layers/ZeroPad2D.h>

// ================================================================
// Conv1D
// ================================================================
// ................................................................
// Does not support groups or dilation
// ................................................................
// Note: The H-dimension is supposed to be 1 so channels retain their meaning (as 1D only operates on rows)
// Doc: https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
// ................................................................

namespace eml::nn
{

template <typename T>
struct Conv1D final
{

    // Creates a new Conv1D layer with the given parameters
    //      - inC:     amount of input channels
    //      - outC:    amount of output channels
    //      - kernel:  dimensions of the kernel
    //      - padding: dimensions of the kernel
    Conv1D( int32_t inC, int32_t outC, int32_t kernel, int32_t stride, int32_t padding, bool bias, PaddingMode pMode );

    // Creates a new Conv1D layer with the given parameters
    //      - inC:     amount of input channels
    //      - outC:    amount of output channels
    //      - kernel:  dimensions of the kernel
    //      - padding: dimensions of the kernel
    Conv1D( int32_t inC, int32_t outC, int32_t kernel );

    // ============ Inference ============

    // Returns an allocated tensor with shape (*, outC, 1, outW)
    Tensor<T> forward( const Tensor<T>& input );

    // Expects a correctly shaped and sufficiently allocated output tensor (*, outC, 1, outW)
    void forward( const Tensor<T>& input, Tensor<T>& output );

    // ============ Info ============

    // Returns the shape of the output tensor given a certain input tensor
    [[nodiscard]] Tuple getOutShape( const Tuple& input ) const;

    // ============ Access ============

    Tensor<T> weights;
    Tensor<T> biases;
    int32_t kernel;
    int32_t stride = 1;
    int32_t padding = 0;
    int32_t inChannels;
    int32_t outChannels;
    PaddingMode pMode;
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
Conv1D<T>::Conv1D( int32_t inC, int32_t outC, int32_t kernel, int32_t stride, int32_t padding, bool bias,
                   PaddingMode pMode )
    : weights( outC, inC, kernel ), biases( outC ), kernel( kernel ), stride( stride ), padding( padding ),
      inChannels( inC ), outChannels( outC ), pMode( pMode ), useBias( bias )
{
    EML_ASSERT( padding >= 0, "Invalid padding parameters" );
    EML_ASSERT( padding >= 0, "Invalid padding parameters" );
    EML_ASSERT( stride >= 0, "Invalid stride parameters" );
    biases.allocate();
    weights.allocate();
    if( useBias )
    {
       rand( biases, T( -1.0 ), T( 1.0 ) );
    }
    else
    {
       zero( biases );
    }
   rand( weights, T( -1.0 ), T( 1.0 ) );
}

template <typename T>
Conv1D<T>::Conv1D( int32_t inC, int32_t outC, int32_t kernel )
    : Conv1D( inC, outC, kernel, 1, 0, true, PaddingMode::ZEROS )
{
}


template <typename T>
Tensor<T> Conv1D<T>::forward( const Tensor<T>& input )
{
    EML_ASSERT( input.isAllocated() || input.isAllocatedCustom(), "Input Tensor is not allocated!" );
    Tensor<T> out{ getOutShape( input.shape() ) };
    out.allocate();
    forward( input, out );
    return out;
}

template <typename T>
void Conv1D<T>::forward( const Tensor<T>& __restrict input, Tensor<T>& __restrict output )
{
    EML_ASSERT( output.shape() == getOutShape( input.shape() ), "Output Tensor has wrong dimensions!" );
    EML_ASSERT( output.isAllocated() || output.isAllocatedCustom(), "Output Tensor is not allocated!" );
    EML_ASSERT( input.isAllocated() || input.isAllocatedCustom(), "Input Tensor is not allocated!" );
    EML_ASSERT( input.shape().second == inChannels, "Input Tensor has wrong dimensions" );

    Tensor<T> paddedInput = input;
    if( padding != 0 ) // Don't copy if there's no padding
    {
        EML_ASSERT( false, "Not implemented" );
    }

    if( useBias )
    {
        for( int32_t n = 0; n < input.n; ++n )
        {
            for( int32_t c = 0; c < outChannels; ++c )
            {
               fillDim( output, biases[ n ], n, c );
            }
        }
    }

    int32_t weightOffset = 0;
    int32_t outputOffset = 0;
    // For each output channel iterate all inputs channels
    for( int32_t out = 0; out < outChannels; ++out )
    {
        // For each input channel iterate the vector and apply the kernel - sum the channels up
        int32_t inputOffset = 0;
        for( int32_t in = 0; in < inChannels; ++in )
        {
            for( int32_t w = 0; w < output.w; ++w )
            {
                int32_t weightOffsetKernel = weightOffset;
                const int32_t kernelInputOffset = inputOffset + w * stride;
                T kSum = T( 0 );
                // Apply the kernel on the input data
                for( int32_t kw = 0; kw < kernel; ++kw )
                {
                    // Accumulate the sum of the element wise product
                    kSum += paddedInput[ kernelInputOffset + kw ] * weights[ weightOffsetKernel++ ];
                }
                output[ outputOffset + w ] += kSum;
            }
            inputOffset += paddedInput.w;
            weightOffset += weights.w; // Jump to next weight matrix row - also jumps to next channel
        }
        outputOffset += output.w;
    }
}

template <typename T>
Tuple Conv1D<T>::getOutShape( const Tuple& input ) const
{
    const int32_t simpleWidth = input.fourth - kernel + ( 2 * padding );
    const int32_t outWidth = ( simpleWidth / stride ) + 1;
    return { input.first, outChannels, 1, outWidth };
}

} // namespace eml::nn

#endif // EML_LAYERS_CONV1D_H