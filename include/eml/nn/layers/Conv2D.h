#ifndef EML_LAYERS_CONV2D_H
#define EML_LAYERS_CONV2D_H

#include <eml/math/TensorOps.h>
#include <eml/nn/LayerUtil.h>
#include <eml/nn/layers/ReflectionPad2D.h>
#include <eml/nn/layers/ZeroPad2D.h>
#include <eml/math/MathUtil.h>

// ================================================================
// Conv2D
// ================================================================
// ................................................................
// Does not support groups or dilation
// ................................................................
// Doc: https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
// ................................................................

namespace eml::nn
{

template <typename T>
struct Conv2D final
{

    // Creates a new Conv2D layer with the given parameters
    //      - inC:     amount of input channels
    //      - outC:    amount of output channels
    //      - kernel:  dimensions of the kernel
    //      - padding: dimensions of the kernel
    Conv2D( int32_t inC, int32_t outC, Pair kernel, Pair stride, Pair padding, bool bias, PaddingMode pMode );

    // Creates a new Conv2D layer with the given parameters
    //      - inC:     amount of input channels
    //      - outC:    amount of output channels
    //      - kernel:  dimensions of the kernel
    //      - padding: dimensions of the kernel
    Conv2D( int32_t inC, int32_t outC, Pair kernel );

    // ============ Inference ============

    // Returns an allocated tensor with shape (*, outC, outH, outW)
    Tensor<T> forward( const Tensor<T>& input );

    // Expects a correctly shaped and sufficiently allocated output tensor with shape: (*, outC, outH, outW)
    void forward( const Tensor<T>& input, Tensor<T>& output );

    // ============ Info ============

    // Returns the shape of the output tensor given a certain input tensor
    [[nodiscard]] Tuple getOutShape( const Tuple& input ) const;

    // ============ Access ============

    Tensor<T> weights; // Learnable weights of shape (kernelX, kernelY)
    Tensor<T> biases; // Learnable bias of the layer (out)
    Pair kernel;
    Pair stride = { 1, 1 };
    Pair padding = { 0, 0 };
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
Conv2D<T>::Conv2D( int32_t inC, int32_t outC, Pair kernel, Pair stride, Pair padding, const bool bias,
                   PaddingMode pMode )
    : weights( outC, inC, kernel.first, kernel.second ), biases( outC ), kernel( kernel ), stride( stride ),
      padding( padding ), inChannels( inC ), outChannels( outC ), pMode( pMode ), useBias( bias )
{
    biases.allocate();
    weights.allocate();
    if( useBias )
    {
        ops::Random( biases, T( -1.0 ), T( 1.0 ) );
    }
    else
    {
        ops::Zero( biases );
    }
    ops::Random( weights, T( -1.0 ), T( 1.0 ) );
}

template <typename T>
Conv2D<T>::Conv2D( int32_t inC, int32_t outC, Pair kernel )
    : Conv2D( inC, outC, kernel, { 1, 1 }, { 0, 0 }, true, PaddingMode::ZEROS )
{
}

namespace impl
{

int reflectIndex( int index, int size )
{
    if( size == 1 )
        return 0; // The only valid index is 0

    int period = 2 * size - 2;
    int i_mod = index % period;
    if( i_mod < 0 )
        i_mod += period; // Ensure positive modulus

    if( i_mod >= size )
        return period - i_mod;
    else
        return i_mod;
}

template <typename T, PaddingMode pMode>
Tensor<T> Conv2DApplyPadding( const Tensor<T>& input, const Pair padding )
{
    EML_ASSERT( padding.first != padding.second || padding.first != 0, "No padding case is filtered" );
    if constexpr( pMode == PaddingMode::REFLECT )
        EML_ASSERT( input.w > padding.second, "Cannot use reflect padding when input is smaller than padding!" );

    Tensor<T> inputCopy{ input.n, input.c, input.h + 2 * padding.first, input.w + 2 * padding.second };
    inputCopy.allocate();

    int32_t padOffset = 0;
    int32_t inOffset = 0;
    for( int32_t n = 0; n < inputCopy.n; ++n )
    {
        int32_t padChannelOff = padOffset;
        int32_t inChannelOff = inOffset;
        for( int32_t c = 0; c < inputCopy.c; ++c )
        {
            int32_t padMatrixOff = padChannelOff;
            int32_t inMatrixOff = inChannelOff;
            for( int32_t h = 0; h < inputCopy.h; ++h )
            {
                const bool inPaddingH = h < padding.first || h >= padding.first + input.h;
                for( int32_t w = 0; w < inputCopy.w; ++w )
                {
                    const bool inPaddingW = w < padding.second || w >= padding.second + input.w;
                    if( inPaddingH || inPaddingW ) [[unlikely]]
                    {
                        if constexpr( pMode == PaddingMode::ZEROS )
                        {
                            inputCopy[ padMatrixOff + h * inputCopy.w + w ] = T( 0 );
                        }
                        else if constexpr( pMode == PaddingMode::REFLECT )
                        {
                            int y = reflectIndex( h - padding.first, input.h );
                            int x = reflectIndex( w - padding.second, input.w );

                            inputCopy[ padMatrixOff + h * inputCopy.w + w ] = input[ inMatrixOff + y * input.w + x ];
                        }
                        else if constexpr( pMode == PaddingMode::REPLICATE )
                        {
                            int inputH = Clamp( h - padding.first, 0, input.h - 1 );
                            int inputW = Clamp( w - padding.second, 0, input.w - 1 );

                            inputCopy[ padMatrixOff + h * inputCopy.w + w ] =
                                input[ inMatrixOff + inputH * input.w + inputW ];
                        }
                        else if constexpr( pMode == PaddingMode::CIRCULAR )
                        {
                            int inputW = w - padding.second;
                            if( inputW < 0 )
                                inputW += input.w;
                            inputW %= input.w;

                            int inputH = h - padding.first;
                            if( inputH < 0 )
                                inputH += input.h;
                            inputH %= input.h;

                            inputCopy[ padMatrixOff + h * inputCopy.w + w ] =
                                input[ inMatrixOff + inputH * input.w + inputW ];
                        }
                    }
                    else
                    {
                        const int inputH = h - padding.first;
                        const int inputW = w - padding.second;
                        inputCopy[ padMatrixOff + h * inputCopy.w + w ] =
                            input[ inMatrixOff + inputH * input.w + inputW ];
                    }
                }
            }
            padChannelOff += inputCopy.hw;
            inChannelOff += input.hw;
        }
        padOffset += inputCopy.chw;
        inOffset += input.chw;
    }
    return inputCopy;
}

} // namespace impl

template <typename T>
Tensor<T> Conv2D<T>::forward( const Tensor<T>& input )
{
    EML_ASSERT( input.isAllocated() || input.isAllocatedCustom(), "Input Tensor is not allocated!" );
    Tensor<T> out{ getOutShape( input.shape() ) };
    out.allocate();
    forward( input, out );
    return out;
}

template <typename T>
void Conv2D<T>::forward( const Tensor<T>& __restrict input, Tensor<T>& __restrict output )
{
    EML_ASSERT( output.shape() == getOutShape( input.shape() ), "Output Tensor has wrong dimensions!" );
    EML_ASSERT( output.isAllocated() || output.isAllocatedCustom(), "Output Tensor is not allocated!" );
    EML_ASSERT( input.isAllocated() || input.isAllocatedCustom(), "Input Tensor is not allocated!" );
    EML_ASSERT( input.shape().second == inChannels, "Input Tensor has wrong dimensions" );

    Tensor<T> paddedInput = input;
    if( padding.first != 0 || padding.second != 0 ) // Don't copy if there's no padding
    {
        switch( pMode )
        {
        case PaddingMode::ZEROS:
            paddedInput = impl::Conv2DApplyPadding<T, PaddingMode::ZEROS>( input, padding );
            break;
        case PaddingMode::REFLECT:
            paddedInput = impl::Conv2DApplyPadding<T, PaddingMode::REFLECT>( input, padding );
            break;
        case PaddingMode::REPLICATE:
            paddedInput = impl::Conv2DApplyPadding<T, PaddingMode::REPLICATE>( input, padding );
            break;
        case PaddingMode::CIRCULAR:
            paddedInput = impl::Conv2DApplyPadding<T, PaddingMode::CIRCULAR>( input, padding );
            break;
        }
    }

    if( useBias )
    {
        for( int32_t i = 0; i < outChannels; i++ )
        {
            ops::FillDim( output, biases[ i ], 0, i );
        }
    }

    int32_t weightOffset = 0;
    int32_t outputOffset = 0;
    // For each output channel iterate all inputs channels
    for( int32_t out = 0; out < output.c; ++out )
    {
        // For each input channel iterate the matrix and apply the kernel - sum the channels up
        for( int32_t in = 0; in < paddedInput.c; ++in )
        {
            int32_t inputOffset = in * paddedInput.hw;
            int32_t outputChannelOffset = outputOffset; // Start new each channel
            for( int32_t y = 0; y < output.h; ++y )
            {
                for( int32_t x = 0; x < output.w; ++x )
                {
                    int32_t weightOffsetKernel = weightOffset;
                    int32_t kernelInputOffset = inputOffset + x * stride.second;

                    T kSum = T( 0 );
                    // Apply the kernel on the input data
                    for( int32_t kh = 0; kh < kernel.first; ++kh )
                    {
                        for( int32_t kw = 0; kw < kernel.second; ++kw )
                        {
                            // Accumulate the sum of the element wise product
                            kSum += paddedInput[ kernelInputOffset + kw ] * weights[ weightOffsetKernel++ ];
                        }
                        kernelInputOffset += paddedInput.w;
                    }
                    output[ outputChannelOffset + x ] += kSum;
                }
                outputChannelOffset += output.w;
                inputOffset += stride.first * paddedInput.w;
            }
            weightOffset += weights.hw; // Jump to next weight matrix channel - also jumps to next batch
        }
        outputOffset += output.hw;
    }
}

template <typename T>
Tuple Conv2D<T>::getOutShape( const Tuple& input ) const
{
    const int32_t simpleHeight = input.third - kernel.first + ( 2 * padding.first );
    const int32_t outHeight = ( simpleHeight / stride.first ) + 1;

    const int32_t simpleWidth = input.fourth - kernel.second + ( 2 * padding.second );
    const int32_t outWidth = ( simpleWidth / stride.second ) + 1;

    return { input.first, outChannels, outHeight, outWidth };
}

} // namespace eml::nn

#endif // EML_LAYERS_CONV2D_H