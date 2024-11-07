#ifndef EML_LAYERS_CONV2D_H
#define EML_LAYERS_CONV2D_H

#include <eml/math/Tensor.h>
#include <eml/math/TensorOps.h>
#include <eml/util/StackAllocator.h>

// ----------------------------------------------------------------
// Conv2D
// ----------------------------------------------------------------
// ................................................................
// Does not support groups or dilation
// ................................................................
// Doc: https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
// ................................................................

namespace eml
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

    // Returns an allocated tensor with shape (in, out)
    Tensor<T> forward( const Tensor<T>& input );

    // Expects a correctly shaped and sufficiently allocated output tensor (in, out)
    void forward( const Tensor<T>& input, Tensor<T>& output );

    // ------------ Info ------------

    [[nodiscard]] int32_t getOutHeight( int32_t inHeight ) const;

    [[nodiscard]] int32_t getOutWidth( int32_t inWidth ) const;

    // ------------ Access ------------

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

} // namespace eml

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

namespace eml
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
        ops::Random( biases, -1.0F, 1.0F );
    }
    else
    {
        ops::Zero( biases );
    }
    ops::Random(weights, -1.0F, 1.0F);
}

template <typename T>
Conv2D<T>::Conv2D( int32_t inC, int32_t outC, Pair kernel )
    : Conv2D( inC, outC, kernel, { 1, 1 }, { 0, 0 }, true, PaddingMode::ZEROS )
{
}

namespace impl
{

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
                bool inPadding = h < padding.first || h > input.h;
                // inputCopy.print();
                for( int32_t w = 0; w < inputCopy.w; ++w )
                {
                    inPadding = inPadding || w < padding.second || w > input.w;
                    if( inPadding ) [[unlikely]]
                    {
                        if constexpr( pMode == PaddingMode::ZEROS )
                        {
                            inputCopy[ padMatrixOff + h * inputCopy.w + w ] = T( 0 );
                        }
                        else if constexpr( pMode == PaddingMode::REFLECT )
                        {
                            int minusY = abs( ( padding.first - h ) );
                            if( h >= input.h + padding.first )
                                minusY = ( input.h + padding.first * 2 - 1 ) - h;
                            int minusX = abs( padding.second - w );
                            if( w >= input.w + padding.second )
                                minusX = ( input.w + padding.second * 2 - 1 ) - w;
                            inputCopy[ padMatrixOff + h * inputCopy.w + w ] =
                                input[ inMatrixOff + input.w * minusY + minusX ];
                        }
                        else if constexpr( pMode == PaddingMode::REPLICATE )
                        {
                            int inputH = h > input.h ? h - padding.first * 2 : h - padding.first;
                            int inputW = w > input.w ? w - padding.second * 2 : w - padding.second;
                            inputH = std::min( std::max( 0, inputH ), input.h );
                            inputW = std::min( std::max( 0, inputW ), input.w );
                            inputCopy[ padMatrixOff + h * inputCopy.w + w ] =
                                input[ inMatrixOff + inputH * input.w + inputW ];
                        }
                        else if constexpr( pMode == PaddingMode::CIRCULAR )
                        {
                            int inputH = abs( h - padding.first ) % input.h;
                            int inputW = abs( w - padding.second ) % input.w;
                            inputCopy[ padMatrixOff + h * inputCopy.w + w ] =
                                input[ inMatrixOff + inputH * input.w + inputW ];
                        }
                    }
                    else
                    {
                        const int inputH = h > input.h ? h - padding.first * 2 : h - padding.first;
                        const int inputW = w > input.w ? w - padding.second * 2 : w - padding.second;
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
    Tensor<T> out{ input.n, outChannels, getOutHeight( input.h ), getOutWidth( input.w ) };
    out.allocate();
    forward( input, out );
    return out;
}

template <typename T>
void Conv2D<T>::forward( const Tensor<T>& input, Tensor<T>& __restrict output )
{
    EML_ASSERT( output.h == getOutHeight( input.h ) && output.w == getOutWidth( input.w ), "Wrong output dimensions!" );

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

    paddedInput.print();

    for( int32_t i = 0; i < outChannels; i++ )
    {
        ops::FillDim( output, biases[ i ], 0, i );
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
int32_t Conv2D<T>::getOutHeight( const int32_t inHeight ) const
{
    const int32_t normal = inHeight - kernel.first + ( 2 * padding.first );
    return ( normal / stride.first ) + 1;
}

template <typename T>
int32_t Conv2D<T>::getOutWidth( const int32_t inWidth ) const
{
    const int32_t normal = inWidth - kernel.second + ( 2 * padding.second );
    return ( normal / stride.second ) + 1;
}

} // namespace eml

#endif // EML_LAYERS_CONV2D_H