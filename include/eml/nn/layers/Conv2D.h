#ifndef EML_LAYERS_CONV2D_H
#define EML_LAYERS_CONV2D_H

#include <eml/math/MathUtil.h>
#include <eml/math/TensorOps.h>
#include <eml/nn/Layer.h>
#include <eml/nn/Model.h>
#include <eml/nn/layers/ReflectionPad2D.h>
#include <eml/nn/layers/ZeroPad2D.h>

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
struct Conv2D final : Layer
{
    // Creates a new Conv2D layer with the given parameters
    //      - inC:     amount of input channels
    //      - outC:    amount of output channels
    //      - kernel:  dimensions of the kernel
    //      - padding: dimensions of the kernel
    //      - model:   the model this layer is part of (if any)
    Conv2D( int32_t inC, int32_t outC, Pair kernel, Pair stride, Pair padding, bool bias, PaddingMode pMode,
            Model<T>* model = nullptr );

    // Creates a new Conv2D layer with the given parameters
    //      - inC:     amount of input channels
    //      - outC:    amount of output channels
    //      - kernel:  dimensions of the kernel
    //      - padding: dimensions of the kernel
    //      - model:   the model this layer is part of (if any)
    Conv2D( int32_t inC, int32_t outC, Pair kernel, Model<T>* model = nullptr );

    // ============ Inference ============

    // Returns an allocated tensor with shape (*, outC, outH, outW)
    [[nodiscard]] Tensor<T> forward( const Tensor<T>& input );

    // Expects a correctly shaped and sufficiently allocated output tensor with shape: (*, outC, outH, outW)
    void forwardI( const Tensor<T>& input, Tensor<T>& output );

    // ============ Info ============

    // Returns the shape of the output tensor given a certain input tensor
    [[nodiscard]] Tuple getOutShape( const Tuple& input ) const;

    // ============ Variables ============

    Tensor<T> weights;       // Learnable weights of shape (kernelX, kernelY)
    Tensor<T> biases;        // Learnable bias of the layer (out)
    Pair kernel;             // (height, width) of the kernel
    Pair stride = { 1, 1 };  // (height, width) of the stride
    Pair padding = { 0, 0 }; // (height, width) of the padding
    Model<T>* model;         // the model this layer is part of
    int32_t inChannels;      // number of channels
    int32_t outChannels;     // number of channels
    PaddingMode pMode;       // Which padding mode
    bool useBias = true;     // use a bias vector per out channel
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
Conv2D<T>::Conv2D( const int32_t inC, const int32_t outC, const Pair kernel, const Pair stride, const Pair padding,
                   const bool bias, const PaddingMode pMode, Model<T>* model )
    : Layer(), weights( outC, inC, kernel.first, kernel.second ), biases( outC ), kernel( kernel ), stride( stride ),
      padding( padding ), model( model ), inChannels( inC ), outChannels( outC ), pMode( pMode ), useBias( bias )
{
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
Conv2D<T>::Conv2D( int32_t inC, int32_t outC, Pair kernel, Model<T>* model )
    : Conv2D( inC, outC, kernel, { 1, 1 }, { 0, 0 }, true, PaddingMode::ZEROS, model )
{
}

namespace impl
{

template <typename T, PaddingMode pMode>
Tensor<T> Conv2DApplyPadding( const Tensor<T>& input, const Pair padding )
{
    EML_ASSERT( padding.first != padding.second || padding.first != 0, "No padding case is filtered" );

    if( pMode == PaddingMode::REFLECT )
    {

        ReflectionPad2D pad{ padding };
        return pad.forward( input );
    }
    if( pMode == PaddingMode::ZEROS )
    {
        ZeroPad2D pad{ padding };
        return pad.forward( input );
    }

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
                        if constexpr( pMode == PaddingMode::REPLICATE )
                        {
                            int inputH = clampt( h - padding.first, 0, input.h - 1 );
                            int inputW = clampt( w - padding.second, 0, input.w - 1 );

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
Tensor<T> Conv2D<T>::forward( const Tensor<T>& restrict input )
{
    EML_ASSERT( input.isAllocated() || input.isAllocatedCustom(), "Input Tensor is not allocated!" );
    Tensor<T> out{ getOutShape( input.shape() ) };
    out.allocate();
    forwardI( input, out );
    return out;
}

template <typename T>
void Conv2D<T>::forwardI( const Tensor<T>& restrict input, Tensor<T>& restrict output )
{
    EML_ASSERT( output.shape() == getOutShape( input.shape() ), "Output Tensor has wrong dimensions!" );
    EML_ASSERT( output.isAllocated() || output.isAllocatedCustom(), "Output Tensor is not allocated!" );
    EML_ASSERT( input.isAllocated() || input.isAllocatedCustom(), "Input Tensor is not allocated!" );
    EML_ASSERT( input.shape().second == inChannels, "Input Tensor has wrong dimensions" );
    EML_ASSERT( ( !input.requiresGrad ) || ( model != nullptr ), "Layer must be part of a model to enable autograd!" );

    Tensor<T> paddedInput = input;
    if( padding.first != 0 || padding.second != 0 ) // Don't copy if there's no padding
    {
        switch( pMode )
        {
        case PaddingMode::ZEROS:
        {
            ZeroPad2D pad{ padding };
            paddedInput = pad.forward( input );
        }

        break;
        case PaddingMode::REFLECT:
        {
            ReflectionPad2D pad{ padding };
            paddedInput = pad.forward( input );
        }
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
            fillDim( output, biases[ i ], 0, i );
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