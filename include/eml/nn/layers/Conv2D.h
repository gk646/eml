#ifndef EML_LAYERS_CONV2D_H
#define EML_LAYERS_CONV2D_H

#include <eml/math/Tensor.h>
#include <eml/math/TensorOps.h>

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
    Tensor<T> forward( Tensor<T>& input );

    // Expects a correctly shaped and sufficiently allocated output tensor (in, out)
    void forward( Tensor<T>& input, Tensor<T>& output );

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
}

template <typename T>
Conv2D<T>::Conv2D( int32_t inC, int32_t outC, Pair kernel )
    : Conv2D( inC, outC, kernel, { 1, 1 }, { 0, 0 }, true, PaddingMode::ZEROS )
{
}

namespace impl
{

}

template <typename T>
Tensor<T> Conv2D<T>::forward( Tensor<T>& input )
{
    Tensor<T> out{ input.n, outChannels, getOutHeight( input.h ), getOutWidth( input.w ) };
    out.allocate();
    forward( input, out );
    return out;
}

template <typename T>
void Conv2D<T>::forward( Tensor<T>& input, Tensor<T>& output )
{
    for( int32_t i = 0; i < outChannels; i++ )
    {
        ops::FillDim( output, biases[ i ], 1, i );
    }
    int32_t weightOffset = 0;
    int32_t outputOffset = 0;
    // For each output channel iterate all inputs channels
    for( int32_t out = 0; out < output.c; ++out )
    {
        // For each input channel iterate the matrix and apply the kernel - sum the channels up
        for( int32_t in = 0; in < input.c; ++in )
        {
            int32_t inputOffset = in * input.hw;
            int32_t outputChannelOffset = outputOffset; // Start new each channel
            for( int32_t y = 0; y < output.h; ++y )
            {
                for( int32_t x = 0; x < output.w; ++x )
                {
                    int32_t weightOffsetKernel = weightOffset; // Cache the current offset
                    int32_t kernelInputOffset = inputOffset + x * stride.second;
                    T kSum = T( 0 );

                    // Apply the kernel on the input data
                    for( int32_t kh = 0; kh < kernel.first; ++kh )
                    {
                        for( int32_t kw = 0; kw < kernel.second; ++kw )
                        {
                            // Accumulate the sum of the element wise product
                            kSum += input[ kernelInputOffset + kw ] * weights[ weightOffsetKernel++ ];
                        }
                        kernelInputOffset += input.w;
                    }
                    output[ outputChannelOffset + x ] += kSum;
                }
                outputChannelOffset += output.w;
                inputOffset += input.w;
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