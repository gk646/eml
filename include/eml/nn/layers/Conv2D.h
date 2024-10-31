#ifndef EML_LAYERS_CONV2D_H
#define EML_LAYERS_CONV2D_H

#include <eml/math/Tensor.h>
#include <eml/math/TensorOps.h>

// ----------------------------------------------------------------
// Conv2D
// ----------------------------------------------------------------
// ................................................................
//  Does not support groups or dilation
// ................................................................
// Misc:
// TODO missing dilation
// https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
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

    int32_t getOutHeight( int32_t inH, Pair kernel, Pair padding, Pair stride );

    int32_t getOutWidth( int32_t inW, Pair kernel, Pair padding, Pair stride );

  private:
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

template <typename T>
Tensor<T> Conv2D<T>::forward( Tensor<T>& input )
{
}

template <typename T>
void Conv2D<T>::forward( Tensor<T>& input, Tensor<T>& output )
{
    for( int32_t out = 0; out < output.c; ++out )
    {
        for( int32_t in = 0; in < input.c; ++in )
        {
            for( int32_t kh = 0; kh < output.c; ++in )
            {

            }
        }
    }
}

template <typename T>
int32_t Conv2D<T>::getOutHeight( int32_t inH, Pair kernel, Pair padding, Pair stride )
{
    const int32_t normal = inH + 2 * padding.first * ( kernel.first - 1 ) - 1;
    return normal / stride.first + 1;
}

template <typename T>
int32_t Conv2D<T>::getOutWidth( int32_t inW, Pair kernel, Pair padding, Pair stride )
{
    const int32_t normal = inW + 2 * padding.second * ( kernel.second - 1 ) - 1;
    return normal / stride.second + 1;
}

} // namespace eml

#endif // EML_LAYERS_CONV2D_H