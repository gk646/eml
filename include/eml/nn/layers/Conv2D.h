#ifndef EML_LAYERS_CONV2D_H
#define EML_LAYERS_CONV2D_H


#include <eml/math/Tensor.h>
#include <eml/math/TensorOps.h>

namespace eml {
template <typename T>
    struct Conv2D final
{
    Conv2D(int32_t in, int32_t out, bool bias = true);

    // Returns an allocated tensor with shape (in, out)
    Tensor<T> forward(Tensor<T>& input);

    // Expects a correctly shaped and sufficiently allocated output tensor (in, out)
    void forward(Tensor<T>& input, Tensor<T>& output);

private:
    int32_t inputSize;
    int32_t outputSize;
    Tensor<T> weights; // Learnable weights of shape (out, in)
    Tensor<T> biases;  // Learnable bias of the layer (out)
    bool useBias = true;
};

}

// ----------- IMPLEMENTATION -----------

namespace eml
{

}


#endif //EML_LAYERS_CONV2D_H