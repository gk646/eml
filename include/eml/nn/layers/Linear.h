#ifndef EML_LAYER_LINEAR_H
#define EML_LAYER_LINEAR_H

#include <eml/math/Tensor.h>
#include <eml/math/TensorOps.h>

namespace eml
{
    template <typename T>
    struct Linear final
    {
        Linear(int32_t in, int32_t out, bool bias = true);

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

} // namespace eml

// ----------- IMPLEMENTATION -----------

namespace eml
{
    template <typename T>
    Linear<T>::Linear(const int32_t in, const int32_t out, const bool bias) :
        inputSize(in), outputSize(out), weights(out, in), biases(out), useBias(bias)
    {
        weights.allocate();
        biases.allocate();
    }

    template <typename T>
    Tensor<T> Linear<T>::forward(Tensor<T>& input)
    {
        Tensor<T> output{input.h, weights.w};
        output.allocate();
        Matmul(input, weights, output);
        if (useBias)
            Matmul(output, biases, output);
        return output;
    }

    template <typename T>
    void Linear<T>::forward(Tensor<T>& input, Tensor<T>& output)
    {
        Matmul(input, weights, output);
        if (useBias)
            Matmul(output, biases, output);
    }
} // namespace eml

#endif //EML_LAYER_LINEAR_H