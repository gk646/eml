#ifndef EML_NN_LAYERUTIL_H
#define EML_NN_LAYERUTIL_H

#include <eml/nn/layers/Conv2D.h>
#include <eml/nn/layers/Linear.h>
#include <eml/util/Macros.h>

// ================================================================
// LayerUtil
// ================================================================
// ................................................................
// Allows to retrieve information about layers such as amount of weights, operations needed
// ................................................................

namespace eml::nn
{

// Returns the number learnable parameters of this layer
template <typename Layer>
int32_t GetLayerWeights( const Layer& layer );

// Returns the number of multiplications needed in this layer to forward the given shape a single time
template <typename Layer>
int32_t GetLayerOps( const Layer& layer, const Tuple& shape );

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

EML_DEFINE_IS_LAYER_TRAIT( is_conv2d, Conv2D );
EML_DEFINE_IS_LAYER_TRAIT( is_linear, Linear );

template <typename Layer>
int32_t GetLayerWeights( const Layer& layer )
{
    if constexpr( is_conv2d<Layer>::value )
    {
        return layer.useBias ? layer.weights.size + layer.biases.size : layer.weights.size;
    }
    else if constexpr( is_linear<Layer>::value )
    {
        return layer.useBias ? layer.weights.size + layer.biases.size : layer.weights.size;
    }
    return 0;
}

template <typename Layer>
int32_t GetLayerOps( const Layer& layer, const Tuple& shape )
{

    if constexpr( is_conv2d<Layer>::value )
    {
        // A single filter operation is output shape times kernel multiplications: (x * y) * (kx * ky) = m
        // This is done for each input channel for each output channel: m * input * output
        // This is then done for each batch
        const auto outShape = layer.getOutShape( shape );
        int32_t perKernel = layer.kernel.first * layer.kernel.second;
        int32_t perInputChannel = outShape.third * outShape.fourth * perKernel;
        int32_t perOutputChannel = shape.second * perInputChannel;
        return shape.first * layer.outChannels * perOutputChannel;
    }
    else if constexpr( is_linear<Layer>::value )
    {
        return shape.third * ( 1 * shape.fourth * layer.weights.h );
    }
    return 0;
}

} // namespace eml::nn

#endif // EML_NN_LAYERUTIL_H