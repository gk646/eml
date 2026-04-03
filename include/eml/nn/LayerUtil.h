#ifndef EML_NN_LAYERUTIL_H
#define EML_NN_LAYERUTIL_H

#include <eml/nn/layers/Conv2D.h>
#include <eml/nn/layers/Linear.h>

// ================================================================
// LayerUtil
// ================================================================
// ................................................................
// Allows to retrieve information about layers such as amount of weights, operations needed
// ................................................................

namespace eml::nn
{

// Returns the number of learnable parameters of this layer
template <typename T>
int32_t GetLayerWeights( const Layer& layer );

// Returns the number of multiplications needed in this layer to forward the given shape a single time
template <typename T>
int32_t GetLayerOps( const Layer& layer, const Tuple& shape );

// Returns the temporary
int32_t GetLayerTempSize();

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
int32_t GetLayerWeights( const Layer& base )
{
    switch( base.type )
    {
    case LayerType::CONV_2D:
    {
        Conv2D<T>& layer = layer;
        return layer.useBias ? layer.weights.size + layer.biases.size : layer.weights.size;
    }
    case LayerType::LINEAR:
    {
        Linear<T>& layer = layer;
        return layer.useBias ? layer.weights.size + layer.biases.size : layer.weights.size;
    }
    case LayerType::REFLECTION_PAD_2D:
    case LayerType::RELU:
    case LayerType::NO_LAYER:
        break;
    }
    return 0;
}

template <typename T>
int32_t GetLayerOps( const Layer& base, const Tuple& shape )
{
    switch( base.type )
    {
    case LayerType::CONV_2D:
    {
        Conv2D<T>& layer = layer;
        // A single filter operation is output shape times kernel multiplications: (x * y) * (kx * ky) = m
        // This is done for each input channel for each output channel: m * input * output
        // This is then done for each batch
        const auto outShape = layer.getOutShape( shape );
        const int32_t perKernel = layer.kernel.first * layer.kernel.second;
        const int32_t perInputChannel = outShape.third * outShape.fourth * perKernel;
        const int32_t perOutputChannel = shape.second * perInputChannel;
        return shape.first * layer.outChannels * perOutputChannel;
    }
    case LayerType::LINEAR:
    {
        Linear<T>& layer = layer;
        return shape.third * ( 1 * shape.fourth * layer.weights.h );
    }
    case LayerType::REFLECTION_PAD_2D:
    case LayerType::RELU:
    case LayerType::NO_LAYER:
        break;
    }
    return 0;
}



} // namespace eml::nn

#endif // EML_NN_LAYERUTIL_H