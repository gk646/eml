#ifndef EML_NN_LAYER_H
#define EML_NN_LAYER_H

// ----------------------------------------------------------------
// nn::Layer
// ----------------------------------------------------------------
// ................................................................
// Base class of all layers to
// ................................................................
// ................................................................

namespace eml::nn
{

struct Layer
{
    virtual ~Layer() = default;

    // Returns the amount of learnable weights the layer has
    [[nodiscard]] virtual int32_t getWeights() const = 0;

    // Returns the amount of multiplications it takes to forwards this layer given an input shape
    [[nodiscard]] virtual int32_t getMults( const Tuple& shape ) const = 0;
};

} // namespace eml::nn

#endif // EML_NN_LAYER_H