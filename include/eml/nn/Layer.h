#ifndef EML_NN_LAYER_H
#define EML_NN_LAYER_H

#include <eml/math/MathUtil.h>
#include <eml/math/Tensor.h>

// ----------------------------------------------------------------
// nn::Layer
// ----------------------------------------------------------------
// ................................................................
// Base class of all layers
// ................................................................

namespace eml::nn
{

struct Layer
{
    virtual ~Layer() = default;

    // Returns the amount of learnable weights the layer has
    [[nodiscard]] virtual int32_t getWeights() const;

    // Returns the amount of multiplications it takes to forwards this layer given an input shape
    [[nodiscard]] virtual int32_t getMults( const Tuple& shape ) const;
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

inline int32_t Layer::getWeights() const
{
    return 0;
}
inline int32_t Layer::getMults( const Tuple& shape ) const
{
    return 0;
}

} // namespace eml::nn

#endif // EML_NN_LAYER_H