#ifndef EML_NN_LAYER_H
#define EML_NN_LAYER_H

#include <cstdint>

// ================================================================
// Layer
// ================================================================
// ................................................................
// Base class of all layers. A single layer of manual runtime polymorphism.
// This means there are no virtual function and no vtable.
// Type information is retained by enum.
// This allows handling layers
// ................................................................

enum class LayerType : uint8_t
{
    CONV_2D,
    REFLECTION_PAD_2D,
    RELU,
    LINEAR,
};

struct Layer
{
    LayerType type;
};

#endif // EML_NN_LAYER_H