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
    NO_LAYER,

    // FLOAT
    CONV_1D,
    CONV_2D,
    REFLECTION_PAD_2D,
    RELU,
    LINEAR,
};

struct Layer
{
    Layer( LayerType type ) : type( type )
    {
    }
    virtual ~Layer() = default;

    virtual TensorList forward(const TensorList& inputs) = 0;
    virtual void forwardI(const TensorList& inputs, TensorList& outputs) = 0;

    virtual uint64_t getWeightCount() const = 0;
    virtual uint64_t getMemoryUsage() const = 0;

  protected:
    LayerType type;
};

#endif // EML_NN_LAYER_H