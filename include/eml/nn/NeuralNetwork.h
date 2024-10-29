#ifndef EML_NEURALNETWORK_H
#define EML_NEURALNETWORK_H

#include <eml/fwd.h>

namespace eml
{
    struct NeuralNetwork
    {
        void addLayer(const Layer& layer);

        template<typename T>
        void forward(Tensor<T>& tensor);
    };

} // namespace eml

#endif //EML_NEURALNETWORK_H