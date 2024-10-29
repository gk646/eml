#ifndef EML_TEST_LOGIC_GATES_H
#define EML_TEST_LOGIC_GATES_H

#include <eml/math/ActivationFunctions.h>
#include <eml/nn/NeuralNetwork.h>
#include <eml/nn/layers/Linear.h>

inline void testAndGate()
{
    struct MyNet final : NeuralNetwork
    {
        Linear<float> linear1{ 5, 5 };
        Linear<float> linear2{ 5, 1 };

        Tensor<float> forward( Tensor<float>& in )
        {
            auto out1 = linear1.forward( in );
            actf::ReLu( out1 );
            auto out2 = linear2.forward( out1 );
            actf::ReLu( out2 );
            return out2;
        }
    };

    MyNet myNet;
}

inline void TestLogicGates()
{
    testAndGate();
}

#endif // EML_TEST_LOGIC_GATES_H