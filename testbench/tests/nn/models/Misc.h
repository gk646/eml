#ifndef EML_TESTS_MODELS_HANDMADE_H
#define EML_TESTS_MODELS_HANDMADE_H

using namespace eml;



inline void TestModelsCustom()
{
    // Step 1: Initialize input
    Tensor<float> input{ 4 };
    input.allocate();
    arange( input, 1.0F ); // input = [1, 2, 3, 4]

    // Step 2: Initialize weights
    Tensor<float> weights{ 1, 4 };
    weights.allocate();
    rand( weights, -1.0F, 1.0F ); // weights = random values between -1 and 1

    // Step 3: Forward pass - matrix multiplication
    Tensor<float> output{ 1 };
    output.allocate();
    matmulBTrans( input, weights, output ); // output = z

    // Step 4: Forward pass - ReLU activation
    Tensor<float> reluOut = ReLU::forward( output ); // reluOut = ReLU(z)

    // Step 5: Compute loss
    Tensor<float> target{ 1 };
    float targetData[] = { 1.0F };
    target.allocateCustom( targetData, 1 );
    float loss = powf( reluOut[ 0 ] - target[ 0 ], 2 );

    // Step 6: Backward pass - compute gradients
    // Compute derivative of loss with respect to reluOut
    float d_loss_d_reluOut = 2.0F * ( reluOut[ 0 ] - target[ 0 ] );

    // Compute derivative of reluOut with respect to z
    float d_reluOut_d_z = ( output[ 0 ] > 0.0F ) ? 1.0F : 0.0F;

    // Compute derivative of loss with respect to z
    float d_loss_d_z = d_loss_d_reluOut * d_reluOut_d_z;

    // Compute derivative of z with respect to input
    Tensor<float> d_loss_d_input{ 4 };
    d_loss_d_input.allocate();

    for( int i = 0; i < 4; ++i )
    {
        float d_z_d_input_i = weights[ i ]; // ∂z/∂input_i = weights_i
        float d_loss_d_input_i = d_loss_d_z * d_z_d_input_i;
        d_loss_d_input[ i ] = d_loss_d_input_i;
    }

    // Now d_loss_d_input contains ∂loss/∂input_i for i = 1 to 4
    // You can print or use these gradients as needed
}

inline void TestNNCustom()
{
    TestModelsCustom();
}

#endif // EML_TESTS_MODELS_HANDMADE_H