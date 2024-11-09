import torch

import gen_utils

def impl_gen_linear(input_size, output_size, use_bias):
    # Create a fixed input tensor for reproducibility
    input_tensor = torch.arange(1, 1+ input_size , dtype=torch.float32)

    # Create a Linear layer in PyTorch
    linear_layer = torch.nn.Linear(input_size, output_size, bias=use_bias)

    # Initialize weights and biases
    torch.nn.init.constant_(linear_layer.weight, 1.0)
    if use_bias:
        torch.nn.init.constant_(linear_layer.bias, 1.0)

    # Get the output tensor
    output_tensor = linear_layer(input_tensor).detach().numpy()

    # Flatten input and output for C++ representation
    input_data = input_tensor.flatten().tolist()
    output_data = output_tensor.flatten().tolist()

    # Generate C++ code for the test function
    input_array = gen_utils.generate_stack_array_data("input", input_data)
    output_array = gen_utils.generate_stack_array_data("expected", output_data)
    input_tensor_decl = gen_utils.generate_tensor("A","input", (1, input_size), input_size)
    linear_layer_decl = f"Linear<qint32_t> layer{{ {input_size}, {output_size}, {str(use_bias).lower()} }};"

    cpp_code = f"""
    {input_array}
    {input_tensor_decl}
    {linear_layer_decl}
    ops::Fill(layer.weights, 1);
    {"ops::Fill(layer.biases, 1);" if use_bias else ""}
    const auto out = layer.forward(A);
    {output_array}
    Tensor<qint32_t> R{{out.shape()}};
    R.allocateCustom(expected, {len(output_data)});
    return ops::Equals(out, R);
"""
    return cpp_code

input_sizes = [4, 8, 16]
output_sizes = [2,8,20]
bias_options = [True, False]

def linear_gen_function(index):
    input_size = input_sizes[index  % len(input_sizes)]
    output_size = output_sizes[index  % len(output_sizes)]
    use_bias = bias_options[index % len(bias_options)]
    if index > 6:
        return None
    return impl_gen_linear(input_size,  output_size, use_bias)
