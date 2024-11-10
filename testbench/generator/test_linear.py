import torch

import gen_utils


def impl_gen_linear(input_size, output_size, use_bias):
    input_tensor = torch.arange(1, 1 + input_size, dtype=torch.float32)

    linear_layer = torch.nn.Linear(input_size, output_size, bias=use_bias)

    torch.nn.init.constant_(linear_layer.weight, 1.0)
    if use_bias:
        torch.nn.init.constant_(linear_layer.bias, 1.0)

    output_tensor = linear_layer(input_tensor).detach().numpy()
    output_data = output_tensor.flatten().tolist()

    input_tensor_decl = gen_utils.generate_decl_tensor_range("A", (1, input_size), input_size)

    linear_layer_decl = f"Linear<qint32_t> layer{{ {input_size}, {output_size}, {str(use_bias).lower()} }};"

    output_tensor_decl = gen_utils.generate_tensor_data("R", output_tensor.shape, output_data)

    cpp_code = f"""
    {input_tensor_decl}
    {linear_layer_decl}
    Fill(layer.weights, 1);
    {"Fill(layer.biases, 1);" if use_bias else ""}
    const auto out = layer.forward(A);
    
    {output_tensor_decl}
    return Equals(out, R);
"""
    return cpp_code


input_sizes = [4, 8, 16]
output_sizes = [2, 8, 20]
bias_options = [True, False]


def gen_function(index):
    input_size = input_sizes[index % len(input_sizes)]
    output_size = output_sizes[index % len(output_sizes)]
    use_bias = bias_options[index % len(bias_options)]
    if index > 6:
        return None
    return impl_gen_linear(input_size, output_size, use_bias)
