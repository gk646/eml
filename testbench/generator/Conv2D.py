import torch

# Function to generate a specific test function using PyTorch
def generate_conv2d_test(test_name, input_size, in_channels, out_channels, kernel_size, stride, padding, use_bias, padding_mode):
    # Create a fixed input tensor for reproducibility
    input_height, input_width = input_size
    input_tensor = torch.arange(1, 1 + in_channels * input_height * input_width, dtype=torch.float32).reshape(1, in_channels, input_height, input_width)

    # Create a Conv2D layer in PyTorch
    conv_layer = torch.nn.Conv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        bias=use_bias,
        padding_mode=padding_mode.lower() if padding_mode != "ZEROS" else "zeros"
    )

    # Fill weights and biases with 1.0 for consistency
    torch.nn.init.constant_(conv_layer.weight, 1.0)
    if use_bias:
        torch.nn.init.constant_(conv_layer.bias, 1.0)


    # Get the output from the PyTorch Conv2D layer
    output_tensor = conv_layer(input_tensor).detach().numpy()

    # Convert input and output to flattened lists for C++
    input_data = input_tensor.flatten().tolist()
    output_data = output_tensor.flatten().tolist()

    input_data_str = ', '.join([f"{x:.1f}" for x in input_data])
    output_data_str = ', '.join([f"{x:.1f}" for x in output_data])

    # Generate the C++ code for the test
    cpp_code = f"""
inline int {test_name}()
{{
    float input[] = {{
        {input_data_str}
    }};
    Tensor<float> A{{ {in_channels}, {input_height}, {input_width} }};
    A.allocateCustom(input, {len(input_data)});
    Conv2D<float> layer{{ {in_channels}, {out_channels}, {{ {kernel_size[0]}, {kernel_size[1]} }}, {{ {stride[0]}, {stride[1]} }}, {{ {padding[0]}, {padding[1]} }}, {str(use_bias).lower()}, PaddingMode::{padding_mode} }};
    ops::Fill(layer.weights, 1.0F);
    {"ops::Fill(layer.biases, 1.0F);" if use_bias else ""}
    const auto out = layer.forward(A);
    float expected[] = {{
        {output_data_str}
    }};
    Tensor R = out.copyDims();
    R.allocateCustom(expected, {len(output_data)});
    return ops::Equals(out, R);
}}
"""
    return cpp_code

# Strategically chosen test parameters
input_sizes = [(3, 2), (2, 3), (3, 3)]  # Even and uneven input sizes
kernels = [(1, 2), (2, 1), (1, 1)]  # Square and rectangular, even and uneven
strides = [(1, 2), (2, 1),(1,1)]
channels_cases = [(1, 1), (3, 1), (1, 3)]  # Less than, same, more channels
paddings = [(0, 0), (1, 0), (0, 1)]
padding_modes = ["ZEROS", "REFLECT", "REPLICATE", "CIRCULAR"]

# Generate 75 test cases
test_cases = ""
test_count = 1

while test_count <= 75:
    input_size = input_sizes[test_count % len(input_sizes)]
    in_channels, out_channels = channels_cases[test_count % len(channels_cases)]
    kernel_size = kernels[test_count % len(kernels)]
    stride = strides[test_count % len(strides)]
    padding = paddings[test_count % len(paddings)]
    use_bias = (test_count % 2 == 0)  # Alternate bias between True and False
    padding_mode = padding_modes[test_count % len(padding_modes)]  # Rotate padding modes

    test_name = f"testConv2DCase{test_count}"
    test_case = generate_conv2d_test(
        test_name, input_size, in_channels, out_channels, kernel_size, stride, padding, use_bias, padding_mode
    )
    test_cases += test_case
    test_count += 1


# Generate the main test function
test_cases += "\ninline void TestNNLayersConv2D()\n{\n"
for i in range(1, test_count):
    test_cases += f"    EML_RUN_TEST(testConv2DCase{i});\n"
test_cases += "}\n"

print(test_cases)