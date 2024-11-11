import os


class HeaderGenerator:
    def __init__(self, test_per_file, tiny_guard_limit, test_macro):
        self.base_path = f"{os.getcwd()}/../tests/"
        self.max_test = test_per_file
        self.tiny_guard_limit = tiny_guard_limit
        self.tiny_guard_name = "EML_TINY_TESTBENCH"
        self.test_macro = test_macro
        os.makedirs(self.base_path, exist_ok=True)

    def create_header(self, path, file_name, content_func):
        include_guard = "EML_TEST"
        path_list = path.split('/')
        for entry in path_list:
            include_guard += f"_{entry.upper()}"
        include_guard += f"_{file_name.upper()}_H"

        file_path = self.base_path
        for entry in path_list:
            file_path = os.path.join(file_path, entry)
        os.makedirs(file_path, exist_ok=True)
        file_path = os.path.join(file_path, f"{file_name}.h")

        num_tests = 0
        content = ""
        for i in range(self.max_test):
            func_str = content_func(i)
            if func_str is None:
                print(f"No content generated for test case {i}, stopping.")
                break
            content += f"inline int test{file_name}Case{i}()\n{{"
            content += func_str
            content += "}\n"
            num_tests += 1

        if num_tests == 0:
            print("No test cases generated.")
            return

        content += f"\ninline void Test{path_list[0].upper()}{file_name}()\n{{"
        for i in range(num_tests):
            content += f"\t{self.test_macro}( test{file_name}Case{i} );\n"
            if i == self.tiny_guard_limit:
                content += f"#ifndef {self.tiny_guard_name}\n"
        if num_tests > self.tiny_guard_limit:
            content += "#endif"
        content += "\n}"

        header_content = f"""
#ifndef {include_guard}
#define {include_guard}

{content}

#endif // {include_guard}
    """
        with open(file_path, 'a+') as f:
            f.seek(0)  # Move to the beginning of the file
            f.truncate(0)  # Clear the file content
            f.write(header_content)

        print(f"Header file '{file_name}.h' created at '{file_path}'")


header_gen = HeaderGenerator(75, 25, "EML_RUN_TEST")

import test_conv2d
import test_conv1d
import test_linear
import test_matmul
import test_reflectionpad2d
import test_zeropad2d
import test_relu
import test_leakyrelu

header_gen.create_header("nn/layers", "Conv2D", test_conv2d.gen_function)
header_gen.create_header("nn/layers", "Conv1D", test_conv1d.gen_function)
header_gen.create_header("nn/layers", "Linear", test_linear.gen_function)
header_gen.create_header("math", "MatMul", test_matmul.gen_function)
header_gen.create_header("nn/layers", "ReflectionPad2D", test_reflectionpad2d.gen_function)
header_gen.create_header("nn/layers", "ZeroPad2D", test_zeropad2d.gen_function)
header_gen.create_header("nn/layers", "ReLU", test_relu.gen_function)
header_gen.create_header("nn/layers", "LeakyReLU", test_leakyrelu.gen_function)

print("Done generating")
