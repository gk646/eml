Attributes:

- Embedded Focused
- No global state
    - For testability and modularity
- more than just NN
    - also Regression, Trees...
- Strong compile options
- Support model imports (pytorch, tensorflow)
- Models are defined at runtime
    - Types for activation and weights are templates (quantization)
- Support quantization
    - Static quantization for weights and activations

Style Guide:

- Similar
  to [FreeRTOS](https://www.freertos.org/Documentation/02-Kernel/06-Coding-guidelines/02-FreeRTOS-Coding-Standard-and-Style-Guide)
- MISRA compliant
    - Use Coverity
    - Heady only
        - necessary due to templates
        - But still in header source fashion /implementation is at the bottom
        - Simpler to build on any platform

## Running tests

`pip install gcovr`

`mkdir build`

`cd build`

`cmake ..`

`cmake . coverage `


tinytorch

minimize binary size
- conv2d forward 10kib


- linear regression
- logical inference engine
  - knowledge base
- bayes net