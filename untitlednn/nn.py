class NeuralNetwork(object):
    """Neural Network
    """

    def __init__(self, layers):
        self.layers = layers

    # @profile    # https://github.com/pythonprofilers/memory_profiler
    def forward(self, inputs):
        for layer in self.layers:
            inputs = layer.forward_with_autodiff(inputs)

        return inputs

    # @profile    # https://github.com/pythonprofilers/memory_profiler
    def backward(self, grad):
        all_grads = []
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
            all_grads.append(layer.grads)

        return all_grads[::-1]

    def get_params_and_grads(self):
        for layer in self.layers:
            yield layer.params, layer.grads

    def get_params(self):
        return [layer.params for layer in self.layers]

    def set_params(self, params):
        for i, layer in enumerate(self.layers):
            for key in layer.params.keys():
                layer.params[key] = params[i][key]
