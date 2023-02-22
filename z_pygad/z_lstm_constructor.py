import keras
import numpy as np


class Constructor:

    def __init__(self, input_model, weights):
        self.input_model = input_model
        self.weights = weights
        self.output_model = None

        self.model_from_weights_as_vector()

    def model_from_weights_as_vector(self):
        sizes, shapes = self.extract_model_structure()
        slices = self.convert_flattened_weights(sizes, shapes)
        return self.build_neural_network(slices)

    def extract_model_structure(self):
        """
        This method extracts the original model structure, allowing
        the reconstruction of a model from a flattened weight matrix.
        """
        sizes = []
        shapes = []

        for layer in self.input_model.layers:
            if layer.trainable:

                layer_weights = layer.get_weights()
                for l_weights in layer_weights:
                    shapes.append(l_weights.shape)
                    sizes.append(l_weights.size)

        return sizes, shapes

    def convert_flattened_weights(self, sizes, shapes):
        """
        This method converts flattened weights to an array
        structure that can be used to replace neural network weights.
        """
        slices = []
        idx = 0

        for size, shape in zip(sizes, shapes):
            slice = self.weights[idx:idx + size]
            slices.append(slice.reshape(shape))
            idx += size

        return slices

    def build_neural_network(self, slices):
        """
        This method, using slices of arrays, builds a neural network
        using the weights provided in the slices iterable.
        """
        self.output_model = keras.models.Sequential()

        idx = 0

        for layer in self.input_model.layers:

            if isinstance(layer, keras.layers.Embedding):
                layer.set_weights([slices[idx]])
                self.output_model.add(layer)
                idx += 1

            elif isinstance(layer, keras.layers.LSTM):
                array = [slices[idx], slices[idx + 1], slices[idx + 2]]
                layer.set_weights(array)
                self.output_model.add(layer)
                idx += 3

            elif isinstance(layer, keras.layers.Dense):
                array = [slices[idx], slices[idx + 1]]
                layer.set_weights(array)
                self.output_model.add(layer)
                idx += 2

        assert idx == len(slices)

