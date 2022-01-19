import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder


class Layer:
    def __init__(self, number_of_neurons, number_of_feature):
        np.random.seed(1)
        self.weight = np.random.randn(number_of_neurons, number_of_feature)
        # print(self.transposed_weight_vectors)
        # print("-----")
        np.random.seed(0)
        self.bias = np.random.randn(number_of_neurons, 1)
        self.pre_activation_output = None
        self.activation_output = None
        self.input = None


class NeuralNetwork:
    def __init__(self, number_of_feature, number_of_class, hidden_layer_cnfig, max_itr):
        self.number_of_feature = number_of_feature
        self.number_of_class = number_of_class
        self.hidden_layer_cnfig = hidden_layer_cnfig
        self.layers = []
        self.number_of_layers = None
        self.deltas = []
        self.max_iteration = max_itr

    def sigmoid(self, X):
        return 1 / (1 + np.exp(-X))

    def derivative(self, X):
        output = self.sigmoid(X)
        return output * (1 - output)

    def add_layers(self):
        layer_sizes = [self.number_of_feature] + self.hidden_layer_cnfig + [self.number_of_class]
        self.number_of_layers = len(layer_sizes) - 1
        for i in range(1, self.number_of_layers + 1):
            self.layers.append(Layer(layer_sizes[i], layer_sizes[i - 1]))

    def update_layers(self, weight, bias):
        self.add_layers()
        for i in range(self.number_of_layers):
            self.layers[i].weight = weight[i]
            self.layers[i].bias = bias[i].reshape((-1, 1))

    def load_learned_parameters(self):
        layers = []
        bias = []
        skip = 1
        for i in range(config.size):
            layer = pd.read_csv("config.txt", nrows=config[i], header=None, sep='\s{1,}', engine="python",
                                skiprows=skip, dtype=np.float64).to_numpy()
            skip += config[i]
            temp = pd.read_csv("config.txt", nrows=1, header=None, sep='\s{1,}', engine="python", skiprows=skip).to_numpy()[0]
            bias.append(temp)
            skip += 2
            layers.append(layer)
        layer = pd.read_csv("config.txt", nrows=number_of_class, header=None, sep='\s{1,}', engine="python",
                            skiprows=skip, dtype=np.float64).to_numpy()
        skip += 4
        temp = pd.read_csv("config.txt", nrows=1, header=None, sep='\s{1,}', engine="python", skiprows=skip,
                           dtype=np.float64).to_numpy()[0]
        bias.append(temp)
        layers.append(layer)
        self.update_layers(layers, bias)

    def forward(self, X):
        output = X.copy().T
        for layer in self.layers:
            layer.input = output.copy()
            layer.pre_activation_output = np.matmul(layer.weight, layer.input) + layer.bias
            layer.activation_output = self.sigmoid(layer.pre_activation_output)
            output = layer.activation_output
        return output

    def predict(self, X):
        output = self.forward(X)
        return np.argmax(output.T, axis=1) + 1

    def test(self, X, Y):
        self.load_learned_parameters()
        predicted_class = self.predict(X)
        mismatched_index = np.where(predicted_class != Y)[0]
        if len(mismatched_index) != 0:
            print("Sample No ", " Features ", "Actual class ", "Predicted Class")
        for i in mismatched_index:
            print(i+1, "  ", X[i], " ", Y[i], " ", predicted_class[i])
        accuracy = (Y.shape[0] - len(mismatched_index)) / len(X) * 100
        return accuracy


train_file = "trainNN.txt"
test_file = "testNN.txt"

train_data = pd.read_csv(train_file, sep='\s{1,}', engine="python", header=None)
test_data = pd.read_csv(test_file, sep='\s{1,}', engine="python", header=None)

feature_vector = train_data[train_data.columns[:-1]].to_numpy()
actual_class = train_data[train_data.columns[-1]].to_numpy()
number_of_features = feature_vector.shape[1]
number_of_class = np.unique(actual_class).shape[0]
encoder = OneHotEncoder(sparse=False)
actual_class = encoder.fit_transform(actual_class.reshape(-1, 1))
feature_vector = (feature_vector - feature_vector.mean(axis=0)) / feature_vector.std(axis=0)

test_feature_vector = test_data[train_data.columns[:-1]].to_numpy()
test_actual_class = test_data[train_data.columns[-1]].to_numpy()
test_feature_vector = (test_feature_vector - test_feature_vector.mean(axis=0)) / test_feature_vector.std(axis=0)

# config = pd.read_csv("config.txt", nrows=1, header=None, sep='\s{1,}', engine="python", dtype=np.float64).values[0]
config = np.loadtxt("config.txt", max_rows=1, dtype=np.int64)

neuralnet = NeuralNetwork(number_of_features, number_of_class, config.tolist(), 1000)
accuracy = neuralnet.test(test_feature_vector, test_actual_class)
print("Number of Layers: ",len(config))
print("Accuracy: ", accuracy)
print("---------------------------------------------------")