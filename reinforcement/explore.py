import numpy as np
import torch
import torch.nn as nn


input_size = 72
hidden_layer_size = 50
output_size = 2

model = nn.Sequential(
        nn.Linear(in_features=input_size, out_features=hidden_layer_size),
        nn.ReLU(),
        nn.Linear(in_features=hidden_layer_size, out_features=output_size),
        nn.Softmax()
    )


one_hot_vector = np.zeros(input_size)
one_hot_vector[0] = 1
one_hot = torch.as_tensor(one_hot_vector, dtype=torch.float32)
#observation_tensor = torch.as_tensor(observation, dtype=torch.float32)

print(one_hot_vector.shape)
print(one_hot_vector)
output = model(one_hot)
print(output)
x = torch.argmax(output)
print(one_hot_vector[x])
print(torch.argmax(output))

