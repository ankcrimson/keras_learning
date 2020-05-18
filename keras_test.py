from keras import backend as kbe
import numpy as np
import os
os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"


# Test
data = kbe.variable(np.random.random((4, 2)))
zero_data = kbe.zeros_like(data)
print(kbe.eval(zero_data))
