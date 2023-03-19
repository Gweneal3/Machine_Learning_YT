import numpy as np
import pickle  # pip install pickle-mixin

loaded_model = pickle.load(open("./trained_model.sav", "rb"))

input_data = (1, 89, 66, 23, 94, 28.1, 0.167, 21)

input_data_as_numpy_array = np.asarray(input_data)

input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

# standardised input data
# std_data = scaler.transform(input_data_reshaped);

prediction = loaded_model.predict(input_data_reshaped)

if prediction[0] == 1:
    print("This person is diabetic")
else:
    print("This person is not diabetic")
