import numpy as np
import pickle

#loading the saved model
loaded_model = pickle.load(open("C:/Users/Shantanu/Desktop/Machine Learning/heart_model.sav", 'rb'))

input_data=(57,1,2,128,229,0,0,150,0,0.4,1,1,3)


#change the data to a numpy array
input_data_as_numpy_array=np.asarray(input_data)

#reshape the numpy array as we are prediction for one instance
input_data_reshaped=input_data_as_numpy_array.reshape(1,-1)

prediction=loaded_model.predict(input_data_reshaped)

if(prediction[0]==0):
  print("The person does not have a heart disease")
else:
  print("The person has a heart disease")