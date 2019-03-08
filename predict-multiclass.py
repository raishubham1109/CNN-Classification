import os
import numpy as np
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from keras.models import Sequential, load_model

img_width, img_height = 150, 150
model_path = './models/model.h5'
model_weights_path = './models/weights.h5'
model = load_model(model_path)
model.load_weights(model_weights_path)

def predict(file):
  x = load_img(file, target_size=(img_width,img_height))
  x = img_to_array(x)
  x = np.expand_dims(x, axis=0)
  array = model.predict(x)
  result = array[0]
  answer = np.argmax(result)
  if answer == 0:
    print("Label: Car")
  elif answer == 1:
    print("Labels: Cat")
  elif answer == 2:
    print("Label: Dog")

  return answer

car_t = 0
car_f = 0
cat_t = 0
cat_f = 0
dog_t = 0
dog_f = 0

for i, ret in enumerate(os.walk('./test-data/car')):
  for i, filename in enumerate(ret[2]):
    if filename.startswith("."):
      continue
    print("Label: Car")
    result = predict(ret[0] + '/' + filename)
    if result == 0:
      car_t += 1
    else:
      car_f += 1

for i, ret in enumerate(os.walk('./test-data/cat')):
  for i, filename in enumerate(ret[2]):
    if filename.startswith("."):
      continue
    print("Label: Cat")
    result = predict(ret[0] + '/' + filename)
    if result == 1:
      cat_t += 1
    else:
      cat_f += 1

for i, ret in enumerate(os.walk('./test-data/dog')):
  for i, filename in enumerate(ret[2]):
    if filename.startswith("."):
      continue
    print("Label: Dog")
    result = predict(ret[0] + '/' + filename)
    if result == 2:
      dog_t += 1
    else:
      dog_f += 1

"""
Check metrics
"""
print("True Car: ", car_t)
print("False Car: ", car_f)
print("True Cat: ", cat_t)
print("False Cat: ", cat_f)
print("True Dog: ", dog_t)
print("False Dog: ", dog_f)
