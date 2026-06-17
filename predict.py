import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

model = tf.keras.models.load_model("cat_dog_model.h5")

img_path = "dog.jpg"     # image name

img = image.load_img(
    img_path,
    target_size=(150, 150)
)

img_array = image.img_to_array(img)

img_array = np.expand_dims(
    img_array,
    axis=0
)

img_array = img_array / 255.0

prediction = model.predict(img_array)

print("Raw Prediction =", prediction[0][0])

if prediction[0][0] > 0.5:
    print("DOG")
else:
    print("CAT")