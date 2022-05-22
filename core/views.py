import numpy as np
from django.conf import settings
from django.core.files.storage import default_storage
from django.shortcuts import render
from keras.applications import vgg16
from keras.applications.imagenet_utils import decode_predictions
from keras.utils import img_to_array, load_img
from tensorflow.python.keras.backend import set_session



def index(request):
    if request.method == 'POST':
        # file = request.FILES['imageFile']
        # file_name = default_storage.save(file.name, file)
        # file_url = default_storage.path(file_name)

        # image = load_img(file_url, target_size=(224, 224))
        # numpy_array = img_to_array(image)
        # image_batch = np.expand_dims(numpy_array, axis=0)
        # processed_image = vgg16.preprocess_input(image_batch.copy())


        # with settings.GRAPH1.as_default():
        #     set_session(settings.SESS)
        #     predictions = settings.IMAGE_MODEL.predict(processed_image)

        import tensorflow as tf
        mnist = tf.keras.datasets.mnist

        (x_train, y_train),(x_test, y_test) = mnist.load_data()
        x_train, x_test = x_train / 255.0, x_test / 255.0

        model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10, activation='softmax')
        ])

        model.compile(optimizer='adam',
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy'])

        model.fit(x_train, y_train, epochs=5)
        predictions = model.evaluate(x_test, y_test)

        # label = decode_predictions(predictions, top=10)
        return render(request, 'index.html', {'predictions' : predictions})
    else:
        return render(request, 'index.html')
    