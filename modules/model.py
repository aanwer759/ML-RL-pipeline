import keras
import tensorflow as tf
import modules.utilityFunctions as uf
import modules.dataLoading as dl

# model Training!
# for now , trainining using CNN model with some pretty basic configurations
# change this file into more dynamic and allowing architecture of model as well or atleast change some
# hyperparameters


# some helping functions

# global variables

trained_model = keras.models.load_model("modelCNN.h5")


def model_training(retrain=False):
    global trained_model
    _, length_directory, _ = uf.getDirectoryList()
    X,y = dl.load_data_from_directory(r'F:\\study material\\AI and ML\\LabWork\\final task\\videoFeedProcessing\\data\\train')
    if retrain:
        print("model training argument is true, therefore , training model !")
        model = keras.Sequential([
            keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=(128, 128, 1)),
            keras.layers.MaxPooling2D(pool_size=2),  # down sampling the output instead of 28*28 it is 14*14
            keras.layers.Dropout(0.2),
            keras.layers.Flatten(),  # flatten out the layers
            keras.layers.Dense(32, activation='relu'),
            keras.layers.Dense(length_directory, activation='softmax')
        ])

        model.summary()

        model.compile(loss='sparse_categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                      metrics=['accuracy'])

        callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=10)
        #y_cat = tf.keras.utils.to_categorical(y,length_directory)

        training_history = model.fit(
            X,
            y,
            epochs=10,
            verbose=1,
            callbacks=[callback],
        )
        model.save("modelCNN.h5")

    else:
        print("model training argument is false, therefore , returning pre trained model !")
        model = keras.models.load_model("modelCNN.h5")

        #trained_model = model
        #print(trained_model)

    return model


def model_accuracy(model, x_test, y_test):
    model_prediction = model.predict(x_test)
    accuracy = uf.get_accuracy(uf.get_max_value_index(model_prediction), y_test)
    return accuracy


def get_trained_model():
    return trained_model
