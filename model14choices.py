import os
import random
import timez

import cv2

import numpy as np

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers import Conv2D, MaxPooling2D
from keras.callbacks import TensorBoard
import keras.backend.tensorflow_backend as backend

import tensorflow as tf


def check_data(choices):
    total_data = 0

    lengths = []
    for choice in choices:
        print(f"Length of {choice} is: {len(choices[choice])}")
        total_data += len(choices[choice])
        lengths.append(len(choices[choice]))

    print("Total data length now is:", total_data)
    return lengths


def get_session(gpu_fraction=0.85):
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)
    return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
backend.set_session(get_session())


def main():
    model = Sequential()
    model.add(Conv2D(32, (7, 7), padding="same", input_shape=(176, 200, 1),
                     activation="relu"))
    model.add(Conv2D(32, (3, 3), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(64, (3, 3), padding="same", activation="relu"))
    model.add(Conv2D(64, (3, 3), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(128, (3, 3), padding="same", activation="relu"))
    model.add(Conv2D(128, (3, 3), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Flatten())
    model.add(Dense(1024, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(14, activation="softmax"))

    learning_rate = 0.001
    opt = keras.optimizers.adam(lr=learning_rate)  # , decay=1e-6)

    model.compile(loss="categorical_crossentropy",
                  optimizer=opt,
                  metrics=["accuracy"])

    logdir = f"logs/STAGE2-{int(time.time())}-{learning_rate}"
    tensorboard = TensorBoard(log_dir=logdir)

    train_data_dir = "train_data"

    model = keras.models.load_model("BasicCNN-5000-epochs-0.001-LR-STAGE2")

    hm_epochs = 5000

    for i in range(hm_epochs):
        current = 0
        increment = 50
        not_maximum = True
        all_files = os.listdir(train_data_dir)
        maximum = len(all_files)
        random.shuffle(all_files)

        while not_maximum:
            try:
                print(f"WORKING ON {current}:{current + increment}, EPOCH:{i}")

                choices = {
                    0: [],
                    1: [],
                    2: [],
                    3: [],
                    4: [],
                    5: [],
                    6: [],
                    7: [],
                    8: [],
                    9: [],
                    10: [],
                    11: [],
                    12: [],
                    13: [],
                }

                for file in all_files[current:current+increment]:
                    try:
                        full_path = os.path.join(train_data_dir, file)
                        data = np.load(full_path)
                        data = list(data)
                        for d in data:
                            choice = np.argmax(d[0])
                            choices[choice].append([d[0], d[1]])
                    except Exception as e:
                        print(e)

                lengths = check_data(choices)

                lowest_data = min(lengths)

                for choice in choices:
                    random.shuffle(choices[choice])
                    choices[choice] = choices[choice][:lowest_data]

                check_data(choices)

                train_data = []

                for choice in choices:
                    for d in choices[choice]:
                        train_data.append(d)

                random.shuffle(train_data)
                print(len(train_data))

                test_size = 100
                batch_size = 128  # 128 best so far.

                x_train = np.array([i[1] for i in train_data[:-test_size]])\
                    .reshape(-1, 176, 200, 1)
                y_train = np.array([i[0] for i in train_data[:-test_size]])

                x_test = np.array([i[1] for i in train_data[-test_size:]])\
                    .reshape(-1, 176, 200, 1)
                y_test = np.array([i[0] for i in train_data[-test_size:]])

                model.fit(x_train, y_train,
                          batch_size=batch_size,
                          validation_data=(x_test, y_test),
                          shuffle=True,
                          epochs=1,
                          verbose=1, callbacks=[tensorboard])

                model.save("BasicCNN-5000-epochs-0.001-LR-STAGE2")
            except Exception as e:
                print(e)
            current += increment
            if current > maximum:
                not_maximum = False


if __name__ == "__main__":
    main()
