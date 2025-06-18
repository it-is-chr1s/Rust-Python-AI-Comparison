# https://www.kaggle.com/code/muhammadimran112233/99-acc-fruits-recognition-using-nn

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Activation, Dense, Flatten, Dropout
from tensorflow.keras import backend as K, Input
from tensorflow.keras.preprocessing.image import array_to_img, img_to_array, load_img
from sklearn.datasets import load_files
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
from TimeMeasuring import TimeMeasuring
from datetime import datetime
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

def load_data(data_path):
    data_loading = load_files(data_path)
    files_add = np.array(data_loading['filenames'])
    targets_fruits = np.array(data_loading['target'])
    target_labels_fruits = np.array(data_loading['target_names'])
    return files_add, targets_fruits, target_labels_fruits

def balance_y_train_and_test(x_train, y_train, x_test, y_test, target_labels_train, target_labels_test):
    label_names_train = np.array([target_labels_train[i] for i in y_train])
    label_names_test = np.array([target_labels_test[i] for i in y_test])

    common_labels = np.intersect1d(label_names_train, label_names_test)

    train_mask = np.isin(label_names_train, common_labels)
    test_mask = np.isin(label_names_test, common_labels)

    x_train_bal = x_train[train_mask]
    y_train_bal_labels = label_names_train[train_mask]
    x_test_bal = x_test[test_mask]
    y_test_bal_labels = label_names_test[test_mask]

    label_to_new_index = {label: idx for idx, label in enumerate(sorted(common_labels))}
    y_train_bal = np.array([label_to_new_index[label] for label in y_train_bal_labels])
    y_test_bal = np.array([label_to_new_index[label] for label in y_test_bal_labels])

    return x_train_bal, y_train_bal, x_test_bal, y_test_bal, sorted(common_labels)

def to_one_hot_encoding(y):
     no_of_classes = len(np.unique(y))
     return to_categorical(y, no_of_classes)

def divide_set(set, middle):
    return set[:middle],set[middle:]

def data_preprocessing(x_train, x_valid, x_test):
    def convert_images_to_array_form(files):
        images_array = []
        for file in files:
            images_array.append(img_to_array(load_img(file)))
        return images_array
    
    x_train = np.array(convert_images_to_array_form(x_train))
    x_valid = np.array(convert_images_to_array_form(x_valid))
    x_test = np.array(convert_images_to_array_form(x_test))

    print('Training set shape : ',x_train.shape)
    print('Validation set shape : ',x_valid.shape)
    print('Test set shape : ',x_test.shape)
    print('1st training image shape ',x_train[0].shape)

    x_train = x_train.astype('float32') / 255
    x_valid = x_valid.astype('float32') / 255
    x_test = x_test.astype('float32') / 255

    return x_train, x_valid, x_test

def convolutional_model():
    model = Sequential([
        Input(shape=(100,100,3)),
        Conv2D(filters = 16, kernel_size = 2,padding='same', activation='relu'),
        MaxPooling2D(pool_size=2),
        Conv2D(filters = 32,kernel_size = 2,activation= 'relu',padding='same'),
        MaxPooling2D(pool_size=2),
        Conv2D(filters = 64,kernel_size = 2,activation= 'relu',padding='same'),
        MaxPooling2D(pool_size=2),
        Conv2D(filters = 128,kernel_size = 2,activation= 'relu',padding='same'),
        MaxPooling2D(pool_size=2),
        Dropout(0.3),
        Flatten(),
        Dense(150, activation='relu'),
        Dropout(0.4),
        Dense(196,activation = 'softmax')
    ])

    model.compile(
        loss='categorical_crossentropy',
        optimizer='rmsprop',
        metrics=['accuracy'])
    
    return model

def save_model_weights(model):
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    model.save_weights(f"./artifacts/models/cnn_model_{timestamp}.weights.h5")

def accuracy_score(model, x_test, y_test):
    acc_score = model.evaluate(x_test, y_test)
    print('\n', 'Test accuracy:', acc_score[1])


def visualization_with_prediction(model, x_test, y_test, target_labels):
    predictions = model.predict(x_test)
    fig = plt.figure(figsize=(16, 9))
    for i, idx in enumerate(np.random.choice(x_test.shape[0], size=16, replace=False)):
        ax = fig.add_subplot(4, 4, i + 1, xticks=[], yticks=[])
        ax.imshow(np.squeeze(x_test[idx]))
        pred_idx = np.argmax(predictions[idx])
        true_idx = np.argmax(y_test[idx])
        ax.set_title("{} ({})".format(target_labels[pred_idx], target_labels[true_idx]),
                    color=("green" if pred_idx == true_idx else "red"))
        
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    plt.savefig(f"./artifacts/visualizations/visualization_with_prediction_{timestamp}.png")
    plt.close()    
        
def visualization_epochs(history):
    plt.figure(1)  
    plt.subplot(211)  
    plt.plot(history.history['accuracy'])  
    plt.plot(history.history['val_accuracy'])  
    plt.title('model accuracy')  
    plt.ylabel('accuracy')  
    plt.xlabel('epoch')  
    plt.legend(['train', 'test'], loc='upper left')  
    plt.subplot(212)  
    plt.plot(history.history['loss'])  
    plt.plot(history.history['val_loss'])  
    plt.title('model loss')  
    plt.ylabel('loss')  
    plt.xlabel('epoch')  
    plt.legend(['train', 'test'], loc='upper left')

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    plt.savefig(f"./artifacts/visualizations/training_visualization_{timestamp}.png")
    plt.close()

def main():
    train_dir = '../../datasets/dataset-fruits-360/Training'
    test_dir = '../../datasets/dataset-fruits-360/Test'

    tm = TimeMeasuring("./artifacts/time_log/cnn_time_log_")
    x_train, y_train, target_labels_train = load_data(train_dir)
    x_test, y_test, target_labels_test = load_data(test_dir)
    tm.took("Loading the dataset")

    x_train, y_train, x_test, y_test, target_labels = balance_y_train_and_test(x_train, y_train, x_test, y_test, target_labels_train, target_labels_test)
    tm.took("Balancing the dataset")

    y_train = to_one_hot_encoding(y_train)
    y_test = to_one_hot_encoding(y_test)
    tm.took("Converting to One Hot Encoding")

    print("Y_train_one-hot-encoding:", y_train.shape)
    print("Y_test_one-hot-encoding:", y_test.shape)

    tm.reset()
    x_valid, x_test = divide_set(x_test, 7000)
    y_valid, y_test = divide_set(y_test, 7000)
    tm.took("Splitting test set into validation- and test set")

    x_train, x_valid, x_test = data_preprocessing(x_train, x_valid, x_test)
    tm.took("Data Preprocessing")

    model = convolutional_model()

    tm.reset()
    training_history = model.fit(x_train,y_train,
        batch_size = 32,
        epochs=30,
        validation_data=(x_valid, y_valid),
        verbose=2, shuffle=True)
    tm.took("Training CNN")

    save_model_weights(model)
    tm.took("Saving Model")

    _ = model.predict(x_test)
    tm.took("Prediction")

    tm.save_log()
    
    accuracy_score(model, x_test, y_test)

    visualization_with_prediction(model, x_test, y_test, target_labels)

    visualization_epochs(training_history)

if __name__ == "__main__":
    main()