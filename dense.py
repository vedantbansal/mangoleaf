import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import os
import cv2
from sklearn.preprocessing import LabelEncoder , OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from matplotlib.pyplot import imread
from matplotlib.pyplot import imshow
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.imagenet_utils import decode_predictions
from tensorflow.keras.applications.imagenet_utils import preprocess_input
from tensorflow.keras import layers
from tensorflow.keras.applications import DenseNet201
import pickle


dataset_path = os.listdir('MangoLeafBD')

class_labels = []

def plot_hist(hist):
    plt.plot(hist.history["accuracy"])
    #plt.plot(hist.history["val_accuracy"])
    plt.title("DenseNet201 Accuracy")
    plt.ylabel("Accuracy")
    plt.xlabel("Epochs")
    plt.legend(["train", "validation"], loc="upper left")
    plt.grid()
    plt.savefig('DenseNet201_acc',dpi=300)
    plt.show()


for item in dataset_path:
    all_classes = os.listdir('MangoLeafBD' + '/' +item)

    for room in all_classes:
        class_labels.append((item, str('dataset_path' + '/' +item) + '/' + room))

df = pd.DataFrame(data=class_labels, columns=['Labels', 'image'])

path = 'MangoLeafBD/'

im_size = 224

images = []
labels = []

for i in dataset_path:
    data_path = path + str(i)
    filenames = [i for i in os.listdir(data_path)]
    
    for f in filenames:
        img = cv2.imread(data_path + '/' + f)
        img = cv2.resize(img, (im_size, im_size))
        images.append(img)
        labels.append(i)


images = np.array(images)
images = images.astype('float32') / 255.0

y=df['Labels'].values
y_labelencoder = LabelEncoder ()
y = y_labelencoder.fit_transform (y)

y=y.reshape(-1,1)
ct = ColumnTransformer([('my_ohe', OneHotEncoder(), [0])], remainder='passthrough')
Y = ct.fit_transform(y).toarray()


images, Y = shuffle(images, Y, random_state=1)


train_x, test_x, train_y, test_y = train_test_split(images, Y, test_size=0.2, random_state=415)

NUM_CLASSES = 8
IMG_SIZE = 224

def build_model(num_classes):
    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    #x = img_augmentation(inputs)
    x = inputs
    model = DenseNet201(include_top=False, input_tensor=x, weights="imagenet")

    # Freeze the pretrained weights
    model.trainable = False

    # Rebuild top
    x = layers.Flatten()(model.output)
#     x = layers.BatchNormalization()(x)

#     top_dropout_rate = 0.2
#     x = layers.Dropout(top_dropout_rate, name="top_dropout")(x)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax", name="pred")(x)

    # Compile
    model = tf.keras.Model(inputs, outputs, name="DenseNet201")
#     optimizer = tf.keras.optimizers.Adam(learning_rate=1e-2)
    model.compile(
        optimizer='adam', loss="categorical_crossentropy", metrics=["accuracy"]
    )
    return model

model = build_model(num_classes=NUM_CLASSES)

epochs = 5


hist = model.fit(train_x, train_y, epochs=epochs, verbose=2)
model.save('DenseNet201_detect.h5')
plot_hist(hist)

preds = model.evaluate(test_x, test_y)
print ("Loss = " + str(preds[0]))
print ("Test Accuracy = " + str(preds[1]))

with open('DenseNet201History','wb') as f:
    pickle.dump(hist.history,f)

df = pd.DataFrame.from_dict(hist.history)
df.to_csv('DenseNet201_history.csv')

test_y = np.argmax(test_y, axis=1)

pred_y = model.predict(test_x)
pred_y = np.argmax(pred_y, axis=1)

result = confusion_matrix(test_y, pred_y)

print(result)