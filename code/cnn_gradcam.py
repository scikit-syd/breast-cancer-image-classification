import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping

# PARAMETERS
image_size = (128, 128)
batch_size = 32
epochs = 50

# LOAD DATASET
data = {"images": [], "labels": []}

data_dirs = [
    "[training_data_here]",
    "[test_data_here]"
]

for dir_path in data_dirs:
    for class_name in os.listdir(dir_path):
        class_path = os.path.join(dir_path, class_name)
        if not os.path.isdir(class_path):
            continue
        for img_name in os.listdir(class_path):
            img_path = os.path.join(class_path, img_name)

            if not img_name.lower().endswith((".png", ".jpg", ".jpeg", ".tiff", ".bmp")):
                continue

            img = Image.open(img_path).convert("L").resize(image_size)
            img_array = np.array(img, dtype="float32") / 255.0
            img_array = np.expand_dims(img_array, axis=-1)
            data["images"].append(img_array)
            data["labels"].append(class_name)

X = np.array(data["images"], dtype="float32")
y = np.array(data["labels"])

# ENCODE LABELS
le = LabelEncoder()
y_encoded = le.fit_transform(y)
y_categorical = to_categorical(y_encoded)
num_classes = y_categorical.shape[1]


# TRAIN / VAL / TEST SPLIT
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y_categorical, test_size=0.2, random_state=42, stratify=y
)

X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full, test_size=0.2,
    random_state=42,
    stratify=np.argmax(y_train_full, axis=1)
)


# DATA AUGMENTATION
datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.05,
    height_shift_range=0.05,
    horizontal_flip=True,
    zoom_range=0.05
)
datagen.fit(X_train)


# BUILD CNN
inputs = Input(shape=(128, 128, 1))

x = Conv2D(32, (3,3), activation="relu", name="conv_1")(inputs)
x = MaxPooling2D((2,2))(x)

x = Conv2D(64, (3,3), activation="relu", name="conv_2")(x)
x = MaxPooling2D((2,2))(x)

# 🚨 last conv layer we will use for Grad-CAM
x = Conv2D(128, (3,3), activation="relu", name="conv_3")(x)
last_conv_layer = MaxPooling2D((2,2), name="last_maxpool")(x)

flat = Flatten()(last_conv_layer)
dense = Dense(128, activation="relu")(flat)
drop = Dropout(0.5)(dense)
outputs = Dense(num_classes, activation="softmax")(drop)

model = Model(inputs=inputs, outputs=outputs)
model.summary()

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])


# TRAINING
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

history = model.fit(
    datagen.flow(X_train, y_train, batch_size=batch_size),
    validation_data=(X_val, y_val),
    epochs=epochs,
    callbacks=[early_stop]
)


# EVALUATION
print("Validation Accuracy:", model.evaluate(X_val, y_val)[1])
print("Test Accuracy:", model.evaluate(X_test, y_test)[1])


# GRAD-CAM FUNCTION
def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    last_conv_layer = model.get_layer(last_conv_layer_name)

    grad_model = tf.keras.models.Model(
        [model.input],
        [last_conv_layer.output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        pred_output = predictions[:, pred_index]

    grads = tape.gradient(pred_output, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]

    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()


# GENERATE GRAD-CAM ON 1 IMAGE
example_img = X_test[0]
example_img_batch = np.expand_dims(example_img, axis=0)

pred_class = np.argmax(model.predict(example_img_batch))
last_conv_name = "conv_3"   # matches model architecture

heatmap = make_gradcam_heatmap(example_img_batch, model, last_conv_name, pred_index=pred_class)


# DISPLAY HEATMAP
pred_label = le.inverse_transform([pred_class])[0]  # get the class name

plt.figure(figsize=(10,4))

plt.subplot(1,3,1)
plt.title("Original")
plt.imshow(example_img.squeeze(), cmap="gray")
plt.axis("off")

plt.subplot(1,3,2)
plt.title(f"Grad-CAM Heatmap\nClass: {pred_label}")
plt.imshow(heatmap, cmap='jet')
plt.axis("off")

plt.subplot(1,3,3)
plt.title(f"Overlay\nClass: {pred_label}")
plt.imshow(example_img.squeeze(), cmap="gray")
plt.imshow(heatmap, cmap='jet', alpha=0.5)
plt.axis("off")

plt.tight_layout()
plt.show()

