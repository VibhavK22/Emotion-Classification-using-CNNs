"""
Step 1: Train a CNN to recognize emotions from face images.

Expects the FER-2013 dataset in this layout:
    data/train/angry/*.jpg, data/train/happy/*.jpg, ...
    data/test/angry/*.jpg,  data/test/happy/*.jpg,  ...

Run:  python train.py
Output: emotion_model.keras and labels.txt
"""
import tensorflow as tf
from tensorflow.keras import layers, models

IMG_SIZE = 48      # FER-2013 images are 48x48 pixels
BATCH_SIZE = 64
EPOCHS = 50        # training stops early if it stops improving

# --- 1. Load the images -----------------------------------------------------
# Each subfolder name (angry, happy, ...) becomes a label automatically.
train_ds = tf.keras.utils.image_dataset_from_directory(
    "data/train",
    color_mode="grayscale",
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
)
test_ds = tf.keras.utils.image_dataset_from_directory(
    "data/test",
    color_mode="grayscale",
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
)

class_names = train_ds.class_names
print("Emotions found:", class_names)

# Save the label names so the webcam script knows what each output means.
with open("labels.txt", "w") as f:
    f.write("\n".join(class_names))

# Makes loading faster by preparing the next batch while the model trains.
train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
test_ds = test_ds.prefetch(tf.data.AUTOTUNE)

# --- 2. Build the CNN --------------------------------------------------------
# Data augmentation: randomly flip/rotate/zoom images so the model
# learns general patterns instead of memorizing the training pictures.
augment = models.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
])


def conv_block(filters):
    """Two convolution layers followed by pooling: the basic CNN building block."""
    return [
        layers.Conv2D(filters, 3, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.Conv2D(filters, 3, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),   # shrinks the image by half
        layers.Dropout(0.25),    # randomly turns off neurons to prevent overfitting
    ]


model = models.Sequential([
    layers.Input(shape=(IMG_SIZE, IMG_SIZE, 1)),
    augment,
    layers.Rescaling(1.0 / 255),  # pixel values 0-255 -> 0-1
    *conv_block(32),
    *conv_block(64),
    *conv_block(128),
    layers.Flatten(),
    layers.Dense(256, activation="relu"),
    layers.Dropout(0.5),
    layers.Dense(len(class_names), activation="softmax"),  # one probability per emotion
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)
model.summary()

# --- 3. Train ----------------------------------------------------------------
callbacks = [
    # Keep the best version of the model seen so far.
    tf.keras.callbacks.ModelCheckpoint(
        "emotion_model.keras", monitor="val_accuracy", save_best_only=True
    ),
    # Stop if the test accuracy hasn't improved for 8 epochs.
    tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy", patience=8, restore_best_weights=True
    ),
    # Lower the learning rate when progress stalls.
    tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3),
]

model.fit(train_ds, validation_data=test_ds, epochs=EPOCHS, callbacks=callbacks)

loss, acc = model.evaluate(test_ds)
print(f"\nFinal test accuracy: {acc:.1%}")
print("Saved model to emotion_model.keras")
