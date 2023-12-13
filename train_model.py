import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from data_loader import load_data
from callbacks import create_model_checkpoint

# Defining the model
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(units=128, activation='relu'))
model.add(Dense(units=1, activation='sigmoid'))

# Compiling the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Loading the data
data_dir = '/home/taylor/Desktop/AIprototype/trainImages/'
training_set = load_data(data_dir)

# Creating model checkpoints
model_checkpoint = create_model_checkpoint()

# Training model with the ModelCheckpoint callback
model.fit(
    training_set,
    epochs=10,
    callbacks=[model_checkpoint]
)

# Saving model for future use (optional, as the best model is already saved by ModelCheckpoint)
model.save('scene_classifier_model.h5')
