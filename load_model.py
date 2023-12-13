import tensorflow as tf
from data_loader import load_data
from callbacks import create_model_checkpoint

# Load the data
data_dir = '/home/taylor/Desktop/AIprototype/trainImages/'
new_data_set = load_data(data_dir)

# Load the model
model = tf.keras.models.load_model('model_checkpoints/best_model.h5')

# Continue training with the new data
model.fit(
    new_data_set,
    epochs=5,
    callbacks=[create_model_checkpoint()]  # Optional: You can continue saving checkpoints
)

# Saving updated model
model.save('updated_scene_classifier_model.h5')
