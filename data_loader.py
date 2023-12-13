from tensorflow.keras.preprocessing.image import ImageDataGenerator

def load_data(data_dir):
    train_datagen = ImageDataGenerator(rescale=1./255)
    training_set = train_datagen.flow_from_directory(data_dir, target_size=(64, 64), batch_size=32, class_mode='binary')
    return training_set
