from tensorflow.keras.callbacks import ModelCheckpoint

def create_model_checkpoint():
    checkpoint_filepath = 'model_checkpoints/best_model.h5'
    model_checkpoint = ModelCheckpoint(
        checkpoint_filepath,
        save_best_only=True,
        monitor='val_accuracy',
        mode='max',
        verbose=1
    )
    return model_checkpoint
