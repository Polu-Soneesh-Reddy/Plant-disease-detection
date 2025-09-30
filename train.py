import os
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Input, Conv2D, GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from utils.dataset import PlantDataset
import matplotlib.pyplot as plt

def build_model(img_size=(512, 512)):
    input_layer = Input(shape=(*img_size, 3))
    
    base_model = EfficientNetB0(
        include_top=False,
        weights='imagenet',
        input_tensor=input_layer
    )
    
    x = base_model.output
    x = Conv2D(256, (3,3), activation='relu', name='top_conv_layer')(x)
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    outputs = Dense(3, activation='softmax')(x)
    
    return Model(inputs=input_layer, outputs=outputs)

def train():
    # Configuration
    DATA_DIR = 'data'
    IMG_SIZE = (512, 512)
    BATCH_SIZE = 16
    EPOCHS = 100
    
    # Dataset
    dataset = PlantDataset(DATA_DIR, IMG_SIZE)
    train_gen, val_gen = dataset.get_generators(BATCH_SIZE)
    
    # Model
    model = build_model(IMG_SIZE)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-5),
        loss='categorical_crossentropy',
        metrics=['accuracy', 
                tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.Recall(name='recall')]
    )
    
    # Callbacks
    os.makedirs('models', exist_ok=True)
    callbacks = [
        ModelCheckpoint(
            'models/best_model.keras',
            monitor='val_accuracy',
            save_best_only=True
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=5,
            min_lr=1e-7
        )
    ]
    
    # Training
    history = model.fit(
        train_gen,
        epochs=EPOCHS,
        validation_data=val_gen,
        callbacks=callbacks
    )
    
    # Plot results
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train')
    plt.plot(history.history['val_accuracy'], label='Validation')
    plt.title('Model Accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train')
    plt.plot(history.history['val_loss'], label='Validation')
    plt.title('Model Loss')
    plt.legend()
    
    plt.savefig('training_history.png')
    plt.show()

if __name__ == "__main__":
    train()