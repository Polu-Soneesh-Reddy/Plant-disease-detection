class Config:
    IMG_SIZE = (256, 256)  # Higher resolution for better accuracy
    BATCH_SIZE = 64        # Adjust based on your GPU memory
    EPOCHS = 30
    NUM_CLASSES = len(os.listdir("dataset"))  # Auto-detect
    DATA_PATH = "dataset"
    MODEL_SAVE_PATH = "trained_models/effnet_model.h5"
    
config = Config()