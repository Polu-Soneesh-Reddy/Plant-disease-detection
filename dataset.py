import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator

class PlantDataset:
    def __init__(self, data_dir, img_size=(256, 256), batch_size=32):
        self.data_dir = os.path.normpath(data_dir)  # Normalize path for Windows
        self.img_size = img_size
        self.batch_size = batch_size
        self.class_names = ['healthy', 'powdery', 'rust']
        
        # Verify paths
        self._verify_paths()
        
    def _verify_paths(self):
        """Check all required directories exist with images"""
        required = ['train', 'val', 'test']
        for folder in required:
            path = os.path.join(self.data_dir, folder)
            if not os.path.exists(path):
                raise FileNotFoundError(f"Missing folder: {path}")
            
            for class_name in self.class_names:
                class_path = os.path.join(path, class_name)
                if not os.path.exists(class_path):
                    raise FileNotFoundError(f"Missing class folder: {class_path}")
                
                if not os.listdir(class_path):
                    raise ValueError(f"No images found in: {class_path}")

    def get_generator(self, subset, augment=False):
        """Create generator for train/val/test"""
        if subset not in ['train', 'val', 'test']:
            raise ValueError("subset must be 'train', 'val' or 'test'")
            
        path = os.path.join(self.data_dir, subset)
        
        if augment:
            return ImageDataGenerator(
                rescale=1./255,
                rotation_range=30,
                width_shift_range=0.2,
                height_shift_range=0.2,
                shear_range=0.2,
                zoom_range=0.2,
                horizontal_flip=True
            ).flow_from_directory(
                path,
                target_size=self.img_size,
                batch_size=self.batch_size,
                class_mode='categorical',
                classes=self.class_names,
                shuffle=augment
            )
        else:
            return ImageDataGenerator(
                rescale=1./255
            ).flow_from_directory(
                path,
                target_size=self.img_size,
                batch_size=self.batch_size,
                class_mode='categorical',
                classes=self.class_names,
                shuffle=False
            )