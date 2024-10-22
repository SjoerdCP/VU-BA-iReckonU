from LoadDataset import augment_and_save_images
import tensorflow as tf

tf.keras.utils.set_random_seed(12345)

augment_and_save_images('./PublicDataset/Train/Undistorted', './NewAugmentedDataset/Sharp')
augment_and_save_images('./PublicDataset/Train/Naturally-Blurred', './NewAugmentedDataset/Blurry')