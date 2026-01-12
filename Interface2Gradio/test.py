import tensorflow as tf

print("=== Modern TensorFlow Test ===")
print(f"TF Version: {tf.__version__}")

# Modern way to check GPU
gpus = tf.config.list_physical_devices('GPU')
print(f"GPU devices: {gpus}")

if gpus:
    for gpu in gpus:
        print(f"GPU: {gpu}")
        # Try to enable memory growth
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except:
            pass
    print("SUCCESS: TensorFlow detected GPU!")
else:
    print("No GPU detected by TensorFlow")