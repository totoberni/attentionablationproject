import tensorflow as tf
import os

def verify_tensorflow():
    """Verify TensorFlow installation and TPU availability."""
    print("\nTensorFlow Version:", tf.__version__)
    
    # Check if TPU is available
    try:
        resolver = tf.distribute.cluster_resolver.TPUClusterResolver()
        tf.config.experimental_connect_to_cluster(resolver)
        tf.tpu.experimental.initialize_tpu_system(resolver)
        strategy = tf.distribute.TPUStrategy(resolver)
        
        print("\nTPU Strategy Information:")
        print("- Number of replicas:", strategy.num_replicas_in_sync)
        print("- TPU devices:", tf.config.list_logical_devices('TPU'))
        print("\nTPU verification successful!")
        
    except Exception as e:
        print("\nError accessing TPU:", str(e))
        return False
    
    return True

if __name__ == "__main__":
    verify_tensorflow() 