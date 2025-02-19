import tensorflow as tf

def main():
    try:
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
        tf.config.experimental_connect_to_cluster(tpu)
        tf.tpu.experimental.initialize_tpu_system(tpu)
        print(f"TensorFlow can access {len(tf.config.list_logical_devices('TPU'))} TPU cores")
        print("TensorFlow and TPU setup verification successful!")
    except Exception as e:
        print("Error: TensorFlow and TPU setup verification failed.")
        print(e)

if __name__ == "__main__":
    main() 