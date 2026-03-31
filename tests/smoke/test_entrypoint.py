import json

# import runpy
import pytest
import tensorflow as tf

# @pytest.mark.smoke_sanity
# def test_module_entrypoint_imports() -> None:
#     runpy.run_module("cade", run_name="__main__")


@pytest.mark.smoke_sanity
def check_gpu() -> None:

    print(f'TF version: {tf.__version__}')
    print(json.dumps(tf.sysconfig.get_build_info(), indent=2))

    gpus = tf.config.list_physical_devices('GPU')
    print(f'GPUs: {gpus}')

    if not gpus:
        raise RuntimeError('TensorFlow did not detect a GPU.')

    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    with tf.device('/GPU:0'):
        a = tf.random.normal((1024, 1024))
        b = tf.random.normal((1024, 1024))
        c = tf.matmul(a, b)
        _ = c.numpy()

    print(f'Matmul device: {c.device}')
