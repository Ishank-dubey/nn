import tensorflow as tf

import numpy as np

def test_eval_mse(target):
    y_hat = np.array([2.4, 4.2])
    y_tmp = np.array([2.3, 4.1])
    result = target(y_hat, y_tmp)
    
    assert np.isclose(result, 0.005, atol=1e-6), f"Wrong value. Expected 0.005, got {result}"
    
    y_hat = np.array([3.] * 10)
    y_tmp = np.array([3.] * 10)
    result = target(y_hat, y_tmp)
    assert np.isclose(result, 0.), f"Wrong value. Expected 0.0 when y_hat == t_tmp, but got {result}"
    
    y_hat = np.array([3.])
    y_tmp = np.array([0.])
    result = target(y_hat, y_tmp)
    assert np.isclose(result, 4.5), f"Wrong value. Expected 4.5, but got {result}. Remember the square termn"
    
    y_hat = np.array([3.] * 5)
    y_tmp = np.array([2.] * 5)
    result = target(y_hat, y_tmp)
    assert np.isclose(result, 0.5), f"Wrong value. Expected 0.5, but got {result}. Remember to divide by (2*m)"
    
    print("\033[92m All tests passed.")
    
def test_eval_cat_err(target):
    y_hat = np.array([1, 0, 1, 1, 1, 0])
    y_tmp = np.array([0, 1, 0, 0, 0, 1])
    result = target(y_hat, y_tmp)
    assert not np.isclose(result, 6.), f"Wrong value. Expected 1, but got {result}. Did you divided by m?"
    
    y_hat = np.array([1, 2, 0])
    y_tmp = np.array([1, 2, 3])
    result = target(y_hat, y_tmp)
    assert np.isclose(result, 1./3., atol=1e-6), f"Wrong value. Expected 0.333, but got {result}"
    
    y_hat = np.array([1, 0, 1, 1, 1, 0])
    y_tmp = np.array([1, 1, 1, 0, 0, 0])
    result = target(y_hat, y_tmp)
    assert np.isclose(result, 3./6., atol=1e-6), f"Wrong value. Expected 0.5, but got {result}"
    
    y_hat = np.array([[1], [2], [0], [3]])
    y_tmp = np.array([[1], [2], [1], [3]])
    res_tmp =  target(y_hat, y_tmp)
    assert type(res_tmp) != np.ndarray, f"The output must be an scalar but got {type(res_tmp)}"
    
    print("\033[92m All tests passed.")
    
def model_test(target, classes, input_size):
    target.build(input_shape=(None,input_size))
    expected_lr = 0.01
    
    assert len(target.layers) == 3, \
        f"Wrong number of layers. Expected 3 but got {len(target.layers)}"
    assert target.input.shape.as_list() == [None, input_size], \
        f"Wrong input shape. Expected [None,  {input_size}] but got {target.input.shape.as_list()}"
    i = 0
    expected = [[tf.keras.layers.Dense, [None, 120], tf.keras.activations.relu],
                [tf.keras.layers.Dense, [None, 40], tf.keras.activations.relu],
                [tf.keras.layers.Dense, [None, classes], tf.keras.activations.linear]]

    for layer in target.layers:
        assert type(layer) == expected[i][0], \
            f"Wrong type in layer {i}. Expected {expected[i][0]} but got {type(layer)}"
        assert layer.output.shape.as_list() == expected[i][1], \
            f"Wrong number of units in layer {i}. Expected {expected[i][1]} but got {layer.output.shape.as_list()}"
        assert layer.activation == expected[i][2], \
            f"Wrong activation in layer {i}. Expected {expected[i][2]} but got {layer.activation}"
        assert layer.kernel_regularizer == None, "You must not specify any regularizer for any layer"
        i = i + 1
        
    assert type(target.loss)==tf.keras.losses.SparseCategoricalCrossentropy, f"Wrong loss function. Expected {tf.keras.losses.SparseCategoricalCrossentropy}, but got {target.loss}"
    assert type(target.optimizer)==tf.keras.optimizers.Adam, f"Wrong loss function. Expected {tf.keras.optimizers.Adam}, but got {target.optimizer}"
    lr = target.optimizer.learning_rate.numpy()
    assert np.isclose(lr, expected_lr, atol=1e-8), f"Wrong learning rate. Expected {expected_lr}, but got {lr}"
    assert target.loss.get_config()['from_logits'], f"Set from_logits=True in loss function"

    print("\033[92mAll tests passed!")
    
def model_s_test(target, classes, input_size):
    target.build(input_shape=(None,input_size))
    expected_lr = 0.01
    
    assert len(target.layers) == 2, \
        f"Wrong number of layers. Expected 3 but got {len(target.layers)}"
    assert target.input.shape.as_list() == [None, input_size], \
        f"Wrong input shape. Expected [None,  {input_size}] but got {target.input.shape.as_list()}"
    i = 0
    expected = [[tf.keras.layers.Dense, [None, 6], tf.keras.activations.relu],
                [tf.keras.layers.Dense, [None, classes], tf.keras.activations.linear]]

    for layer in target.layers:
        assert type(layer) == expected[i][0], \
            f"Wrong type in layer {i}. Expected {expected[i][0]} but got {type(layer)}"
        assert layer.output.shape.as_list() == expected[i][1], \
            f"Wrong number of units in layer {i}. Expected {expected[i][1]} but got {layer.output.shape.as_list()}"
        assert layer.activation == expected[i][2], \
            f"Wrong activation in layer {i}. Expected {expected[i][2]} but got {layer.activation}"
        assert layer.kernel_regularizer == None, "You must not specify any regularizer any layer"
        i = i + 1
        
    assert type(target.loss)==tf.keras.losses.SparseCategoricalCrossentropy, f"Wrong loss function. Expected {tf.keras.losses.SparseCategoricalCrossentropy}, but got {target.loss}"
    assert type(target.optimizer)==tf.keras.optimizers.Adam, f"Wrong loss function. Expected {tf.keras.optimizers.Adam}, but got {target.optimizer}"
    lr = target.optimizer.learning_rate.numpy()
    assert np.isclose(lr, expected_lr, atol=1e-8), f"Wrong learning rate. Expected {expected_lr}, but got {lr}"
    assert target.loss.get_config()['from_logits'], f"Set from_logits=True in loss function"

    print("\033[92mAll tests passed!")
    
def model_r_test(target, classes, input_size):
    target.build(input_shape=(None,input_size))
    expected_lr = 0.01
    print("ddd")
    assert len(target.layers) == 3, \
        f"Wrong number of layers. Expected 3 but got {len(target.layers)}"
    assert target.input.shape.as_list() == [None, input_size], \
        f"Wrong input shape. Expected [None,  {input_size}] but got {target.input.shape.as_list()}"
    i = 0
    expected = [[tf.keras.layers.Dense, [None, 120], tf.keras.activations.relu, (tf.keras.regularizers.l2, 0.1)],
                [tf.keras.layers.Dense, [None, 40], tf.keras.activations.relu, (tf.keras.regularizers.l2, 0.1)],
                [tf.keras.layers.Dense, [None, classes], tf.keras.activations.linear, None]]

    for layer in target.layers:
        assert type(layer) == expected[i][0], \
            f"Wrong type in layer {i}. Expected {expected[i][0]} but got {type(layer)}"
        assert layer.output.shape.as_list() == expected[i][1], \
            f"Wrong number of units in layer {i}. Expected {expected[i][1]} but got {layer.output.shape.as_list()}"
        assert layer.activation == expected[i][2], \
            f"Wrong activation in layer {i}. Expected {expected[i][2]} but got {layer.activation}"
        if not (expected[i][3] == None):
            assert type(layer.kernel_regularizer) == expected[i][3][0], f"Wrong regularizer. Expected L2 regularizer but got {type(layer.kernel_regularizer)}"
            assert np.isclose(layer.kernel_regularizer.l2,  expected[i][3][1]), f"Wrong regularization factor. Expected {expected[i][3][1]}, but got {layer.kernel_regularizer.l2}"
        else:
            assert layer.kernel_regularizer == None, "You must not specify any regularizer for the 3th layer"
        i = i + 1
        
    assert type(target.loss)==tf.keras.losses.SparseCategoricalCrossentropy, f"Wrong loss function. Expected {tf.keras.losses.SparseCategoricalCrossentropy}, but got {target.loss}"
    assert type(target.optimizer)==tf.keras.optimizers.Adam, f"Wrong loss function. Expected {tf.keras.optimizers.Adam}, but got {target.optimizer}"
    lr = target.optimizer.learning_rate.numpy()
    assert np.isclose(lr, expected_lr, atol=1e-8), f"Wrong learning rate. Expected {expected_lr}, but got {lr}"
    assert target.loss.get_config()['from_logits'], f"Set from_logits=True in loss function"

    print("\033[92mAll tests passed!")
