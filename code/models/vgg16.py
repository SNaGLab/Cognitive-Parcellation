import keras
import tensorflow as tf
from tensorflow.keras import backend as K

class VGG16:

	def __init__(self):
		K.set_image_dim_ordering('th')

	
	#Code snippet needed to read activation values from each layer of the pre-trained artificial neural networks
	def get_activations(self,model, layer, X_batch):
	    #keras.backend.function(inputs, outputs, updates=None)
	    get_activations = tf.keras.backend.function([model.layers[0].input, tf.keras.backend.learning_phase()], [model.layers[layer].output,])
	    #The learning phase flag is a bool tensor (0 = test, 1 = train)
	    activations = get_activations([X_batch,0])
	    return activations


	def create_model(self):
		from __future__ import absolute_import
		from __future__ import division
		from __future__ import print_function

		import os
		from tensorflow.keras import layers
		from tensorflow.keras import models
		import tensorflow.keras.utils as keras_utils
		from tensorflow.keras.layers import Lambda

		from  keras.applications import imagenet_utils
		from  keras.applications.imagenet_utils import decode_predictions
		from  keras_applications.imagenet_utils import _obtain_input_shape

		preprocess_input = imagenet_utils.preprocess_input

		WEIGHTS_PATH = ('https://github.com/fchollet/deep-learning-models/'
				'releases/download/v0.1/'
				'vgg16_weights_tf_dim_ordering_tf_kernels.h5')
		WEIGHTS_PATH_NO_TOP = ('https://github.com/fchollet/deep-learning-models/'
				       'releases/download/v0.1/'
				       'vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5')


		def VGG16(include_top=True,
			  weights='imagenet',
			  input_tensor=None,
			  input_shape=None,
			  pooling=None,
			  classes=1000,
			  lambda_mask=None,
			  **kwargs):
		    
		    backend= tf.keras.backend
		    layers = tf.keras.layers
		    models = tf.keras.models
		    keras_utils = tf.keras.utils
		    
		    if not (weights in {'imagenet', None} or os.path.exists(weights)):
			raise ValueError('The `weights` argument should be either '
				         '`None` (random initialization), `imagenet` '
				         '(pre-training on ImageNet), '
				         'or the path to the weights file to be loaded.')

		    if weights == 'imagenet' and include_top and classes != 1000:
			raise ValueError('If using `weights` as `"imagenet"` with `include_top`'
				         ' as true, `classes` should be 1000')
		    # Determine proper input shape
		    input_shape = _obtain_input_shape(input_shape,
				                      default_size=224,
				                      min_size=32,
				                      data_format=backend.image_data_format(),
				                      require_flatten=include_top,
				                      weights=weights)

		    if input_tensor is None:
			img_input = layers.Input(shape=input_shape)
		    else:
			if not backend.is_keras_tensor(input_tensor):
			    img_input = layers.Input(tensor=input_tensor, shape=input_shape)
			else:
			    img_input = input_tensor
		    # Block 1
		    x = layers.Conv2D(64, (3, 3),
				      activation='relu',
				      padding='same',
				      name='block1_conv1')(img_input)
		    
		    if lambda_mask is not None:
			block_1_conv_1_mask  = np.reshape(lambda_mask[0:3211264], (64, 224, 224))
		    else:
			block_1_conv_1_mask = np.ones(shape=((64, 224, 224)))
		    
		    block_1_conv_1_mask  = backend.variable(block_1_conv_1_mask)
		    block_1_conv_1_lambda = Lambda(lambda x: x * block_1_conv_1_mask)(x)
		    
		    x = layers.Conv2D(64, (3, 3),
				      activation='relu',
				      padding='same',
				      name='block1_conv2')(block_1_conv_1_lambda)
		    
		    if lambda_mask is not None:
			block_1_conv_2_mask  = np.reshape(lambda_mask[3211264:6422528], (64, 224, 224))
		    else:
			block_1_conv_2_mask = np.ones(shape=((64, 224, 224)))
		    
		    block_1_conv_2_mask  = backend.variable(block_1_conv_2_mask)
		    block_1_conv_2_lambda = Lambda(lambda x: x * block_1_conv_2_mask)(x)
		    
		    
		    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(block_1_conv_2_lambda)

		    # Block 2
		    x = layers.Conv2D(128, (3, 3),
				      activation='relu',
				      padding='same',
				      name='block2_conv1')(x)
		    
		    if lambda_mask is not None:
			block_2_conv_1_mask  = np.reshape(lambda_mask[6422528:8028160], (128, 112, 112))
		    else:
			block_2_conv_1_mask = np.ones(shape=((128, 112, 112)))
		    
		    block_2_conv_1_mask  = backend.variable(block_2_conv_1_mask)
		    block_2_conv_1_lambda = Lambda(lambda x: x * block_2_conv_1_mask)(x)
		    
		    x = layers.Conv2D(128, (3, 3),
				      activation='relu',
				      padding='same',
				      name='block2_conv2')(block_2_conv_1_lambda)
		    
		    
		    if lambda_mask is not None:
			block_2_conv_2_mask  = np.reshape(lambda_mask[8028160:9633792], (128, 112, 112))
		    else:
			block_2_conv_2_mask = np.ones(shape=((128, 112, 112)))
		    
		    block_2_conv_2_mask  = backend.variable(block_2_conv_2_mask)
		    block_2_conv_2_lambda = Lambda(lambda x: x * block_2_conv_2_mask)(x)
		    
		    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(block_2_conv_2_lambda)

		    # Block 3
		    x = layers.Conv2D(256, (3, 3),
				      activation='relu',
				      padding='same',
				      name='block3_conv1')(x)
		    
		    if lambda_mask is not None:
			block_3_conv_1_mask  = np.reshape(lambda_mask[9633792:10436608], (256, 56, 56))
		    else:
			block_3_conv_1_mask = np.ones(shape=((256, 56, 56)))
		    
		    block_3_conv_1_mask  = backend.variable(block_3_conv_1_mask)
		    block_3_conv_1_lambda = Lambda(lambda x: x * block_3_conv_1_mask)(x)
		    
		    
		    x = layers.Conv2D(256, (3, 3),
				      activation='relu',
				      padding='same',
				      name='block3_conv2')(block_3_conv_1_lambda)
		    
		    if lambda_mask is not None:
			block_3_conv_2_mask  = np.reshape(lambda_mask[10436608:11239424], (256, 56, 56))
		    else:
			block_3_conv_2_mask = np.ones(shape=((256, 56, 56)))
		    
		    block_3_conv_2_mask  = backend.variable(block_3_conv_2_mask)
		    block_3_conv_2_lambda = Lambda(lambda x: x * block_3_conv_2_mask)(x)
		    
		    x = layers.Conv2D(256, (3, 3),
				      activation='relu',
				      padding='same',
				      name='block3_conv3')( block_3_conv_2_lambda)
		    
		    if lambda_mask is not None:
			block_3_conv_3_mask  = np.reshape(lambda_mask[11239424:12042240], (256, 56, 56))
		    else:
			block_3_conv_3_mask = np.ones(shape=((256, 56, 56)))
		    
		    block_3_conv_3_mask  = backend.variable(block_3_conv_3_mask)
		    block_3_conv_3_lambda = Lambda(lambda x: x * block_3_conv_3_mask)(x)
		    
		    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(block_3_conv_3_lambda)

		    # Block 4
		    x = layers.Conv2D(512, (3, 3),
				      activation='relu',
				      padding='same',
				      name='block4_conv1')(x)
		    
		    if lambda_mask is not None:
			block_4_conv_1_mask  = np.reshape(lambda_mask[12042240:12443648], (512, 28, 28))
		    else:
			block_4_conv_1_mask = np.ones(shape=((512, 28, 28)))
		    
		    block_4_conv_1_mask  = backend.variable(block_4_conv_1_mask)
		    block_4_conv_1_lambda = Lambda(lambda x: x * block_4_conv_1_mask)(x)
		    
		    
		    x = layers.Conv2D(512, (3, 3),
				      activation='relu',
				      padding='same',
				      name='block4_conv2')(block_4_conv_1_lambda)
		    
		    if lambda_mask is not None:
			block_4_conv_2_mask  = np.reshape(lambda_mask[12443648:12845056], (512, 28, 28))
		    else:
			block_4_conv_2_mask = np.ones(shape=((512, 28, 28)))
		    
		    block_4_conv_2_mask  = backend.variable(block_4_conv_2_mask)
		    block_4_conv_2_lambda = Lambda(lambda x: x * block_4_conv_2_mask)(x)
		    
		    
		    x = layers.Conv2D(512, (3, 3),
				      activation='relu',
				      padding='same',
				      name='block4_conv3')(block_4_conv_2_lambda)
		    
		    
		    if lambda_mask is not None:
			block_4_conv_3_mask  = np.reshape(lambda_mask[12845056:13246464], (512, 28, 28))
		    else:
			block_4_conv_3_mask = np.ones(shape=((512, 28, 28)))
		    
		    block_4_conv_3_mask  = backend.variable(block_4_conv_3_mask)
		    block_4_conv_3_lambda = Lambda(lambda x: x * block_4_conv_3_mask)(x)
		    
		    
		    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')( block_4_conv_3_lambda)

		    # Block 5
		    x = layers.Conv2D(512, (3, 3),
				      activation='relu',
				      padding='same',
				      name='block5_conv1')(x)
		    
		    if lambda_mask is not None:
			block_5_conv_1_mask  = np.reshape(lambda_mask[13246464:13346816], (512, 14, 14))
		    else:
			block_5_conv_1_mask = np.ones(shape=((512, 14, 14)))
		    
		    block_5_conv_1_mask  = backend.variable(block_5_conv_1_mask)
		    block_5_conv_1_lambda = Lambda(lambda x: x * block_5_conv_1_mask)(x)
		    
		    x = layers.Conv2D(512, (3, 3),
				      activation='relu',
				      padding='same',
				      name='block5_conv2')(block_5_conv_1_lambda)
		    
		    if lambda_mask is not None:
			block_5_conv_2_mask  = np.reshape(lambda_mask[13346816:13447168], (512, 14, 14))
		    else:
			block_5_conv_2_mask = np.ones(shape=((512, 14, 14)))
		    
		    block_5_conv_2_mask  = backend.variable(block_5_conv_2_mask)
		    block_5_conv_2_lambda = Lambda(lambda x: x * block_5_conv_2_mask)(x)
		    
		    x = layers.Conv2D(512, (3, 3),
				      activation='relu',
				      padding='same',
				      name='block5_conv3')(block_5_conv_2_lambda)
		    
		    if lambda_mask is not None:
			block_5_conv_3_mask  = np.reshape(lambda_mask[13447168:13547520], (512, 14, 14))
		    else:
			block_5_conv_3_mask = np.ones(shape=((512, 14, 14)))
		    
		    block_5_conv_3_mask  = backend.variable(block_5_conv_3_mask)
		    block_5_conv_3_lambda = Lambda(lambda x: x * block_5_conv_3_mask)(x)
		    
		    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')( block_5_conv_3_lambda)

		    if include_top:
			# Classification block
			x = layers.Flatten(name='flatten')(x)
			x = layers.Dense(4096, activation='relu', name='fc1')(x)
		
			if lambda_mask is not None:
			    block_fc1_mask  = np.reshape(lambda_mask[13547520:13551616], (4096,))
			else:
			    block_fc1_mask = np.ones(shape=((4096,)))
			block_fc1_mask  = backend.variable(block_fc1_mask)
			block_fc1_lambda = Lambda(lambda x: x * block_fc1_mask)(x)
		    
			x = layers.Dense(4096, activation='relu', name='fc2')(block_fc1_lambda)
		
			if lambda_mask is not None:
			    block_fc2_mask  = np.reshape(lambda_mask[13551616:13555712], (4096,))
			else:
			    block_fc2_mask = np.ones(shape=((4096,)))
			block_fc2_mask  = backend.variable(block_fc2_mask)
			block_fc2_lambda = Lambda(lambda x: x * block_fc2_mask)(x)
		
			x = layers.Dense(classes, activation='softmax', name='predictions')(block_fc2_lambda)
		    else:
			if pooling == 'avg':
			    x = layers.GlobalAveragePooling2D()(x)
			elif pooling == 'max':
			    x = layers.GlobalMaxPooling2D()(x)

		    # Ensure that the model takes into account
		    # any potential predecessors of `input_tensor`.
		    if input_tensor is not None:
			inputs = keras_utils.get_source_inputs(input_tensor)
		    else:
			inputs = img_input
		    # Create model.
		    model = models.Model(inputs, x, name='vgg16')

		    # Load weights.
		    if weights == 'imagenet':
			if include_top:
			    weights_path = keras_utils.get_file(
				'vgg16_weights_tf_dim_ordering_tf_kernels.h5',
				WEIGHTS_PATH,
				cache_subdir='models',
				file_hash='64373286793e3c8b2b4e3219cbf3544b')
			else:
			    weights_path = keras_utils.get_file(
				'vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5',
				WEIGHTS_PATH_NO_TOP,
				cache_subdir='models',
				file_hash='6d6bbae143d832006294945121d1f1fc')
			model.load_weights(weights_path)
			if backend.backend() == 'theano':
			    keras_utils.convert_all_kernels_in_model(model)
		    elif weights is not None:
			model.load_weights(weights)

		    return model
