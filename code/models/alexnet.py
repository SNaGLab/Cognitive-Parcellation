import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten, MaxPooling2D, Input, ZeroPadding2D,merge,Lambda
from keras.layers.convolutional import Convolution2D
from keras.models import Model
from keras.utils.layer_utils import convert_all_kernels_in_model
from keras.optimizers import SGD
from keras import backend as K
from keras.engine import Layer
from keras.layers.core import Lambda
from keras.utils.vis_utils import plot_model
from keras.layers.core import  Lambda
from keras.regularizers import l2


class Alexnet:


	def __init__(self):
		K.set_image_dim_ordering('th')

	#Code snippet needed to read activation values from each layer of the pre-trained artificial neural networks
	def get_activations(self,model, layer, X_batch):
	    #keras.backend.function(inputs, outputs, updates=None)
	    get_activations = K.function([model.layers[0].input, K.learning_phase()], [model.layers[layer].output,])
	    #The learning phase flag is a bool tensor (0 = test, 1 = train)
	    activations = get_activations([X_batch,0])
	    return activations

	def crosschannelnormalization(self,alpha=1e-4, k=2, beta=0.75, n=5, **kwargs):
	    """
	    This is the function used for cross channel normalization in the original
	    Alexnet
	    """
	    def f(X):
		if K.image_dim_ordering()=='tf':
		    b, r, c, ch = X.get_shape()
		else:
		    b, ch, r, c = X.shape

		half = n // 2
		square = K.square(X)
		scale = k
		if K.image_dim_ordering() == 'th':
		    extra_channels = K.spatial_2d_padding(K.permute_dimensions(square, (0, 2, 3, 1)), ((0,0),(half,half)))
		    extra_channels = K.permute_dimensions(extra_channels, (0, 3, 1, 2))
		    for i in range(n):
		        scale += alpha * extra_channels[:, i:i+ch, :, :]
		if K.image_dim_ordering() == 'tf':
		    extra_channels = K.spatial_2d_padding(K.permute_dimensions(square, (0, 3, 1, 2)), (half, 0))
		    extra_channels = K.permute_dimensions(extra_channels, (0, 2, 3, 1))
		    for i in range(n):
		        scale += alpha * extra_channels[:, :, :, i:i+int(ch)]
		scale = scale ** beta
		return X / scale


	    return Lambda(f, output_shape=lambda input_shape: input_shape, **kwargs)


	#Helper Function to split tensor
	def splittensor(axis=1, ratio_split=1, id_split=0, **kwargs):
	    def f(X):
		div = K.shape(X)[axis] // ratio_split

		if axis == 0:
		    output = X[id_split*div:(id_split+1)*div, :, :, :]
		elif axis == 1:
		    output = X[:, id_split*div:(id_split+1)*div, :, :]
		elif axis == 2:
		    output = X[:, :, id_split*div:(id_split+1)*div, :]
		elif axis == 3:
		    output = X[:, :, :, id_split*div:(id_split+1)*div]
		else:
		    raise ValueError("This axis is not possible")
		return output

	    def g(input_shape):
		output_shape = list(input_shape)
		output_shape[axis] = output_shape[axis] // ratio_split
		return tuple(output_shape)


	    return Lambda(f, output_shape=lambda input_shape: g(input_shape), **kwargs)

	#Alexnet layer architecture class
	def create_model(img_shape=(3, 227, 227), n_classes=1000, l2_reg=0.,weights_path=None):):
	
	    dim_ordering = K.image_dim_ordering()
	    print dim_ordering
	    if dim_ordering == 'th':
		batch_index = 0
		channel_index = 1
		row_index = 2
		col_index = 3
	    if dim_ordering == 'tf':
		batch_index = 0
		channel_index = 3
		row_index = 1
		col_index = 2

	    inputs = Input(img_shape)

	    conv_1 = Convolution2D(96, 11, 11, subsample=(4, 4), activation='relu',
		                   name='conv_1', W_regularizer=l2(l2_reg))(inputs)

	    conv_1_mask = np.ones(shape=((96, 55, 55)))
	    conv_1_mask  = K.variable(conv_1_mask)
	    conv_1_lambda = Lambda(lambda x: x * conv_1_mask)(conv_1)

	    conv_2 = MaxPooling2D((3, 3), strides=(2, 2))(conv_1_lambda)
	    conv_2 = crosschannelnormalization(name="convpool_1")(conv_2)
	    conv_2 = ZeroPadding2D((2, 2))(conv_2)
	    conv_2 = merge([
		Convolution2D(128, 5, 5, activation="relu", name='conv_2_'+str(i+1),
		              W_regularizer=l2(l2_reg))(
		    splittensor(axis=channel_index, ratio_split=2, id_split=i)(conv_2)
		) for i in range(2)], mode='concat', concat_axis=channel_index, name="conv_2")

	    conv_2_mask = np.ones(shape=((256, 27, 27)))
	    conv_2_mask = K.variable(conv_2_mask)
	    conv_2_lambda = Lambda(lambda x: x * conv_2_mask)(conv_2)

	    conv_3 = MaxPooling2D((3, 3), strides=(2, 2))(conv_2_lambda)
	    conv_3 = crosschannelnormalization()(conv_3)
	    conv_3 = ZeroPadding2D((1, 1))(conv_3)
	    conv_3 = Convolution2D(384, 3, 3, activation='relu', name='conv_3',
		                   W_regularizer=l2(l2_reg))(conv_3)

	    conv_3_mask = np.ones(shape=((384, 13, 13)))
	    conv_3_mask = K.variable(conv_3_mask)
	    conv_3_lambda = Lambda(lambda x: x * conv_3_mask)(conv_3)

	    conv_4 = ZeroPadding2D((1, 1))(conv_3_lambda)
	    conv_4 = merge([
		Convolution2D(192, 3, 3, activation="relu", name='conv_4_'+str(i+1),
		              W_regularizer=l2(l2_reg))(
		    splittensor(axis=channel_index, ratio_split=2, id_split=i)(conv_4)
		) for i in range(2)], mode='concat', concat_axis=channel_index, name="conv_4")

	    conv_4_mask = np.ones(shape=((384, 13, 13)))
	    conv_4_mask = K.variable(conv_4_mask)
	    conv_4_lambda = Lambda(lambda x: x * conv_4_mask)(conv_4)

	    conv_5 = ZeroPadding2D((1, 1))(conv_4_lambda)
	    conv_5 = merge([
		Convolution2D(128, 3, 3, activation="relu", name='conv_5_'+str(i+1),
		              W_regularizer=l2(l2_reg))(
		    splittensor(axis=channel_index, ratio_split=2, id_split=i)(conv_5)
		) for i in range(2)], mode='concat', concat_axis=channel_index, name="conv_5")

	    conv_5_mask = np.ones(shape=((256, 13, 13)))
	    conv_5_mask = K.variable(conv_5_mask)
	    conv_5_lambda = Lambda(lambda x: x * conv_5_mask)(conv_5)

	    dense_1 = MaxPooling2D((3, 3), strides=(2, 2), name="convpool_5")(conv_5_lambda)

	    dense_1 = Flatten(name="flatten")(dense_1)
	    dense_1 = Dense(4096, activation='relu', name='dense_1',
		            W_regularizer=l2(l2_reg))(dense_1)

	    dense_1_mask = np.ones(shape=((4096,)))
	    dense_1_mask = K.variable(dense_1_mask)
	    dense_1_lambda = Lambda(lambda x: x * dense_1_mask)(dense_1)

	    dense_2 = Dropout(0.5)(dense_1_lambda)
	    dense_2 = Dense(4096, activation='relu', name='dense_2',
		            W_regularizer=l2(l2_reg))(dense_2)

	    dense_2_mask = np.ones(shape=((4096,)))
	    dense_2_mask = K.variable(dense_2_mask)
	    dense_2_lambda = Lambda(lambda x: x * dense_2_mask)(dense_2)

	    dense_3 = Dropout(0.5)(dense_2_lambda)
	    if n_classes == 1000:
		dense_3 = Dense(n_classes, name='dense_3',
		                W_regularizer=l2(l2_reg))(dense_3)
		dense_3_mask = np.ones(shape=((1000,)))
		dense_3_mask = K.variable(dense_3_mask)
		dense_3_lambda = Lambda(lambda x: x * dense_3_mask)(dense_3)
	    else:
		# We change the name so when loading the weights_file from a
		# Imagenet pretrained model does not crash
		dense_3 = Dense(n_classes, name='dense_3_new',
		                W_regularizer=l2(l2_reg))(dense_3)
		dense_3_mask = np.ones(shape=((1000,)))
		dense_3_mask = K.variable(dense_3_mask)
		dense_3_lambda = Lambda(lambda x: x * dense_3_mask)(dense_3)

	    prediction = Activation("softmax", name="softmax")(dense_3_lambda)

	    model = Model(input=inputs, output=prediction)
	    if weights_path:
		model.load_weights(weights_path)

	    return model
