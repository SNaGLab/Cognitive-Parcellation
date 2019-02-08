from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
import numpy as np
import keras
import os
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten,Conv2D, MaxPooling2D, Input, ZeroPadding2D,merge
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.utils.layer_utils import convert_all_kernels_in_model
from keras.optimizers import SGD
from scipy.misc import imread
from scipy.misc import imresize
from keras import backend as K
from keras.engine import Layer
from keras.layers.core import Lambda
import pickle
from keras.utils.vis_utils import plot_model
import matplotlib
import matplotlib.pyplot as plt


from keras import backend as K
def get_activations(model, layer, X_batch):
    get_activations = K.function([model.layers[0].input, K.learning_phase()], [model.layers[layer].output,])
    activations = get_activations([X_batch,0])
    return activations

def preprocess_image_batch(image_paths, img_size=None, crop_size=None, color_mode='rgb', out=None):
    """
    Consistent preprocessing of images batches
    :param image_paths: iterable: images to process
    :param crop_size: tuple: crop images if specified
    :param img_size: tuple: resize images if specified
    :param color_mode: Use rgb or change to bgr mode based on type of model you want to use
    :param out: append output to this iterable if specified
    """
    img_list = []

    for im_path in image_paths:
        img = imread(im_path, mode='RGB')
	#print im_path
	#print img.shape
        if img_size:
            img = imresize(img, img_size)
	
        img = img.astype('float32')
        # We normalize the colors (in RGB space) with the empirical means on the training set
        img[:, :, 0] -= 123.68
        img[:, :, 1] -= 116.779
        img[:, :, 2] -= 103.939
        # We permute the colors to get them in the BGR order
        if color_mode == 'bgr':
            img[:, :, [0, 1, 2]] = img[:, :, [2, 1, 0]]
        img = img.transpose((2, 0, 1))

        if crop_size:
            img = img[:, (img_size[0] - crop_size[0]) // 2:(img_size[0] + crop_size[0]) // 2
            , (img_size[1] - crop_size[1]) // 2:(img_size[1] + crop_size[1]) // 2]

        img_list.append(img)

    try:
        img_batch = np.stack(img_list, axis=0)
    except:
	print im_path
        raise ValueError('when img_size and crop_size are None, images'
                ' in image_paths must have the same shapes.')

    if out is not None and hasattr(out, 'append'):
        out.append(img_batch)
    else:
        return img_batch

from os.path import dirname
from os.path import join

from scipy.io import loadmat

meta_clsloc_file = join(dirname(__file__), '../../data', 'meta_clsloc.mat')

synsets = loadmat(meta_clsloc_file)['synsets'][0]

synsets_imagenet_sorted = sorted([(int(s[0]), str(s[1][0])) for s in synsets[:1000]],key=lambda v: v[1])

corr = {}
for j in range(1000):
    corr[synsets_imagenet_sorted[j][0]] = j

corr_inv = {}
for j in range(1, 1001):
    corr_inv[corr[j]] = j

def id_to_words(id_):
    return synsets[corr_inv[id_] - 1][2][0]


def pprint_output(out, n_max_synsets=10):
    wids = []
    best_ids = out.argsort()[::-1][:10]
    for u in best_ids:
	wids.append(str(synsets[corr_inv[u] - 1][1][0]))
        #print('%.2f' % round(100 * out[u], 2) + ' : ' + id_to_words(u)+' '+ str(synsets[corr_inv[u] - 1][1][0]))
    return wids

truth = {}
with open('../../data/ILSVRC2014_clsloc_validation_ground_truth.txt') as f:
	line_num = 1
	for line in f.readlines():
		ind_ = int(line)
		temp  = None
		for i in synsets_imagenet_sorted:
			if i[0] == ind_:
				temp = i
		if temp != None:		
			truth[line_num] = temp

		else:
			print '##########', ind_
		line_num += 1


model = VGG16(weights='imagenet')

out_r = []
p_num = 1
for p in ['animate','inanimate']:
	
	url_path = '../../data/'+p+'/'

	true_wids = []
	im_list = []
	for i in os.listdir(url_path):
		temp = i.split('.')[0].split('_')[2]
		true_wids.append(truth[int(temp)][1])
		im_list.append(url_path+i)


	im = preprocess_image_batch(im_list,img_size=(256,256), crop_size=(224,224), color_mode="rgb")

	out = model.predict(im,batch_size=64)

	def top5accuracy(true, predicted):
		assert len(true) == len(predicted)
		result = []	
		flag  = 0
		for i in range(len(true)): 	
			flag  = 0
			temp = true[i]
			for j in predicted[i]:
				if j == temp:
					flag = 1
					break
			if flag == 1:
				result.append(1)
			else:
				result.append(0)
		counter = 0.		
		for i in result:
			if i == 1:
			 counter += 1.
		error = 1.0 - counter/float(len(result))
		print len(np.where(np.asarray(result) == 1)[0])
		return error
	




	predicted_wids = []
	for i in range(len(im_list)):
		#print im_list[i], pprint_output(out[i]), true_wids[i]
		predicted_wids.append(pprint_output(out[i]))

	print len(true_wids), len(predicted_wids), len(im_list)
	print top5accuracy(true_wids, predicted_wids)


	data = np.array([])
	i = 0
	result ={}
	for layer in model.layers:
	    weights = layer.get_weights()
	    if len(weights) > 0:
		activations = get_activations(model,i,im)
		if result.get(layer.name, None) is None:
			result[layer.name] = activations[0]
	
		temp = np.mean(activations[0], axis=0).ravel()	
	    	print layer.name,len(weights),len(activations), activations[0].shape, np.mean(activations[0], axis=0).shape, temp.shape
		data = np.append(data, temp)

	    i += 1 
	print data.shape
	out_r.append(data)
	plt.figure(p_num)
	plt.hist(data,bins='auto',color = 'blue')
	plt.title('Histogram of '+p+ ' class')
	plt.savefig('../../results/histograms/'+p+'_hist_vgg16.png')
	p_num += 1
with open('../../data/data_vgg16.pkl', 'w') as f:
	 pickle.dump(out_r, f)




plt.show() 

