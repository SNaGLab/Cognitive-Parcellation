def load_data():

	# Model parmeters and running the model from the loaded weights{
	model_name = 'VGG16'
	model = VGG16(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000)
	#KFold
	k = 4

	im_train, im_test = train_test_split(im_list, test_size=0.2, random_state=234)
	#}

	#Training data pkl
	fp_name = '../../data/pkl/'+str(p)+'_train_no_lesion_performance.txt'
	fp = open(fp_name,'a+')
	im_temp = preprocess_image_batch(im_train,img_size=(256,256), crop_size=(224,224), color_mode="rgb")
	out = model.predict(im_temp,batch_size=64)

	true_valid_wids = []
	for i in im_train:
		temp1 = i.split('/')[4]
		temp = temp1.split('.')[0].split('_')[2]
		true_valid_wids.append(truth[int(temp)][1])

	predicted_valid_wids = []
	for i in range(len(im_train)):
	    #print im_list[i], pprint_output(out[i]), true_wids[i]
	    predicted_valid_wids.append(pprint_output(out[i]))

	count, error  = top5accuracy(true_valid_wids, predicted_valid_wids)

	fp.write(str(p)+' '+str(count)+' '+str(len(im_train))+' '+str(error)+'\n')


	print(len(true_valid_wids), len(predicted_valid_wids), len(im_train))
	print(count, error)


	#}
	# Code snippet to get the activation values and saving information{
	data = np.array([])

	i = 0
	result ={}
	for layer in model.layers:
	    weights = layer.get_weights()
	    if len(weights) > 0:
		activations = get_activations(model,i,im_temp)
		if result.get(layer.name, None) is None:
		    result[layer.name] = activations[0]
		    temp = np.mean(activations[0], axis=0).ravel()
		    if layer.name != 'predictions':
		        print(layer.name,len(weights),len(activations), activations[0].shape, np.mean(activations[0], axis=0).shape, temp.shape)
		        data = np.append(data, temp)
	    i += 1

	fp.close()
	with open('../../data/pkl/'+str(p)+'_train_'+model_name+'.pkl', 'wb') as f:
	    pickle.dump(data, f)


	#Testing data pkl
	fp_name = '../../data/pkl/'+str(p)+'_test_no_lesion_performance.txt'
	fp = open(fp_name,'a+')

	im_temp = preprocess_image_batch(im_test,img_size=(256,256), crop_size=(224,224), color_mode="rgb")
	out = model.predict(im_temp,batch_size=64)

	true_valid_wids = []
	for i in im_test:
		temp1 = i.split('/')[4]
		temp = temp1.split('.')[0].split('_')[2]
		true_valid_wids.append(truth[int(temp)][1])

	predicted_valid_wids = []
	for i in range(len(im_test)):
	    #print im_list[i], pprint_output(out[i]), true_wids[i]
	    predicted_valid_wids.append(pprint_output(out[i]))

	count, error  = top5accuracy(true_valid_wids, predicted_valid_wids)

	fp.write(str(p)+' '+str(count)+' '+str(len(im_test))+' '+str(error)+'\n')


	print(len(true_valid_wids), len(predicted_valid_wids), len(im_test))
	print(count, error)


	#}
	# Code snippet to get the activation values and saving information{
	data = np.array([])

	i = 0
	result ={}
	for layer in model.layers:
	    weights = layer.get_weights()
	    if len(weights) > 0:
		activations = get_activations(model,i,im_temp)
		if result.get(layer.name, None) is None:
		    result[layer.name] = activations[0]
		    temp = np.mean(activations[0], axis=0).ravel()
		    if layer.name != 'predictions':
		        print(layer.name,len(weights),len(activations), activations[0].shape, np.mean(activations[0], axis=0).shape, temp.shape)
		        data = np.append(data, temp)
	    i += 1

	fp.close()
	with open('../../data/pkl/'+str(p)+'_test_'+model_name+'.pkl', 'wb') as f:
	    pickle.dump(data, f)


	out_r = []

	image_list_test = '../../data/pkl/'+p+'_image_list_test.txt'
	with open(image_list_test,'w+') as f:
	    for i in im_test:
		f.write(i+'\n')

	kf = KFold(n_splits= k)
	fold = 1
	fp_name = '../../data/pkl/'+str(p)+'_no_lesion_performance.txt'
	fp = open(fp_name,'a+')
	for train_index, valid_index in kf.split(im_train):
	    print("Starting Fold: ", fold)
	    im_valid_train = [im_train[i] for i in train_index] 
	    im_valid_test = [im_train[i] for i in valid_index]
	    
	    image_list_train = '../../data/pkl/'+p+'_image_list_train_fold_'+str(fold)+'.txt'
	    with open(image_list_train,'w+') as f:
		for i in im_valid_train:
		    f.write(i+'\n')
	    
	    image_list_valid = '../../data/pkl/'+p+'_image_list_valid_fold_'+str(fold)+'.txt'
	    with open(image_list_valid,'w+') as f:
		for i in im_valid_test:
		    f.write(i+'\n')
	    
	   
	    im_temp = preprocess_image_batch(im_valid_test,img_size=(256,256), crop_size=(224,224), color_mode="rgb")
	    out = model.predict(im_temp,batch_size=64)

	    true_valid_wids = []
	    for i in im_valid_test:
		    temp1 = i.split('/')[4]
		    temp = temp1.split('.')[0].split('_')[2]
		    true_valid_wids.append(truth[int(temp)][1])
	    
	    predicted_valid_wids = []
	    for i in range(len(im_valid_test)):
		#print im_list[i], pprint_output(out[i]), true_wids[i]
		predicted_valid_wids.append(pprint_output(out[i]))
		
	    count, error  = top5accuracy(true_valid_wids, predicted_valid_wids)
	    
	    fp.write(str(p)+' '+str(fold)+' '+str(count)+' '+str(len(im_valid_test))+' '+str(error)+'\n')

	    
	    print(len(true_valid_wids), len(predicted_valid_wids), len(im_valid_test))
	    print(count, error)
	    
	    
	    #}
	    # Code snippet to get the activation values and saving information{
	    data = np.array([])

	    i = 0
	    result ={}
	    for layer in model.layers:
		weights = layer.get_weights()
		if len(weights) > 0:
		    activations = get_activations(model,i,im_temp)
		    if result.get(layer.name, None) is None:
		        result[layer.name] = activations[0]
		        temp = np.mean(activations[0], axis=0).ravel()
		        if layer.name != 'predictions':
		            print(layer.name,len(weights),len(activations), activations[0].shape, np.mean(activations[0], axis=0).shape, temp.shape)
		            data = np.append(data, temp)
		i += 1
	    print(data.shape)
	    out_r.append(data)
	    fold += 1
	    
	fp.close()


	#Saving all the data into pkl files
	for i in range(k):
	    name = p+'_fold_'+str(i+1)+'_train_'+model_name
	    out_data = out_r[i]
	    with open('../../data/pkl/'+name+'.pkl', 'wb') as f:
		pickle.dump(out_data, f)
	    print(name, len(out_data))



def compare_layer_lesions():

layer_info = {}
#Comparing Layer lesions
classes = ['animate','inanimate']
%time
for label in [ana,ina]:
    layer_info[label] = {}
    for layer in [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]:
        if layer == 1:
            start = 0
            end = 3211264
        elif layer == 2:
            start = 3211264
            end = 6422528
        elif layer == 3:
            start = 6422528
            end = 8028160
        elif layer == 4:
            start = 8028160
            end = 9633792
        elif layer == 5:
            start = 9633792
            end = 10436608
        elif layer == 6:
            start = 10436608
            end = 11239424
        elif layer == 7:
            start = 11239424
            end = 12042240
        elif layer == 8:
            start = 12042240
            end = 12443648
        elif layer == 9:
            start = 12443648
            end = 12845056
        elif layer == 10:
            start = 12845056
            end = 13246464
        elif layer == 11:
            start = 13246464
            end = 13346816
        elif layer == 12:
            start = 13346816
            end = 13447168
        elif layer == 13:
            start = 13447168
            end = 13547520
        elif layer == 14:
            start = 13547520
            end = 13551616
        elif layer == 15:
            start = 13551616
            end = 13555712

        layer_info[label][layer] = {}
    
        #No lesion
        #print('No-lesioning')
        #print('Label:',label)
        #print('Layer:',layer)

        pred = clf.predict(X_new)
        lambda_mask = np.ones(shape=((13555712,)))
       

        model = VGG16(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000,lambda_mask=lambda_mask)

        flag = 0
        dprime = 0.
        for p in classes:
            im_valid_test = []
            image_list_valid = '../../data/pkl/'+p+'_image_list_test.txt'
            with open(image_list_valid,'r') as f:
                for line in f.readlines():
                    im_valid_test.append(line.strip('\n'))
            im_temp = preprocess_image_batch(im_valid_test,img_size=(256,256), crop_size=(224,224), color_mode="rgb")
            out = model.predict(im_temp,batch_size=64)

            true_valid_wids = []
            for i in im_valid_test:
                    temp1 = i.split('/')[4]
                    temp = temp1.split('.')[0].split('_')[2]
                    true_valid_wids.append(truth[int(temp)][1])

            predicted_valid_wids = []
            for i in range(len(im_valid_test)):
                #print im_list[i], pprint_output(out[i]), true_wids[i]
                predicted_valid_wids.append(pprint_output(out[i]))

            count, error  = top5accuracy(true_valid_wids, predicted_valid_wids)

            print(str(p)+' '+str(count)+' '+str(len(im_valid_test))+' '+str(error)+' '+str(1-error))
              
            if flag == 0:
                dprime = error
                flag = 1
            else:
                dprime -= error
        print('Layer: ',layer,'Label: ', label)
        print('No lesion: ',dprime)
        layer_info[label][layer]['no'] = dprime
        
        tf.keras.backend.clear_session()
        gc.collect()
        del model
        #Before lesion
        #print('Pre-layer-lesioning')
        #print('Label:',label)
        #print('Layer:',layer)

        pred = clf.predict(X_new)
        loc = np.where(pred==label)[0]
        loc_new =[]
        for i in range(len(loc)):
            temp = np.where(pred_kmeans==loc[i])[0]
            loc_new.extend(temp)

        lambda_mask = np.ones(shape=((13555712,)))
        lambda_mask[loc_new] = 0.
        print('pre-loc', len(loc_new))
        model = VGG16(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000,lambda_mask=lambda_mask)

        flag = 0
        dprime = 0.
        for p in classes:
            im_valid_test = []
            image_list_valid = '../../data/pkl/'+p+'_image_list_test.txt'
            with open(image_list_valid,'r') as f:
                for line in f.readlines():
                    im_valid_test.append(line.strip('\n'))
            im_temp = preprocess_image_batch(im_valid_test,img_size=(256,256), crop_size=(224,224), color_mode="rgb")
            out = model.predict(im_temp,batch_size=64)

            true_valid_wids = []
            for i in im_valid_test:
                    temp1 = i.split('/')[4]
                    temp = temp1.split('.')[0].split('_')[2]
                    true_valid_wids.append(truth[int(temp)][1])

            predicted_valid_wids = []
            for i in range(len(im_valid_test)):
                #print im_list[i], pprint_output(out[i]), true_wids[i]
                predicted_valid_wids.append(pprint_output(out[i]))

            count, error  = top5accuracy(true_valid_wids, predicted_valid_wids)

            print(str(p)+' '+str(count)+' '+str(len(im_valid_test))+' '+str(error)+' '+str(1-error))

            if flag == 0:
                dprime = error
                flag = 1
            else:
                dprime -= error
        print('Cluster Only: ',dprime)
        layer_info[label][layer]['pre'] = dprime   
        tf.keras.backend.clear_session()
        gc.collect()
        del model
             
            
        #After Lesion
       # print('Post-layer-lesioning')
        #print('Label:',label)
        #print('Layer:',layer)
        pred = clf.predict(X_new)
        loc = np.where(pred==label)[0]
        loc_new =[]
        for i in range(len(loc)):
            temp = np.where(pred_kmeans==loc[i])[0]
            temp2 = temp[np.asarray(np.where((temp >end) | (temp <=start))[0])]
            #print(len(temp), len(temp2))
            loc_new.extend(temp2)


        lambda_mask = np.ones(shape=((13555712,)))
        lambda_mask[loc_new] = 0.
        print('post-loc', len(loc_new))
        model = VGG16(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000,lambda_mask=lambda_mask)

        flag = 0
        dprime = 0.
        for p in classes:
            im_valid_test = []
            image_list_valid = '../../data/pkl/'+p+'_image_list_test.txt'
            with open(image_list_valid,'r') as f:
                for line in f.readlines():
                    im_valid_test.append(line.strip('\n'))
            im_temp = preprocess_image_batch(im_valid_test,img_size=(256,256), crop_size=(224,224), color_mode="rgb")
            out = model.predict(im_temp,batch_size=64)

            true_valid_wids = []
            for i in im_valid_test:
                    temp1 = i.split('/')[4]
                    temp = temp1.split('.')[0].split('_')[2]
                    true_valid_wids.append(truth[int(temp)][1])

            predicted_valid_wids = []
            for i in range(len(im_valid_test)):
                #print im_list[i], pprint_output(out[i]), true_wids[i]
                predicted_valid_wids.append(pprint_output(out[i]))

            count, error  = top5accuracy(true_valid_wids, predicted_valid_wids)

            print(str(p)+' '+str(count)+' '+str(len(im_valid_test))+' '+str(error)+' '+str(1-error))

            if flag == 0:
                dprime = error
                flag = 1
            else:
                dprime -= error
        print('Cluster - layer: ',dprime)
        layer_info[label][layer]['post'] = dprime
        
        
         #Random Lesion
       # print('Post-layer-lesioning')
        #print('Label:',label)
        #print('Layer:',layer)
        pred = clf.predict(X_new)
        loc = np.where(pred==label)[0]
        loc_new =[]
        for i in range(len(loc)):
            temp = np.where(pred_kmeans==loc[i])[0]
            temp2 = temp[np.asarray(np.where((temp >end) | (temp <=start))[0])]
            #print(len(temp), len(temp2))
            loc_new.extend(temp2)

        loc_new2 = np.random.randint(start,end,len(loc_new))
        lambda_mask = np.ones(shape=((13555712,)))
        lambda_mask[loc_new2] = 0.
        print('post-rand-loc', len(loc_new))
        model = VGG16(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000,lambda_mask=lambda_mask)

        flag = 0
        dprime = 0.
        for p in classes:
            im_valid_test = []
            image_list_valid = '../../data/pkl/'+p+'_image_list_test.txt'
            with open(image_list_valid,'r') as f:
                for line in f.readlines():
                    im_valid_test.append(line.strip('\n'))
            im_temp = preprocess_image_batch(im_valid_test,img_size=(256,256), crop_size=(224,224), color_mode="rgb")
            out = model.predict(im_temp,batch_size=64)

            true_valid_wids = []
            for i in im_valid_test:
                    temp1 = i.split('/')[4]
                    temp = temp1.split('.')[0].split('_')[2]
                    true_valid_wids.append(truth[int(temp)][1])

            predicted_valid_wids = []
            for i in range(len(im_valid_test)):
                #print im_list[i], pprint_output(out[i]), true_wids[i]
                predicted_valid_wids.append(pprint_output(out[i]))

            count, error  = top5accuracy(true_valid_wids, predicted_valid_wids)

            print(str(p)+' '+str(count)+' '+str(len(im_valid_test))+' '+str(error)+' '+str(1-error))

            if flag == 0:
                dprime = error
                flag = 1
            else:
                dprime -= error
        print('Random: ',dprime)
        layer_info[label][layer]['rand'] = dprime
        tf.keras.backend.clear_session()
        gc.collect()
        del model
       

from matplotlib.ticker import MaxNLocator
X = np.arange(1,16)
Y = []
Z =[]
tmp = 0.
for item in X:
    print(item,layer_info[ana][item]['pre'],layer_info[ana][item]['post'],layer_info[ana][item]['rand'])
    tmp += layer_info[ana][item]['pre']-layer_info[ana][item]['post']
    Y.append(layer_info[ana][item]['pre']-layer_info[ana][item]['post'])
    if layer_info[ana][item]['rand'] == 0.:
        Z.append(0.)
    else:
        Z.append(layer_info[ana][item]['rand']-layer_info[ana][item]['no'])
print(tmp)
width = 0.35  # the width of the bars

rects1 = plt.bar(X- width/2, Y, width, label='Layer-specific cluster lesions')
rects2 = plt.bar(X+width/2, Z, width, label='Random layer lesions')

plt.ylabel('Relative change in performance impact')
plt.xlabel('Different layers of Alexnet')
plt.xticks(X, ('conv_1_1', 'conv_1_2', 'conv_2_1','conv_2_2','conv_3_1','conv_3_2','conv_3_3','conv_4_1','conv_4_2','conv_4_3','conv_5_1','conv_5_2', 'conv_5_3','fc1','fc2'))
plt.title('Change in Performance by layer for Animate')
plt.legend()
plt.gca().xaxis.set_major_locator(MaxNLocator(prune='lower'))
plt.savefig('../../results/scree/animate_by_layer.png', format='png')
#plt.ylim([0,1])|


X = np.arange(1,16)
Y = []
Z = []
tmp = 0.
for item in X:
    print(item,layer_info[ina][item]['pre'],layer_info[ina][item]['post'],layer_info[ana][item]['rand'])
    Y.append((-1*layer_info[ina][item]['pre'])-(-1*layer_info[ina][item]['post']))
    if layer_info[ana][item]['rand'] == 0.:
        Z.append(0.)
    else:
        Z.append(layer_info[ana][item]['rand']-layer_info[ana][item]['no'])
    tmp += (-1*layer_info[ina][item]['pre'])-(-1*layer_info[ina][item]['post'])
print(tmp)
width = 0.35  # the width of the bars

rects1 = plt.bar(X- width/2, Y, width, label='Layer-specific cluster lesions')
rects2 = plt.bar(X+width/2, Z, width, label='Random layer lesions')

plt.ylabel('Relative change in performance impact')
plt.xticks(X, ('conv_1_1', 'conv_1_2', 'conv_2_1','conv_2_2','conv_3_1','conv_3_2','conv_3_3','conv_4_1','conv_4_2','conv_4_3','conv_5_1','conv_5_2', 'conv_5_3','fc1','fc2'))
plt.xlabel('Different layers of Alexnet')
plt.title('Change in Performance by layer for Inanimate')
plt.legend()
plt.gca().xaxis.set_major_locator(MaxNLocator(prune='lower'))
plt.savefig('../../results/scree/inaniamte_by_layer.png', format='png')
#plt.ylim([0,1])



def histogram():
label_loc = np.where(pred==ana)[0]
Z = []
for i in range(len(label_loc)):
    temp = np.where(pred_kmeans==label_loc[i])[0]
    for i in temp:# Create a ClassificationModel
        Z.append(i)
X = np.arange(15)
Y = np.zeros((15,))

for i in Z:
    if i in range(0,3211264): 
        Y[0] += 1
    elif i in range(3211264,6422528):
        Y[1] += 1
    elif i in range(6422528,8028160):
        Y[2] += 1
    elif i in range(8028160,9633792):
        Y[3] += 1
    elif i in range(9633792,10436608):
        Y[4] += 1
    elif i in range(10436608,11239424):
        Y[5] += 1
    elif i in range(11239424,12042240):
        Y[6] += 1
    elif i in range(12042240,12443648):
        Y[7] += 1
    elif i in range(12443648,12845056):
        Y[8] += 1
    elif i in range(12845056,13246464):
        Y[9] += 1
    elif i in range(13246464,13346816):
        Y[10] += 1
    elif i in range(13346816,13447168):
        Y[11] += 1
    elif i in range(13447168,13547520):
        Y[12] += 1
    elif i in range(13547520,13551616):
        Y[13] += 1
    elif i in range(13551616,13555712):
        Y[14] += 1
    else:
        print(i)

Y[0] = float(Y[0]) /3211264
Y[1] = float(Y[0]) / 3211264
Y[2] = float(Y[2]) / 1605632
Y[3] = float(Y[3]) / 1605632
Y[4] = float(Y[4]) / 802816
Y[5] = float(Y[5]) /802816
Y[6] = float(Y[6]) /802816
Y[7] = float(Y[7]) /401408
Y[8] = float(Y[8]) /401408
Y[9] = float(Y[9]) /401408
Y[10] = float(Y[10]) /100352
Y[11] = float(Y[11]) /100352
Y[12] = float(Y[12]) /100352
Y[13] = float(Y[13]) /4096
Y[14] = float(Y[14]) /4096

plt.ylim([0,1.])
rect = plt.bar(X,Y)
plt.xticks(X, ('conv_1_1', 'conv_1_2', 'conv_2_1','conv_2_2','conv_3_1','conv_3_2','conv_3_3','conv_4_1','conv_4_2','conv_4_3','conv_5_1','conv_5_2', 'conv_5_3','fc1','fc2'))
plt.ylabel('Relative count of neurons')
plt.title('Neurons from the animate cluster')
plt.gca().xaxis.set_major_locator(MaxNLocator(prune='lower'))
#autolabel(rect)
plt.savefig('../../results/scree/'+str(method)+'_results_ana_hist_alt.png', format='png')

label_loc = np.where(pred==ina)[0]
Z = []# Create a ClassificationModel
for i in range(len(label_loc)):
    temp = np.where(pred_kmeans==label_loc[i])[0]
    for i in temp:
        Z.append(i)
X = np.arange(15)
Y = np.zeros((15,))

for i in Z:
    if i in range(0,3211264): 
        Y[0] += 1
    elif i in range(3211264,6422528):
        Y[1] += 1
    elif i in range(6422528,8028160):
        Y[2] += 1
    elif i in range(8028160,9633792):
        Y[3] += 1
    elif i in range(9633792,10436608):
        Y[4] += 1
    elif i in range(10436608,11239424):
        Y[5] += 1
    elif i in range(11239424,12042240):
        Y[6] += 1
    elif i in range(12042240,12443648):
        Y[7] += 1
    elif i in range(12443648,12845056):
        Y[8] += 1
    elif i in range(12845056,13246464):
        Y[9] += 1
    elif i in range(13246464,13346816):
        Y[10] += 1
    elif i in range(13346816,13447168):
        Y[11] += 1
    elif i in range(13447168,13547520):
        Y[12] += 1
    elif i in range(13547520,13551616):
        Y[13] += 1
    elif i in range(13551616,13555712):
        Y[14] += 1
    else:
        print(i)

Y[0] = float(Y[0]) /3211264
Y[1] = float(Y[0]) / 3211264
Y[2] = float(Y[2]) / 1605632
Y[3] = float(Y[3]) / 1605632
Y[4] = float(Y[4]) / 802816
Y[5] = float(Y[5]) /802816
Y[6] = float(Y[6]) /802816
Y[7] = float(Y[7]) /401408
Y[8] = float(Y[8]) /401408
Y[9] = float(Y[9]) /401408
Y[10] = float(Y[10]) /100352
Y[11] = float(Y[11]) /100352
Y[12] = float(Y[12]) /100352
Y[13] = float(Y[13]) /4096
Y[14] = float(Y[14]) /4096

plt.ylim([0,1.])
rect = plt.bar(X,Y)
plt.ylabel('Relative count of neurons')
plt.title('Neurons from the inanimate cluster')
plt.xticks(X, ('conv_1_1', 'conv_1_2', 'conv_2_1','conv_2_2','conv_3_1','conv_3_2','conv_3_3','conv_4_1','conv_4_2','conv_4_3','conv_5_1','conv_5_2', 'conv_5_3','fc1','fc2'))
plt.gca().xaxis.set_major_locator(MaxNLocator(prune='lower'))
#autolabel(rect)
plt.savefig('../../results/scree/'+str(method)+'_results_ina_hist_alt.png', format='png')

