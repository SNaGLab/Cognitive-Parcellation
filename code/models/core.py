def  load_data(p):
	p = 'animate'
	url_path = '../../data/'+p+'/'

	# Prepare the image list and pre-process them{
	true_wids = []
	im_list = []
	for i in os.listdir(url_path):
	    if not i.startswith('~') and not i.startswith('.'):
		#print i, truth
		temp = i.split('.')[0].split('_')[2]
		true_wids.append(truth[int(temp)][1])
		im_list.append(url_path+i)

	im = preprocess_image_batch(im_list,img_size=(256,256), crop_size=(227,227), color_mode="rgb")
	#}


def generate_train_test_pickles():
	# Model parmeters and running the model from the loaded weights{

	out_r = []

	sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
	model = AlexNet(weights_path="../../data/weights/alexnet_weights.h5")
	model.compile(optimizer=sgd, loss='mse')
	#print model.summary()
	#plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)


	#KFold
	k = 4

	im_train, im_test = train_test_split(im_list, test_size=0.2, random_state=234)
	#}
	#Training data pkl
	fp_name = '../../data/pkl/'+str(p)+'_train_no_lesion_performance.txt'
	fp = open(fp_name,'a+')
	im_temp = preprocess_image_batch(im_train,img_size=(256,256), crop_size=(227,227), color_mode="rgb")
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


	print len(true_valid_wids), len(predicted_valid_wids), len(im_train)
	print count, error


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
		    if layer.name != 'dense_3':
		        print layer.name,len(weights),len(activations), activations[0].shape, np.mean(activations[0], axis=0).shape, temp.shape
		        data = np.append(data, temp)
	    i += 1

	fp.close()
	with open('../../data/pkl/'+str(p)+'_train.pkl', 'wb') as f:
	    pickle.dump(data, f)
	#Testing data pkl
	fp_name = '../../data/pkl/'+str(p)+'_test_no_lesion_performance.txt'
	fp = open(fp_name,'a+')

	im_temp = preprocess_image_batch(im_test,img_size=(256,256), crop_size=(227,227), color_mode="rgb")
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


	print len(true_valid_wids), len(predicted_valid_wids), len(im_test)
	print count, error


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
		    if layer.name != 'dense_3':
		        print layer.name,len(weights),len(activations), activations[0].shape, np.mean(activations[0], axis=0).shape, temp.shape
		        data = np.append(data, temp)
	    i += 1

	fp.close()
	with open('../../data/pkl/'+str(p)+'_test.pkl', 'wb') as f:
	    pickle.dump(data, f)

	image_list_test = '../../data/pkl/'+p+'_image_list_test.txt'
	with open(image_list_test,'wb') as f:
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
	    with open(image_list_train,'wb') as f:
		for i in im_valid_train:
		    f.write(i+'\n')
	    
	    image_list_valid = '../../data/pkl/'+p+'_image_list_valid_fold_'+str(fold)+'.txt'
	    with open(image_list_valid,'wb') as f:
		for i in im_valid_test:
		    f.write(i+'\n')
	    
	   
	    im_temp = preprocess_image_batch(im_valid_test,img_size=(256,256), crop_size=(227,227), color_mode="rgb")
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

	    
	    print len(true_valid_wids), len(predicted_valid_wids), len(im_valid_test)
	    print count, error
	    
	    
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
		        if layer.name != 'dense_3':
		            print layer.name,len(weights),len(activations), activations[0].shape, np.mean(activations[0], axis=0).shape, temp.shape
		            data = np.append(data, temp)
		i += 1
	    print data.shape
	    out_r.append(data)
	    fold += 1
	    
	fp.close()

	#Saving all the data into pkl files
	for i in range(k):
	    name = p+'_fold_'+str(i+1)+'_train'
	    out_data = out_r[i]
	    with open('../../data/pkl/'+name+'.pkl', 'wb') as f:
		pickle.dump(out_data, f)
	    print(name, len(out_data))


def first_level_clustering():
	#Test Cell 1
	from sklearn.cluster import KMeans
	from sklearn.cluster import MiniBatchKMeans

	data_path = '../../data/pkl/'
	classes = ['animate','inanimate']
	fold = 1

	with open(data_path+classes[0]+'_train.pkl') as f:
		X_fold = pickle.load(f)
	with open(data_path+classes[1]+'_train.pkl') as f:
		y_fold = pickle.load(f)
	    
	X = np.column_stack((X_fold,y_fold))
	X = np.float32(X)

	kmeans = MiniBatchKMeans(n_clusters=65827,
		                 batch_size=6,
		                 max_iter=10).fit(X)
	kmeans.cluster_centers_
	pred_kmeans = kmeans.predict(X)
	X_new = kmeans.cluster_centers_

	with open('../../data/pkl/kmeans_first_train.pickle', 'wb') as handle:
	    pickle.dump([X_new,pred_kmeans,kmeans], handle, protocol=pickle.HIGHEST_PROTOCOL)


def second_level_clustering():
	#Version 1 - Reading pkl files from step 0 and clustering it{
	data_path = '../../data/pkl/'
	classes = ['animate','inanimate']


	result= {}

	k = 4 #Total Number of folds
	fold = 1

	for i in range(k):
	    
	    print('Perfoming Fold: ', fold)
	    clf_result = {}
	    
	    if os.path.exists('../../data/pkl/kmeans_first_'+str(fold)+'.pickle'):
		with open('../../data/pkl/kmeans_first_'+str(fold)+'.pickle',"rb") as f:
		    X_new,pred_kmeans,kmeans = pickle.load(f)
	    else:   
		with open(data_path+classes[0]+'_fold_'+str(fold)+'_train.pkl') as f:
		    X_fold = pickle.load(f)
		with open(data_path+classes[1]+'_fold_'+str(fold)+'_train.pkl') as f:
		    y_fold = pickle.load(f)

		X = np.column_stack((X_fold,y_fold))
		kmeans = MiniBatchKMeans(n_clusters=65827,
		                         random_state=0,
		                         batch_size=6,
		                         max_iter=10).fit(X)
		#print kmeans.cluster_centers_
		pred_kmeans = kmeans.predict(X)
		X_new = kmeans.cluster_centers_

		with open('../../data/pkl/kmeans_first_'+str(fold)+'.pickle', 'wb') as handle:
		    pickle.dump([X_new,pred_kmeans,kmeans], handle, protocol=pickle.HIGHEST_PROTOCOL)

	    #DO CLUSTERING AND GET CLUSTERS
	    
	    from sklearn.cluster import KMeans
	    from sklearn.cluster import SpectralClustering
	    from sklearn.cluster import AgglomerativeClustering
	    from sklearn.cluster import Birch
	    from sklearn.mixture import GaussianMixture
	    
	    method ='GMM'
	    print(method)
	    for j in range(1,51,1):
	 
		clf_result[j] = {}
	       
		print j

		clf = KMeans(n_clusters=j)    
		#clf =  AgglomerativeClustering(n_clusters=j, linkage='ward')
		#clf = Birch(branching_factor=50, n_clusters=j, threshold=0.5,compute_labels=True)
		clf = GaussianMixture(n_components=j, covariance_type='full')
		y_pred = clf.fit_predict(X_new)
		#print clf.cluster_centers_

		for label in set(y_pred):
		    print('Cluster: ',j,'Label: ', label)
		    
		    #Lesioning and measuring performance
		    pred = y_pred.copy()
		    loc = np.where(pred==label)
		    loc_temp = kmeans.predict(X_new[loc[0]])
		    loc_new =[]
		    for entry in set(loc_temp):
		        temp = np.where(pred_kmeans==entry)[0]
		        loc_new.extend(temp)

		    lambda_mask = np.ones(shape=((658272,)))
		    lambda_mask[loc_new] = 0.

		    #plt.scatter(X[:,0],X[:,1], c=y_pred) 

		    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
		    model = AlexNet(weights_path="../../data/weights/alexnet_weights.h5",lambda_mask=lambda_mask)
		    model.compile(optimizer=sgd, loss='mse')
		    
		    flag = 0
		    dprime = 0.
		    for p in classes:
		        im_valid_test = []
		        image_list_valid = '../../data/pkl/'+p+'_image_list_valid_fold_'+str(fold)+'.txt'
		        with open(image_list_valid,'rb') as f:
		            for line in f.readlines():
		                im_valid_test.append(line.strip('\n'))
		        im_temp = preprocess_image_batch(im_valid_test,img_size=(256,256), crop_size=(227,227), color_mode="rgb")
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

		        print str(p)+' '+str(fold)+' '+str(count)+' '+str(len(im_valid_test))+' '+str(error)
		        
		        if flag == 0:
		            dprime = error
		            flag = 1
		        else:
		            dprime -= error
		            
		    clf_result[j][label] = dprime
	    
	    with open('../../data/pkl/'+str(method)+'_50_scree_fold_'+str(fold)+'.pickle', 'wb') as handle:
		pickle.dump(clf_result, handle, protocol=pickle.HIGHEST_PROTOCOL)
	    
	    result[fold] = clf_result
	    fold += 1
	#}


def generate_scree_plots():
	#Loading the pickle files
	method ='KMeans'

	k = 4
	result ={}
	for i in range(1,k+1,1):
	    name = '../../data/pkl/'+str(method)+'_50_scree_fold_'+str(i)+'.pickle'   #CHANGE
	    with open(name,"rb") as f:
		result[i] = pickle.load(f)
	f = 1
	clf_result = result[f]


	fig = plt.figure(1)
	#X = range(1,51,1)
	X = range(2,51,1)
	for cl in X:
	    i = 0
	    for item in clf_result[cl].keys():
		plt.plot(cl,clf_result[cl][item],'ro')
		i += 1
		
	plt.xticks(X)
	plt.xlabel('Number of cluster(s) k')
	plt.ylabel("Performance Impact(Animate vs Inanimate)")
	plt.title('Scree Plot for fold '+ str(f))
	plt.gca().xaxis.set_major_locator(MaxNLocator(prune='lower'))
	plt.savefig('../../results/scree/'+str(method)+'_results_fold_'+str(f)+'.png', format='png', dpi=200)


	f = 2
	clf_result = result[f]


	plt.figure(1)
	#X = range(1,51,1)
	X = range(2,51,1)
	for cl in X:
	    i = 0
	    for item in clf_result[cl].keys():
		plt.plot(cl,clf_result[cl][item],'ro')
		i += 1
		
	plt.xticks(X)
	plt.xlabel('Number of cluster(s) k')
	plt.ylabel("Performance - d' (Animate vs Inanimate)")
	plt.title('Scree Plot for fold '+ str(f))
	plt.gca().xaxis.set_major_locator(MaxNLocator(prune='lower'))
	plt.savefig('../../results/scree/'+str(method)+'_results_fold_'+str(f)+'.svg', format='svg', dpi=1200)


	f = 3
	clf_result = result[f]


	plt.figure(1)
	#X = range(1,51,1)
	X = range(2,51,1)
	for cl in X:
	    i = 0
	    for item in clf_result[cl].keys():
		plt.plot(cl,clf_result[cl][item],'ro')
		i += 1
		
	plt.xticks(X)
	plt.xlabel('Number of cluster(s) k')
	plt.ylabel("Performance - d' (Animate vs Inanimate)")
	plt.title('Scree Plot for fold '+ str(f))
	plt.gca().xaxis.set_major_locator(MaxNLocator(prune='lower'))
	plt.savefig('../../results/scree/'+str(method)+'_results_fold_'+str(f)+'.svg', format='svg', dpi=1200)



	f = 4
	clf_result = result[f]


	plt.figure(1)
	#X = range(1,51,1)
	X = range(2,51,1)
	for cl in X:
	    i = 0
	    for item in clf_result[cl].keys():
		plt.plot(cl,clf_result[cl][item],'ro')
		i += 1
		
	plt.xticks(X)
	plt.xlabel('Number of cluster(s) k')
	plt.ylabel("Performance - d' (Animate vs Inanimate)")
	plt.title('Scree Plot for fold '+ str(f))
	plt.gca().xaxis.set_major_locator(MaxNLocator(prune='lower'))
	plt.savefig('../../results/scree/'+str(method)+'_results_fold_'+str(f)+'.svg', format='svg', dpi=1200)

	#Find MaxAd', MaxId' and its average
	plt.figure()
	noc = 4
	for i in range(1,noc+1,1):
	    X = []
	    Y = []
	    for j in range(2,51,1):
		X.append(j)
		temp = []
		for key, value in result[i][j].iteritems():
		    temp.append(value)
		maxa = max(temp)
		maxi = min(temp)
		avg = float(maxa - maxi)
		Y.append(avg)
	    #print X,Y
	    plt.plot(X,Y)


	#Smooth average graph

	noc = 4
	flag = 0
	X = range(2,51,1)
	an_fold =[]
	ian_fold = []
	Y = []
	for i in range(1,noc+1,1):
	    if i == 2:
		flag = 1
	    for j in range(2,51,1):
		temp = []
		for key, value in result[i][j].iteritems():
		    temp.append(value)
		maxa = max(temp)
		maxi = min(temp)
		if flag == 0:
		    an_fold.append(maxa)
		    ian_fold.append(maxi)
		else:
		    an_fold[j-2] += maxa
		    ian_fold[j-2] = maxi

	for j in range(2,51,1):
	    maxa = (an_fold[j-2]) / 4.
	    maxi = (ian_fold[j-2]) /4.
	    diff = maxa - maxi
	    Y.append(diff)
	    
	x_sm = np.array(X)
	y_sm = np.array(Y)

	x_smooth = np.linspace(x_sm.min(), x_sm.max(), 200)
	y_smooth = spline(X, Y, x_smooth)

	plt.plot(x_smooth, y_smooth, 'r', linewidth=1)
	plt.plot(Y.index(max(Y))+1,max(Y),'o')
	plt.xlabel('Number of cluster(s) k')
	plt.ylabel("Average Performance")
	plt.savefig('../../results/scree/'+str(method)+'_results_fold_avg.png', format='png', dpi=200)
	print max(Y), Y.index(max(Y)) + 1



def evaluate():

	#Testing on test data{
	data_path = '../../data/pkl/'
	classes = ['animate','inanimate']

	result = {}

	with open(data_path+classes[0]+'_test.pkl') as f:
	    X_fold = pickle.load(f)
	with open(data_path+classes[1]+'_test.pkl') as f:
	    y_fold = pickle.load(f)

	X = np.column_stack((X_fold,y_fold))  
	if os.path.exists('../../data/pkl/kmeans_first_test.pickle'):
	    with open('../../data/pkl/kmeans_first_test.pickle',"rb") as f:
		X_new,pred_kmeans,kmeans = pickle.load(f)
	else:   
	   
	    kmeans = MiniBatchKMeans(n_clusters=65827,
		                     random_state=0,
		                     batch_size=6,
		                     max_iter=10).fit(X)
	    #print kmeans.cluster_centers_
	    pred_kmeans = kmeans.predict(X)
	    X_new = kmeans.cluster_centers_


	#DO CLUSTERING AND GET CLUSTERS

	from sklearn.cluster import KMeans
	from sklearn.cluster import SpectralClustering
	from sklearn.cluster import AgglomerativeClustering
	from sklearn.cluster import Birch
	from sklearn.cluster import DBSCAN
	from sklearn.mixture import GaussianMixture
	import genieclust
	import hdbscan
	import smm

	j = 21 #Set this value from scree plot!
	method = 'SMM'
	print j
	#clf = hdbscan.HDBSCAN(min_cluster_size=j, gen_min_span_tree=True)
	#clf = DBSCAN(eps=5.443)
	#clf = KMeans(n_clusters=j,random_state=143)
	#clf= SpectralClustering(n_clusters=j,random_state=143)
	#clf =  AgglomerativeClustering(n_clusters=j, linkage='ward')
	#clf = Birch(branching_factor=50, n_clusters=j, threshold=0.5,compute_labels=True)
	clf = GaussianMixture(n_components=j, covariance_type='full',random_state=143)
	#clf= genieclust.genie.Genie(n_clusters=j)
	clf= smm.SMM(n_components=j, covariance_type='full', random_state=143, tol=1e-12,min_covar=1e-6, n_iter=1000, n_init=1, params='wmcd', init_params='wmcd')
	temp = clf.fit(X_new)
	y_pred = clf.predict(X_new)
	#y_pred = clf.fit_predict(X_new)
	print set(y_pred)
	#Z = clf.predict(X)

	for label in set(y_pred):
	    print('Cluster: ',j,'Label: ', label)

	    #Lesioning and measuring performance
	    #pred = clf.fit_predict(X_new)
	    temp = clf.fit(X_new)
	    pred = clf.predict(X_new)
	    loc = np.where(pred==label)
	    loc_temp = kmeans.predict(X_new[loc[0]])
	    loc_new =[]
	    for entry in set(loc_temp):
		temp = np.where(pred_kmeans==entry)[0]
		loc_new.extend(temp)

	    lambda_mask = np.ones(shape=((658272,)))
	    lambda_mask[loc_new] = 0.

	    #plt.scatter(X[:,0],X[:,1], c=y_pred) 

	    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
	    model = AlexNet(weights_path="../../data/weights/alexnet_weights.h5",lambda_mask=lambda_mask)
	    model.compile(optimizer=sgd, loss='mse')

	    flag = 0
	    dprime = 0.
	    for p in classes:
		im_valid_test = []
		image_list_valid = '../../data/pkl/'+p+'_image_list_test.txt'
		with open(image_list_valid,'rb') as f:
		    for line in f.readlines():
		        im_valid_test.append(line.strip('\n'))
		im_temp = preprocess_image_batch(im_valid_test,img_size=(256,256), crop_size=(227,227), color_mode="rgb")
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

		print str(p)+' '+str(count)+' '+str(len(im_valid_test))+' '+str(error)+' '+str(1-error)

		if flag == 0:
		    dprime = error
		    flag = 1
		else:
		    dprime -= error

	    result[label] = dprime

	z_temp = []

	for item in y_pred:
	    z_temp.append(result[item])
	print(len(z_temp),len(X_new))
	loc_z = kmeans.predict(X_new)
	z = np.ones(shape=((658272,)))
	for i in range(len(loc_z)):
	    temp = np.where(pred_kmeans==loc_z[i])[0]
	    z[temp] = z_temp[i]
	x = X[:,0]
	y = X[:,1]


	#Density Plot for Animate/Inanimate

	print(x.shape,y.shape,z.shape)
	fig, ax = plt.subplots()
	cs = ax.scatter(x, y, c=z, s=10,cmap='coolwarm')
	cbar = fig.colorbar(cs)
	lims = [
	    np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
	    np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
	]
	ax.plot(lims, lims, 'k-', alpha=0.75)
	plt.xlabel('Animate')
	plt.ylabel('Inanimate')
	plt.title('Performance Impact - Density plot')
	plt.savefig('../../results/scree/'+str(method)+'_results_density.png', format='png',dpi=200)


	print result.values().index(max(result.values())), result.values().index(min(result.values()))
	ana = int(result.values().index(max(result.values())))
	ina = int(result.values().index(min(result.values())))
	print result[ana], -1*(result[ina])


def class_analysis():
	fig, ax = plt.subplots()
	label_loc = np.where(pred==ana)[0]
	#print len(label_loc)
	Zx = []
	Zy = []
	for i in range(len(label_loc)):
	    temp = np.where(pred_kmeans==label_loc[i])[0]
	    for i in temp:
		Zx.append(X[i][0])
		Zy.append(X[i][1])
	#print len(np.where(Z!=0)[0])
	#print Z
	sc = plt.scatter(Zx,Zy) 
	plt.xlim([-2.2,300])
	plt.ylim([-2.2,300])
	lims = [
	    np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
	    np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
	]
	​
	# now plot both limits against eachother
	ax.plot(lims, lims, 'k-', alpha=0.75, zorder=1)
	plt.xlabel('Animate')
	plt.ylabel('Inanimate')
	plt.title('Most selective cluster for animate class')
	plt.savefig('../../results/scree/'+str(method)+'_results_ana.png', format='png')


	fig, ax = plt.subplots()
	label_loc = np.where(pred==ina)[0]
	print len(label_loc)
	Zx = []
	Zy = []
	for i in range(len(label_loc)):
	    temp = np.where(pred_kmeans==label_loc[i])[0]
	    for i in temp:
		Zx.append(X[i][0])
		Zy.append(X[i][1])
	sc = plt.scatter(Zx,Zy) 
	plt.xlim([-2.2,300])
	plt.ylim([-2.2,300])
	lims = [
	    np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
	    np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
	]

	# now plot both limits against eachother
	ax.plot(lims, lims, 'k-', alpha=0.75, zorder=1)
	plt.xlabel('Animate')
	plt.ylabel('Inanimate')
	plt.title('Most selective cluster for inanimate class')
	plt.savefig('../../results/scree/'+str(method)+'_results_ina.png', format='png')


def histogram():
	label_loc = np.where(pred==ana)[0]
	Z = []
	for i in range(len(label_loc)):
	    temp = np.where(pred_kmeans==label_loc[i])[0]
	    for i in temp:
		Z.append(i)
	X = np.arange(7)
	Y = np.zeros((7,))

	for i in Z:
	    if i in range(0,290400): 
		Y[0] += 1
	    elif i in range(290400,477024):
		Y[1] += 1
	    elif i in range(477024,541920):
		Y[2] += 1
	    elif i in range(541920,606816):
		Y[3] += 1
	    elif i in range(606816,650080):
		Y[4] += 1
	    elif i in range(650080,654176):
		Y[5] += 1
	    elif i in range(654176,658272):
		Y[6] += 1
	    else:
		print i

	Y[0] = float(Y[0]) /290400
	Y[1] = float(Y[0]) / 196624
	Y[2] = float(Y[2]) / 64896
	Y[3] = float(Y[3]) / 64896
	Y[4] = float(Y[4]) /43264
	Y[5] = float(Y[5]) /4096
	Y[6] = float(Y[6]) /4096

	plt.ylim([0,1.])
	rect = plt.bar(X,Y)
	plt.xticks(X, ('conv_1', 'conv_2', 'conv_3', 'conv_4','conv_5','fc1','fc2'))
	plt.ylabel('Relative count of neurons')
	plt.title('Neurons from the animate cluster')
	#autolabel(rect)
	plt.savefig('../../results/scree/'+str(method)+'_results_ana_hist_alt.png', format='png')


	label_loc = np.where(pred==ina)[0]
	Z = []
	for i in range(len(label_loc)):
	    temp = np.where(pred_kmeans==label_loc[i])[0]
	    for i in temp:
		Z.append(i)
	X = np.arange(7)
	Y = np.zeros((7,))

	for i in Z:
	    if i in range(0,290400): 
		Y[0] += 1
	    elif i in range(290400,477024):
		Y[1] += 1
	    elif i in range(477024,541920):
		Y[2] += 1
	    elif i in range(541920,606816):
		Y[3] += 1
	    elif i in range(606816,650080):
		Y[4] += 1
	    elif i in range(650080,654176):
		Y[5] += 1
	    elif i in range(654176,658272):
		Y[6] += 1
	    else:
		print i

	Y[0] = float(Y[0]) /290400
	Y[1] = float(Y[0]) / 196624
	Y[2] = float(Y[2]) / 64896
	Y[3] = float(Y[3]) / 64896
	Y[4] = float(Y[4]) /43264
	Y[5] = float(Y[5]) /4096
	Y[6] = float(Y[6]) /4096

	plt.ylim([0,1.])
	rect = plt.bar(X,Y)
	plt.ylabel('Relative count of neurons')
	plt.title('Neurons from the inanimate cluster')
	plt.xticks(X, ('conv_1', 'conv_2', 'conv_3', 'conv_4','conv_5','fc1','fc2'))
	#autolabel(rect)
	plt.savefig('../../results/scree/'+str(method)+'_results_ina_hist_alt.png', format='png')


def layerwise_lesions():
	label_loc = np.where(pred==ina)[0]
	Z = []
	for i in range(len(label_loc)):
	    temp = np.where(pred_kmeans==label_loc[i])[0]
	    for i in temp:
		Z.append(i)
	X = np.arange(7)
	Y = np.zeros((7,))

	for i in Z:
	    if i in range(0,290400): 
		Y[0] += 1
	    elif i in range(290400,477024):
		Y[1] += 1
	    elif i in range(477024,541920):
		Y[2] += 1
	    elif i in range(541920,606816):
		Y[3] += 1
	    elif i in range(606816,650080):
		Y[4] += 1
	    elif i in range(650080,654176):
		Y[5] += 1
	    elif i in range(654176,658272):
		Y[6] += 1
	    else:
		print i

	Y[0] = float(Y[0]) /290400
	Y[1] = float(Y[0]) / 196624
	Y[2] = float(Y[2]) / 64896
	Y[3] = float(Y[3]) / 64896
	Y[4] = float(Y[4]) /43264
	Y[5] = float(Y[5]) /4096
	Y[6] = float(Y[6]) /4096

	plt.ylim([0,1.])
	rect = plt.bar(X,Y)
	plt.ylabel('Relative count of neurons')
	plt.title('Neurons from the inanimate cluster')
	plt.xticks(X, ('conv_1', 'conv_2', 'conv_3', 'conv_4','conv_5','fc1','fc2'))
	#autolabel(rect)
	plt.savefig('../../results/scree/'+str(method)+'_results_ina_hist_alt.png', format='png')

	X = np.arange(1,8)
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
	print tmp
	width = 0.35  # the width of the bars

	rects1 = plt.bar(X- width/2, Y, width, label='Layer-specific cluster lesions')
	rects2 = plt.bar(X+width/2, Z, width, label='Random layer lesions')

	plt.ylabel('Relative change in performance impact')
	plt.xlabel('Different layers of Alexnet')
	plt.xticks(X, ('conv_1', 'conv_2', 'conv_3', 'conv_4','conv_5','fc1','fc2'))
	plt.title('Change in Performance by layer for Animate')
	plt.legend()
	plt.savefig('../../results/scree/animate_by_layer.png', format='png')
	#plt.ylim([0,1])

	X = np.arange(1,8)
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
	plt.xticks(X, ('conv_1', 'conv_2', 'conv_3', 'conv_4','conv_5','fc1','fc2'))
	plt.xlabel('Different layers of Alexnet')
	plt.title('Change in Performance by layer for Inanimate')
	plt.legend()
	plt.savefig('../../results/scree/inaniamte_by_layer.png', format='png')
	#plt.ylim([0,1])


def return_idx():
	import imp
	import matplotlib.pyplot as plt
	import numpy as np
	import os


	utils = imp.load_source("utils", '/content/utils.py')

	# Load an image.
	# Need to download examples images first.
	# See script in images directory.
	image1 = utils.load_image('/content/ILSVRC2012_val_00011795.jpeg', 224)

	# Code snippet.
	plt.imshow(image1/255)
	plt.axis('off')
	plt.savefig("readme_example_input.png")

	import innvestigate
	import innvestigate.utils
	import keras.applications.vgg16 as vgg16

	# Get model
	model, preprocess = vgg16.VGG16(), vgg16.preprocess_input
	# Strip softmax layer
	model = innvestigate.utils.model_wo_softmax(model)

	# Create analyzer
	analyzer = innvestigate.create_analyzer("lrp.z", model,reverse_keep_tensors=True)


	#gradient, deconvnet, guided_backprop, deep_taylor, input_t_gradient, lrp.sequential_preset_a_flat, lrp.sequential_preset_b_flat, lrp.z, lrp.epsilon


	# Add batch axis and preprocess
	x = preprocess(image1[None])
	# Apply analyzer w.r.t. maximum activated output-neuron
	a = analyzer.analyze(x)

	# Aggregate along color channels and normalize to [-1, 1]
	a = a.sum(axis=np.argmax(np.asarray(a.shape) == 3))
	a /= np.max(np.abs(a))
	# Plot
	plt.imshow(a[0], cmap="seismic", clim=(-1, 1))
	plt.axis('off')

	# Add batch axis and preprocess
	x = preprocess(image1[None])
	# Apply analyzer w.r.t. maximum activated output-neuron
	a = analyzer.analyze(x)

	# Aggregate along color channels and normalize to [-1, 1]
	a = a.sum(axis=np.argmax(np.asarray(a.shape) == 3))
	a /= np.max(np.abs(a))
	# Plot
	plt.imshow(a[0], cmap="seismic", clim=(-1, 1))
	plt.axis('off')


	image_activations = np.mean(image_activations, axis=0).ravel()
	image_activations.shape


	number = 30 
	index= sorted(range(len(image_activations), key=lambda k: abs(image_activations[k]),reverse=True)
	print(index[0:number])

	idx = np.ones(image_activations.shape)
	idx[index[0:number]] = 0.
	
