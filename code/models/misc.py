def clustering_performance():
	#D' histogram for animate and inanimate double bars for different clustering techniques

	labels = ['KMeans','GMM','SMM','Birch','Single','Ward','DBSCAN','HDBSCAN','Genie']
	animate_means = [64.10,66.67,69.23, 25.64,0,58.97,0,17.59,12.82]
	inanimate_means = [79.49,79.49,82.05,71.79,23.07,82.07,23.07,23.07,38.46]

	x = np.arange(len(labels))  # the label locations
	width = 0.35  # the width of the bars

	fig, ax = plt.subplots()
	rects1 = ax.bar(x - width/2, animate_means, width, label='Animate')
	rects2 = ax.bar(x + width/2, inanimate_means, width, label='Inanimate')

	# Add some text for labels, title and custom x-axis tick labels, etc.
	ax.set_ylabel('Performance Impact')
	ax.set_title('Performance by Class')
	ax.set_xticks(x)
	ax.set_xticklabels(labels,rotation=90)
	ax.legend()
	plt.ylim([0.,100.])

	def autolabel(rects, xpos='center'):
	    xpos = xpos.lower()  # normalize the case of the parameter
	    ha = {'center': 'center', 'right': 'left', 'left': 'right'}
	    offset = {'center': 0.5, 'right': 0.57, 'left': 0.43}  # x_txt = x + w*off

	    for rect in rects:
		height = rect.get_height()
		ax.text(rect.get_x() + rect.get_width()*offset[xpos], 1.01*height,
		        '{}'.format(height), ha=ha[xpos], va='bottom',size=8)
		
	autolabel(rects1)
	autolabel(rects2)
	plt.savefig('../../results/scree/clustering_results.png', format='png')
