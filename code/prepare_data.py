import shutil
import os

classNames = ['animate','inanimate']
ids = [['3','6','389','653'],['300','512','664','934']]

for i in range(len(classNames)):
	class_name = classNames[i]
	try:
		os.mkdir('../data/'+class_name)
	except Exception as e:
		pass
	id_list = ids[i]
	with open('../data/ILSVRC2014_clsloc_validation_ground_truth.txt') as f:
		line_num = 1	
		for line in f.readlines():
			line_id = line.strip('\n')
			zeros = '0'*(8 - len(str(line_num)))
			id_ = zeros+str(line_num)
			img = 'ILSVRC2012_val_'+id_+'.JPEG'
			if line_id in id_list:
				shutil.copyfile('../data/ILSVRC2012_img_val/'+img, '../data/'+class_name+'/'+img)  
				print line_num, line_id, img
			line_num += 1
