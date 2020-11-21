import shutil
import os

classNames = ['T5A','T5B']
ids = [['34','58','111','398'],['304','518','668','960']]
#ids_t4 = [['30','28','414','634'],['376','548','683','958']]
#ids_t3 = [['1','8','393','647'],['379','540','674','955']]
#ids_t2 = [['5','7','391','655'],['375','516','675','944']]
#ids_t1 = [['4','22','390','654'],['301','665','515','951']]
#ids_5 =  [['3','300','664','653'],['6','512','389','934']]
#ids_4 =  [['3','300','389','653'],['6','512','664','934']]
#ids_3= [['300','512','664','653'],['3','6','389','934']]
#ids_2 = [['300','512','389','653'],['3','6','664','934']]
#ids_1 = [['300','6','389','653'],['3','512','664','934']]
#ids_org = [['3','6','389','653'],['300','512','664','934']]

for i in range(len(classNames)):
		class_name = classNames[i]
		try:
			os.mkdir('../../data/'+class_name)
		except Exception as e:
			pass
		id_list = ids[i]
		with open('../../data/ILSVRC2014_clsloc_validation_ground_truth.txt') as f:
			line_num = 1	
			for line in f.readlines():
				line_id = line.strip('\n')
				zeros = '0'*(8 - len(str(line_num)))
				id_ = zeros+str(line_num)
				img = 'ILSVRC2012_val_'+id_+'.JPEG'
				if line_id in id_list:
					shutil.copyfile('../../../ILSVRC2012_img_val/'+img, '../../data/'+class_name+'/'+img)  
					print(line_num, line_id, img)
				line_num += 1

'''
import itertools
a = ['3','6','389','653','300','512','664','934']
res = {}
for item in list(itertools.permutations(a,4)):
    if res.get(tuple(sorted(list(item))),None) != None:
        continue
    else:
        res[tuple(sorted(list(item)))] = 1
id_list =[]
for key in res.keys():
    ids = []
    ids.append(list(key))
    temp=[]
    for i in a:
        if i not in list(key):
            temp.append(i)
    ids.append(list(temp))
    id_list.append(ids)

idx = 1
for ids in id_list:
	classNames = [str(idx)+'A', str(idx)+'B']
	idx+=1
	for i in range(len(classNames)):
		class_name = classNames[i]
		try:
			os.mkdir('../../data/mobile/'+class_name)
		except Exception as e:
			pass
		id_list = ids[i]
		with open('../../data/ILSVRC2014_clsloc_validation_ground_truth.txt') as f:
			line_num = 1	
			for line in f.readlines():
				line_id = line.strip('\n')
				zeros = '0'*(8 - len(str(line_num)))
				id_ = zeros+str(line_num)
				img = 'ILSVRC2012_val_'+id_+'.JPEG'
				if line_id in id_list:
					shutil.copyfile('../../../ILSVRC2012_img_val/'+img, '../../data/'+class_name+'/'+img)  
					print(line_num, line_id, img)
				line_num += 1
'''
