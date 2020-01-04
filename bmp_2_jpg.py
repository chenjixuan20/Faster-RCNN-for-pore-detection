import os
import random

from PIL import Image

file_path= '/data/VOCdevkit2007/VOC2007_new/ground_truth/original/'
ftrain = open('/data/VOCdevkit2007/VOC2007_new/ImageSets/Layout/train.txt', 'w')
fval = open('/data/VOCdevkit2007/VOC2007_new/ImageSets/Layout/val.txt', 'w')

total=os.listdir(file_path)
num=len(total)
list=range(num)
tr=int(num*0.5)
train=random.sample(list, tr)

for i in list:
    name = total[i][:-4]+'\n'
    print(name)
    if i in train:
        ftrain.write(name)
    else:
        fval.write(name)
ftrain.close()
fval.close()


# for i, fileName in enumerate(sorted(os.listdir(file_path))):
#     print(fileName)
#     print(type(fileName))
#     if fileName.endswith('.jpg'):
#         print(fileName[:-4])
#         ftrainval.write(fileName[:-4] + '\n')
#         # if i < 39:
#         #     os.rename(file_path+fileName, file_path+'0000'+str(i+61) + '.jpg')
#         # else:
#         #     os.rename(file_path+fileName, file_path+'000'+str(i+61) + '.jpg')
    # im=open(file_path+'/'+fileName)
    # im.save(file_path+'/'+NewfileName)
    # os.system('rm ' + file_path+'/'+fileName)

# os.system('del'+file_path+'/*.bmp')

