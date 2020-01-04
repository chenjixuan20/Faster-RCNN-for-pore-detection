#! /usr/bin/python
# -*- coding:UTF-8 -*-
import os, sys
import glob
from PIL import Image

# 图像存储位置
src_img_dir = '/data/VOCdevkit2007/VOC2007_new/JPEGImages'
# 图像的 ground truth 的 txt 文件存放位置
src_txt_dir = '/data/VOCdevkit2007/VOC2007_new/Annotations_txt'
src_xml_dir = '/data/VOCdevkit2007/VOC2007/Annotations_4'

# glob.glob文件名模式匹配 返回路径符合参数的所有的文件
# （*表示匹配零个或多个字符/ ？表示匹配任何单个字符/ 可以用[]限定范围）
# 所以这里返回的是src_img_dir图片存储路径下所有的以.jpg结尾的文件名
img_Lists = glob.glob(src_img_dir + '/*.jpg')



# os.path.basename返回的是路径中最低层的文件名 如100.jpg
# 所以最后img_basenames保存了该路径下的所有的图片的文件名，包括扩展名后缀
img_basenames = []  # e.g. 100.jpg
for item in img_Lists:
    img_basenames.append(os.path.basename(item))

# os.path.splitext是分离文件名和扩展名
# 依次返回文件名和扩展名：（fname,fextension）（如果有的话）
# 所以最后img_names中保存了所有文件的名称，不包括后缀
img_names = []  # e.g. 100
for item in img_basenames:
    temp1, temp2 = os.path.splitext(item)
    img_names.append(temp1)

for img in img_names:
    # 打开图片获取图片的size
    im = Image.open((src_img_dir + '/' + img + '.jpg'))
    width, height = im.size

    # 打开图片对应的txt文档，读取文档中每一行的数据存储在gt中
    # 对应到我们的txt文档也就是，依次读取每一个毛孔的坐标
    # open the crospronding txt file
    gt = open(src_txt_dir + '/' + img + '.txt').read().splitlines()
    # gt = open(src_txt_dir + '/gt_' + img + '.txt').read().splitlines()

    # 创建一个对应的xml文件
    # 将在txt中读取的内容以xml要求的格式写入新创建的xml文档
    # write in xml file
    # os.mknod(src_xml_dir + '/' + img + '.xml')
    xml_file = open((src_xml_dir + '/' + img + '.xml'), 'w')
    xml_file.write('<annotation>\n')
    xml_file.write('    <folder>VOC2007</folder>\n')
    xml_file.write('    <filename>' + str(img) + '.jpg' + '</filename>\n')
    xml_file.write('    <source>\n')
    xml_file.write('        <database>The VOC2007 Database</database>\n')
    xml_file.write('        <annotation>PASCAL VOC2007</annotation>\n')
    xml_file.write('        <image>flickr</image>\n')
    xml_file.write('        <flickrid>341012865</flickrid>\n')
    xml_file.write('    </source>\n')
    xml_file.write('    <owner>\n')
    xml_file.write('        <flickrid>Fried Camels</flickrid>\n')
    xml_file.write('        <name>Jinky the Fruit Bat</name>\n')
    xml_file.write('    </owner>\n')
    xml_file.write('    <size>\n')
    xml_file.write('        <width>' + str(width) + '</width>\n')
    xml_file.write('        <height>' + str(height) + '</height>\n')
    xml_file.write('        <depth>1</depth>\n')
    xml_file.write('    </size>\n')
    xml_file.write('    <segmented>0</segmented>\n')

    # write the region of image on xml file
    for img_each_label in gt:
        spt = img_each_label.split(' ')  # 这里如果txt里面是以逗号‘，’隔开的，那么就改为spt = img_each_label.split(',')。
        xml_file.write('    <object>\n')
        xml_file.write('        <name>pore</name>\n')
        xml_file.write('        <pose>Unspecified</pose>\n')
        xml_file.write('        <truncated>0</truncated>\n')
        xml_file.write('        <difficult>0</difficult>\n')
        xml_file.write('        <bndbox>\n')
        xml_file.write('            <xmin>' + str(max(int(spt[1])-4, 1)) + '</xmin>\n')
        xml_file.write('            <ymin>' + str(max(int(spt[0])-4, 1)) + '</ymin>\n')
        xml_file.write('            <xmax>' + str(min(int(spt[1])+4, width)) + '</xmax>\n')
        xml_file.write('            <ymax>' + str(min(int(spt[0])+4, height)) + '</ymax>\n')
        xml_file.write('        </bndbox>\n')
        xml_file.write('    </object>\n')

    xml_file.write('</annotation>')
