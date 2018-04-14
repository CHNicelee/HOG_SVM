# -*- coding=utf-8 -*-
import glob
import time
from PIL import Image
from skimage.feature import hog
import numpy as np
import os
import joblib
from sklearn.svm import LinearSVC
import shutil
from matplotlib import pyplot

#这里写类别对应的名称
#如果预测的结果也用数字表示 那么把字符串的内容改为数字即可
label_map = {0:'train',
             1:'bus',
             2:'person',
             3:'cat',
             4:'car'}
			 
#测试集图片的位置
test_image_path = '/home/icelee/Downloads/dataset/周三比赛/'
size = 128

train_feat_path = 'train/'
test_feat_path = 'test/'
model_path = 'model/'


def get_map():
    pic_map = {}
    file_name = ''
    with open('/home/icelee/Downloads/dataset/annotations.txt','r') as f:
        for line in  f.readlines():
            if line.find('.jpg')>0:
                pic_map[line.split('.')[0]+'.jpg'] = ''
                file_name = line.split('.')[0]+'.jpg'
                continue
            if len(line)<4:
                continue
            pic_map[file_name] = pic_map[file_name]+line.split(' ')[0]
    return pic_map


#获得图片列表
def get_image_list(filePath,nameList):
    img_list = []
    for name in nameList:
        img_list.append(Image.open(filePath+name))
    return img_list


#变成灰度图片
def rgb2gray(im):
    gray = im[:, :, 0]*0.2989+im[:, :, 1]*0.5870+im[:, :, 2]*0.1140
    return gray

#提取特征
def extra_feat():
    get_feat_new()

# 只提取待测图片的特征
def get_feat_new():
    for path in glob.glob(test_image_path+'*.jpg'):
        image = Image.open(path).resize((size, size), Image.ANTIALIAS)

		#如果你的图片不是彩色的  可能需要把3改为-1
        image = np.reshape(image, (size, size, 3))
        gray = rgb2gray(image) / 255.0
        # 这句话根据你的尺寸改改
        fd = hog(gray, orientations=12, pixels_per_cell=[8, 8], cells_per_block=[4, 4], visualise=False,
                 transform_sqrt=True)
        fd = np.concatenate((fd, [1]))
        fd_name = path[-10:] + '.feat'
        fd_path = os.path.join(test_feat_path, fd_name)
        joblib.dump(fd, fd_path)
    print "Test features are extracted and saved."


# 创建存放特征的文件夹
def mkdir():
    if not os.path.exists(train_feat_path):
        os.mkdir(train_feat_path)
    if not os.path.exists(test_feat_path):
        os.mkdir(test_feat_path)

#训练和测试
def train_and_test():
    t0 = time.time()
    clf_type = 'LIN_SVM'
    fds = []
    labels = []
    num = 0
    total = 0

    if clf_type is 'LIN_SVM':
		
		#model名称自己修改一下
        clf = joblib.load(model_path+'model')
        print "已加载model"

        result_list = []
        for feat_path in glob.glob(os.path.join(test_feat_path, '*.feat')):
            # total += 1
            image_name = feat_path.split('/')[1].split('.feat')[0]
            data_test = joblib.load(feat_path)
            data_test_feat = data_test[:-1].reshape((1,-1)).astype(np.float64)
            result = clf.predict(data_test_feat)
            result_list.append(image_name+' '+label_map[int(result[0])]+'\n')
        write_to_txt(result_list)
        # os.system('cat '+os.getcwd()+"/result.txt | sort -t '.' -k 1 > result_sorted.txt" )
        # rate = float(num)/total
        t1 = time.time()
        # print '准确率是： %f'%rate
        print '耗时是 : %f'%(t1-t0)
		
# 将结果写入文件
def write_to_txt(list):
    with open('result.txt','w') as f:
        f.writelines(list)

if __name__ == '__main__':

    mkdir()  # 不存在文件夹就创建

    need_input = raw_input('是否手动输入各个信息？y/n\n')

    if need_input == 'y':
        test_image_path = raw_input('请输入测试图片文件夹的位置,如 /home/icelee/image/ 结尾一定要带有分隔符\n')
        size = int(raw_input('请输入您图片的大小：如64x64，则输入64\n'))

    need_extra_feat = raw_input('是否需要重新获取特征？y/n\n')

    if need_extra_feat == 'y':
        # shutil.rmtree(train_feat_path)
        shutil.rmtree(test_feat_path)
        mkdir()
        extra_feat()#获取特征并保存在文件夹
    # exit()
    train_and_test() #训练并预测

