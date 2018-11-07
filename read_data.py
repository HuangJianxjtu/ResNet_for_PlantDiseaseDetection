import cv2
import json
import pandas as pd
import os
import numpy as np
import h5py
import os


# filePath = './images'
# shapes=[]
# Doc_names =  os.listdir(filePath)
# for i in range(100):
# 	img = cv2.imread(filePath+str('/')+Doc_names[i])
# 	# cv2.imshow("image",img)
# 	# cv2.waitKey(1000)
# 	shapes.append(img.shape)
# print(shapes)


# type of img  == numpy.ndarry
# img = cv2.imread('./images/0ecbf075894bda7b5993de6209b7147c.jpg')  #height >= width
# img1 = cv2.imread('./images/0f20c05e75d152dc0eb3f07f05779648.jpg')   #height < width
# print(type(img))
# print('img.shape',img.shape)
#print(img[:,:,0])
# # show image
#cv2.imshow("original image",img)

def img_resize(image, img_size):
  height = image.shape[0]
  width = image.shape[1]
  #print(height,width)
  if(height>=width):
    #print('height >= width')
    scale = float(img_size)/float(height)
    width_new = int(float(width)*float(scale))
    height_new= img_size
    res = cv2.resize(image, (width_new,height_new), interpolation=cv2.INTER_AREA)   #(width,height)
    # padding
    a_half = int((float(img_size)-float(width_new))/2)
    top = 0
    bottom = 0
    left = a_half
    right = img_size-width_new-left 
    #print(top,bottom,left,right,"a half = ",a_half)
    res_padding = cv2.copyMakeBorder(res,top,bottom,left,right,cv2.BORDER_CONSTANT,value=(0,0,0)) 
  else:
    #print('height < width')
    scale = float(img_size)/float(width)
    width_new = img_size
    height_new= int(float(height)*float(scale))
    res = cv2.resize(image, (width_new,height_new), interpolation=cv2.INTER_AREA)   #(width,height)
    # padding
    a_half = int((float(img_size)-float(height_new))/2)
    top = a_half
    bottom = img_size-height_new-top
    left = 0 
    right = 0 
    #print(top,bottom,left,right,"a half = ",a_half)
    res_padding = cv2.copyMakeBorder(res,top,bottom,left,right,cv2.BORDER_CONSTANT,value=(0,0,0)) 

  return res_padding

# new_img = img_resize(img,256)
# new_img1 = img_resize(img1,256)
# print('shape of new image',new_img.shape)
# print('type(new_img)',type(new_img))
# cv2.imshow("new image",new_img)

# cv2.waitKey(8000)
# def read_one_image_to_json(img_path,json_file_path):

# image_sets=[]
# image_sets.append(new_img)
# image_sets.append(new_img1)
# disease_ids=[1,2]
# print('type(image_sets)',type(image_sets))
# print('type(image_sets[0])',type(image_sets[0]))

# f = h5py.File('data/test_temp.h5', 'w')
# print('new img',new_img[100,:,:])
#incrementally writes to hdf5 with h5py
# for i in range(2):
#     dataset_1 = f.create_dataset('X_test_image',data = image_sets[i])
#     f.create_dataset('disease_ids',data = disease_ids[i])
#     print(i)
# f.close()



# set file path and other parameters---- IMPORTANT
DIR_Path = 'images1'
json_file_path = 'AgriculturalDisease_validation_annotations.json'
read_step_num = 3        # the number of files we write to .h5 file for one time
dataset_name_X= 'X_valid'   # for valid set
dataset_name_Y= 'Y_valid'   # for valid set
image_size = 128
data_name = 'data/validSet.h5'   #for valid set

# read the names of files
image_name_IDs = os.listdir(DIR_Path)
m = len(image_name_IDs)    # the total number of samples
image_name_IDs = np.array(image_name_IDs)        # np.array
#print('type(image_name_IDs)',type(image_name_IDs))
# print(image_name_IDs[[1,2]])

# scheduling the reading process
print('scheduling the reading process')
iterate_num = int(m/read_step_num)
rest_num = m-iterate_num*read_step_num
print('total number of images =',m)
print('number of reading files a time for =',read_step_num)
print('number of iterations =',iterate_num)
print('the rest number of files =',rest_num)
print('\n')

# read json for images' disease classes
label_data_temp = pd.read_json(json_file_path,typ='frame',orient='table')
label_data = label_data_temp.set_index('image_id')
#print(label_data)

# reading files and write them into .h5 file
f = h5py.File(data_name, 'w')
X_set = f.create_dataset(dataset_name_X, shape = (read_step_num,image_size,image_size,3),maxshape=(None,image_size,image_size,3))
Y_set = f.create_dataset(dataset_name_Y, shape = (read_step_num,image_size,image_size,3),maxshape=(None,image_size,image_size,3))

# scrable the order of files
permutation_nums = np.random.permutation(m)      # np.array

# read data and write them to .h5 file
    # incrementally writes to .h5 file 
for i in range(iterate_num):                
    image_name_IDs_batch = image_name_IDs[permutation_nums[i*read_step_num:(i+1)*read_step_num]]
    dataset_x_temp = []
    dataset_y_temp = []
    for item_id in image_name_IDs_batch:
        X_item_temp = cv2.imread(DIR_Path+'/'+item_id)
        X_item_temp = img_resize(X_item_temp,image_size)       # resize
        dataset_x_temp.append(X_item_temp)
        Y_item_temp = int(label_data.loc[item_id])         # access the element through index
        dataset_y_temp.append(Y_item_temp)
        # checking
        cv2.imshow("new image",X_item_temp)
        print('ID = ',item_id)
        print('label (disease classe) =',Y_item_temp)
        cv2.waitKey()
    # write to .j5 file
    X_set[i*read_step_num:(i+1)*read_step_num] = dataset_x_temp
    Y_set[i*read_step_num:(i+1)*read_step_num] = dataset_y_temp
    print('I have red and wrote ',(i+1)*iterate_num,'of',m,'images')
    # resize the .h5 file
    if(i < iterate_num-1):
        X_set.resize((len(dataset_x_temp)+read_step_num,image_size,image_size,3))
        Y_set.resize((len(dataset_y_temp)+read_step_num,image_size,image_size,3))
    else:
        print('incrementally writing completed')

if rest_num != 0:
    print('\n')
    print('reading the rest images now')
    X_set.resize((len(dataset_x_temp)+rest_num,image_size,image_size,3))
    Y_set.resize((len(dataset_y_temp)+rest_num,image_size,image_size,3))
    image_name_IDs_batch = image_name_IDs[-rest_num:]
    dataset_x_temp = []
    dataset_y_temp = []
    for item_id in image_name_IDs_batch:
        X_item_temp = cv2.imread(DIR_Path+'/'+item_id)
        X_item_temp = img_resize(X_item_temp,image_size)       # resize
        dataset_x_temp.append(X_item_temp)
        Y_item_temp = int(label_data.loc[item_id])         # access the element through index
        dataset_y_temp.append(Y_item_temp)
        # write to .j5 file
    X_set[(iterate_num+1)*read_step_num:m] = dataset_x_temp
    Y_set[(iterate_num+1)*read_step_num:m] = dataset_y_temp
print('reading completed')
f.close()
# dst1 = f.create_dataset('X_test_image', shape = (1,256,256,3),maxshape=(None,256,256,3))
# print('shape of dst1',len(dst1))
# dst1[0] = image_sets[0]
# dst1.resize((2,256,256,3))
# dst1[1] = image_sets[1]
# print('shape of dst1',len(dst1))
# f.close()


# # print(data.loc['43234193db4aefa1245592ab36d6c946.jpg'])    # access the element through index
# # #print(data.loc['43234193db4aefa1245592ab36d6c946.jpg']==1)
# # print('str = ',str(data.loc['43234193db4aefa1245592ab36d6c946.jpg']['disease_class']))