import os
import numpy as np
import cv2

#data loading
def load_data_from_directory(par_dir_from_user):
    parent_dir=par_dir_from_user
    #parent_dir = r"{tmp}".format(tmp=par_dir_from_user)
    dir_list=os.listdir(parent_dir)
    dir_list.sort()
    print(dir_list)
    train_data_list=[]
    train_labels = []
    tmp_train_list=[]


    for i in range (len(dir_list)):
      sub_dir_list = os.path.join(parent_dir,dir_list[i])
      #print("sub_dir_list")
      #print(sub_dir_list)

      files= os.listdir(r"{path}".format(path=sub_dir_list))
      for j in range(len(files)):
        img_location_tmp = sub_dir_list+"/"+files[j]
        tmp_img = cv2.imread(img_location_tmp,1)
        tmp_img_resized = cv2.resize(tmp_img,(128,128),interpolation=cv2.INTER_AREA)
        tmp_img_recolor = cv2.cvtColor(tmp_img_resized,cv2.COLOR_BGR2GRAY)
        train_data_list.append(tmp_img_recolor)
        train_labels.append(dir_list[i])
        tmp_train_list.append(i)


    np_array_img = np.array(train_data_list)

    X = np_array_img/255

    y = np.array(tmp_train_list)

    return X,y
