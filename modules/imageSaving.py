import os
import modules.utilityFunctions as uf
import cv2


# this function saves new image fcr training
# it gets index number of directory and based on index number from dir_list which is class for classification
# it saves image there
def saveNewImage(list_ind,frame):
    _, _, test_save = uf.getDirectoryList()
    target_dir = str(test_save[int(list_ind) - 1])
    #print("inside save new image routine")
    #print(list_ind)
    #print(target_dir)
    parent_path = r'F:\\study material\\AI and ML\\LabWork\\final task\\videoFeedProcessing\\data\\train\\'
    #print(parent_path)

    path = os.path.join(parent_path, target_dir)
    res = os.listdir(path)
    #print(len(res))
    #print(path)
    tmp_text = str(len(res) + 1) + ".jpg"
    #print(tmp_text)
    path_2 = os.path.join(path, tmp_text)
    #print(path_2)

    w, h, c = frame.shape
    w = w / 2
    h = h / 2

    img_cropped = frame[int(w - 128):int(w + 128), int(h - 128):int(h + 128)]
    cv2.imwrite(path_2, img_cropped)



def addNewClass(dir_name,frame):
    #print("name and image recieved !")

    w, h, c = frame.shape
    w = w / 2
    h = h / 2

    img_cropped = frame[int(w - 128):int(w + 128), int(h - 128):int(h + 128)]


    parent_path = r'F:\\study material\\AI and ML\\LabWork\\final task\\videoFeedProcessing\\data\\train\\'
    path = os.path.join(parent_path, dir_name)
    os.mkdir(path)
    #print(path)

    res = os.listdir(path)
    #print(len(res))
    #print(path)
    tmp_text = str(len(res) + 1) + ".jpg"
    #print(tmp_text)
    path_2 = os.path.join(path, tmp_text)
    #print(path_2)

    cv2.imshow("saving image", img_cropped)
    cv2.imwrite(path_2, img_cropped)
    #print("image saved")
