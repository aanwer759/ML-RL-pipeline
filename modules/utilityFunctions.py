import numpy as np
import cv2
from sklearn.metrics import accuracy_score
import os
import pyshine
import string
import modules.model as model

model1=''

def image_prep(tmp_img):
    img_tst = []
    tmp_img_resized = cv2.resize(tmp_img, (128, 128), interpolation=cv2.INTER_AREA)
    tmp_img_recolor = cv2.cvtColor(tmp_img_resized, cv2.COLOR_BGR2GRAY)
    img_tst.append(tmp_img_recolor)
    test_img = np.array(img_tst)
    testtt = test_img / 255

    return testtt

def get_model():
    global model1
    model1 = model.get_trained_model()
def get_result(tmp_img):
    # model.predict(tmp_img)
    #make one global model and send that model again or save this model here

    pred = model1.predict(tmp_img)
    res = get_max_value_index(pred)

    return res[0], pred

def get_accuracy(pred_list, y_test):
    accuracy = accuracy_score(pred_list, y_test)
    # print(accuracy)
    return accuracy


def get_max_value_index(predictions):
    pred_list = []
    for i in range(len(predictions)):
        pred_list.append(np.argmax(predictions[i]))
    return pred_list


def getDirectoryList():
    # folder path
    dir_path = r'F:\\study material\\AI and ML\\LabWork\\final task\\videoFeedProcessing\\data\\train'

    # list file and directories
    res = os.listdir(dir_path)
    # print(res)
    s = ''
    for i in range(len(res)):
        tmp_str = res[i]
        s = s + " " + str(i + 1) + " " + tmp_str

    print("inside get directory function")
    print(res)

    return s, len(res), res



def putTextOnImg(img, text, count):
    count = count * 30
    pyshine.putBText(img, text, text_offset_x=50, text_offset_y=30 + count, vspace=10, hspace=10,
                     font_scale=0.5,
                     background_RGB=(0, 0, 0), text_RGB=(255, 250, 250))



# keyboard input on CV2 window itself
def keyboard_input():
    text = ""
    letters = string.ascii_lowercase + string.digits
    while True:
        key = cv2.waitKey(1)
        for letter in letters:
            if key == ord(letter):
                text = text + letter

        if key == ord("\n") or key == ord("\r"):  # Enter Key
            break
    return text, True

