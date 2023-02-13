#some helping funcions for main function


import cv2
import modules.utilityFunctions as uf
import modules.validationSection as vs




#this function is called in main file/main loop
def capture_image(img,trainable=False):
    w, h, c = img.shape
    w = w / 2
    h = h / 2

    img_cropped = img[int(w - 128):int(w + 128), int(h - 128):int(h + 128)]
    img_processed = uf.image_prep(img_cropped)

    res, all_pred = uf.get_result(img_processed)
    _, _, all_classes = uf.getDirectoryList()


    #print("first pred" + str(all_pred[0][0]))

    sample_image_resized = cv2.resize(img_cropped, (600, 600), interpolation=cv2.INTER_AREA)
    sample_image_resized_copy = sample_image_resized.copy()
    print(res)
    print(all_classes)
    uf.putTextOnImg(sample_image_resized, all_classes[res], 0)
    for i in range(len(all_classes)):
        conf_txt = (all_classes[i] + " Confidence  " + str(all_pred[0][i]))
        uf.putTextOnImg(sample_image_resized, conf_txt, 1+i)


    cv2.imshow("sample image", sample_image_resized)

    # calling 3rd section of code
    if trainable == True:
        flag_value_to_start_training = vs.check_predicion(sample_image_resized_copy,res,img)
        return flag_value_to_start_training
    #cv2.destroyWindow("sample image")
