import cv2
import modules.utilityFunctions as uf
import modules.imageSaving as imgSav
import time

def check_predicion(img_win,model_res,img):
    count = 0
    chk = False
    not_good = True
    flag_value_to_start_training=False
    #print("inside 3rd section")
    uf.putTextOnImg(img_win, "Was Prediction right? (y/n)", count)
    count += 1
    cv2.imshow("validation window", img_win)
    res, _ = uf.keyboard_input()

    if res == 'Y' or res == 'y':
        #print("prediction was right!")
        uf.putTextOnImg(img_win, "Am I Allowed to Save image for learning?", count)
        count += 1
        not_good = True
        cv2.imshow("validation window", img_win)
        res_yes, _ = uf.keyboard_input()
        if res_yes == 'Y' or res_yes == 'y':
            imgSav.saveNewImage(model_res+1,img)
            not_good = True
            cv2.destroyWindow("validation window")

        elif res_yes == "n":
            uf.putTextOnImg(img_win, "Discarding image", count)
            count += 1
            #print("not allowed to save")
            cv2.destroyWindow("validation window")

    # if prediction is wrong
    if res == 'n' or res == 'N':
        uf.putTextOnImg(img_win, "Will you help me improve? (y/n)", count)
        count += 1
        #print("prediction was wrong")
        cv2.imshow("validation window", img_win)
        res_no, _ = uf.keyboard_input()
        if res_no == 'y' or res_no == 'y':
            tmp_txt, dir_len, _ = uf.getDirectoryList()
            tmp_txt = tmp_txt + " "+ str(0) + " Other "

            #print("showing directory list")
            #print(tmp_txt)
            uf.putTextOnImg(img_win, tmp_txt, count)
            count += 1
            cv2.imshow("validation window", img_win)
            optTxt = "Choose accurate Label(Enter number)"
            uf.putTextOnImg(img_win, optTxt, count)
            count += 1
            cv2.imshow("validation window", img_win)
            res_opt, _ = uf.keyboard_input()
            ## save image routine !!

            if res_opt == 0 or res_opt == '0':
                #print("starting Other class Routine")
                nameTxt = "What is this object?"
                uf.putTextOnImg(img_win, nameTxt, count)
                cv2.imshow("validation window", img_win)
                res_save, _ = uf.keyboard_input()
                imgSav.addNewClass(res_save,img)
                flag_value_to_start_training = True

            else:
                imgSav.saveNewImage(res_opt,img)
                cv2.destroyWindow("validation window")



        else:
            #print("dabadee daba dee 1")
            cv2.destroyWindow("validation window")

    # if some other key press
    else:
        if not_good==False:
            uf.putTextOnImg(img_win, "Something went wrong", count)
            count += 1
            cv2.imshow("validation window", img_win)
        else:
            #cv2.destroyWindow("validation window")
            pass




    return flag_value_to_start_training
