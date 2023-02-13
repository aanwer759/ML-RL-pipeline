

import cv2
import modules.predictImageClass as pic
import modules.utilityFunctions as uf
import modules.model as model





# initializing video feed
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)


mode_check = input ("Enter 1 for continous and 2 for picture/Training mood")
mode_check = int(mode_check)

uf.get_model()

while True:
    _, frame = cap.read()
    img_boundary = cv2.rectangle(frame, (192, 112), (448, 368), (100, 200, 150), 10)

    if mode_check == 1:
        cv2.imshow("frame", img_boundary)
        pic.capture_image(frame)
    if mode_check == 2:
        uf.putTextOnImg(img_boundary, "Press C or c to capture image", 0)
        cv2.imshow("frame", img_boundary)
        if cv2.waitKey(1) == ord('c') or cv2.waitKey(1) == ord('C'):
            print("capturing image for recognition !")
            flag_value_to_start_training = pic.capture_image(frame,trainable=True)
            print("inside Image capture mood")
            if flag_value_to_start_training:
               model.model_training(True)
               uf.get_model()

    if cv2.waitKey(1) == ord('q'):
        break

# releasing all windows and clearing all remaining memory footprints
cap.release()
cv2.destroyAllWindows()



