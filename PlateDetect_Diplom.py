import cv2 as cv
import recog
import datetime

cap = cv.VideoCapture(0, cv.CAP_DSHOW)
cascade = cv.CascadeClassifier(r"cascades\plates_cascade.xml")

def capt():
    while True:
        _, image = cap.read()
        #image = cv.imread('plate2.png')
        threshold = 100
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        plate = cascade.detectMultiScale(gray,
                                        scaleFactor=1.15,
                                        minNeighbors=6,
                                        minSize=(4, 4),
                                        flags=cv.CASCADE_SCALE_IMAGE)

        cv.imshow("PlateDetectRU_DiplomDEV", image)
        if str(plate) != "()":
            blurdetect = cv.Laplacian(image, cv.CV_64F).var()
            if blurdetect > threshold:
                for x, y, width, height in plate:
                    cv.rectangle(image, (x, y), (x + width, y + height), color=(255, 0, 0), thickness=1)
                today = datetime.datetime.today()
                rndmplate = today.strftime("%H_%M_%S")
                platename = f"{rndmplate}.jpg"
                cv.imshow("PlateDetectRU_DiplomDEV", image)
                path_to_notfilt = r"license_plates\notfilt\{}".format(platename)
                cv.imwrite(path_to_notfilt, image)
                recog.process(path_to_notfilt, False)
            else:
                pass
        else:
            pass

        if cv.waitKey(1) == ord("q"):
            break

capt()

cap.release()
cv.destroyAllWindows()