import cv2
import numpy as np
import imutils
import pytesseract as tes
import datetime
import sqlite3
import datetime

from rotate import find_angle
from rotate import rotate

#часть с бд
conn = sqlite3.connect('logs_plates.db')
cur = conn.cursor()
global text
text = None
table_query = '''CREATE TABLE IF NOT EXISTS LOGS(
                                id INTEGER PRIMARY KEY AUTOINCREMENT,
                                date TEXT NOT NULL,
                                date_hms TEXT NOT NULL,
                                plate_number TEXT NOT NULL,
                                plate_path TEXT NOT NULL);'''

cur.execute(table_query)


# путь к фотографии для обработки

# путь к файлу с каскадом Хаара
faceCascade = cv2.CascadeClassifier(r'cascades/plates_cascade.xml')
tes.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

"""
Функция удаляет одинаковые плашки номеров, чтобы те не обрабатывались два и более раз
"""
def only_different(plaques):
    plaques = list(plaques)
    for i,plq in enumerate(plaques):
        plq = list(plq)
        plaques[i] = plq
    for plq in plaques:
        i = plaques.index(plq)
        for j in range(0, len(plaques) - 1):
            if plaques[i] == plaques[j]:
                plaques.remove(plaques[j])
    return plaques


"""
Эта функция создает "маски" которые фильтруют все пиксели, оставляя только те, цвет которых
указан

"""
def apply_filter(img):
    #нужно перевести картинку в формат HLS
    image = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)



    # маска для белого цвета в формате HLS!!!
    # эти числа получаю через прогу set_filter
    #                H   S    V
    lower = np.uint8([0, 200, 0])
    upper = np.uint8([255, 255, 255])
    white_mask = cv2.inRange(image, lower, upper)

    #старый, но работающий вариант
    #маска для серого цвета
    lower = np.uint8([52,0,114])
    upper = np.uint8([152,96,220])
    gray_mask = cv2.inRange(image, lower, upper)


    lower = np.uint8([0, 0, 103])
    upper = np.uint8([255, 255, 255])
    super_gray_mask = cv2.inRange(image, lower, upper)

    """
    Теперь с помощью операции дизъюнкции мы объеденяем все маски в одну
    В итоге, при наложении этой маски на фотографии, отсеятся все пиксели, кроме
    белых пока что (смотри return)
    Именно этими цветами обычно рисуются полосы парковки и госномера
    """
    #mask = cv2.bitwise_or(white_mask, gray_mask)
    mask = super_gray_mask
    return mask



"""
Функция удаляет из массива контуров все с очень маленькой площадью
"""
def sort_by_area(image, cnts):
    width, height = image.shape[:2]
    new_cnts = []
    square = width * height
    min_area = 0.0002 * square # минимальная площадь контура как часть от общей
    for cnt in cnts:
        if cv2.contourArea(cnt) >= min_area:
            new_cnts.append(cnt)
        else:
            pass
    return new_cnts


"""
Функция для форматирования текста и удаление мусора из него
"""
def format_text(text):
    invalid_symbols = ["-", ":", ".", "'", "`", "|", "[", ']', "_", '"', "^", "/"]
    for i,symbol in enumerate(text):
        # если символ нижнего регистра, то переводим в верхний
        text = text.replace(symbol, symbol.upper())
        # если увидели пробел - убираем
        if symbol == " ":
            text = text.replace(symbol, "")
        # если в номере были распознаны какие-то символы, кроме букв и цифр, они отбрасываются
        if symbol in invalid_symbols:
            print("Removing invalid symbol: ", symbol)
            text = text.replace(symbol, "")
        # заменяем букву О на нули в цифрах
        if (i in range(1,4)) and (symbol == "О"):
            text = text.replace(symbol, "0")
    if len(text) > 9:
        text = text[:9]
    return text

"""
Основаня функция программы
Вход: путь к фото; индикатор того, надо ли показывать шаги работы
Выход: текст, найденный на фото
"""
def process(img_path, show_steps):
    try:
        path_to_text = None  # инициализируем пустыми значениеми
        text = None
        frame = cv2.imread(img_path)
        if frame is None:
            print("Wrong path.Try again!")
            exit()
        frame = imutils.resize(frame, width = 800)
        # убираем шумы с фото
        frame = cv2.bilateralFilter(frame, 11, 17, 17)

        # копия для рисования на ней
        for_number_rect = frame.copy()

        # переводим в серый цвет
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # 1.3 = scale, 5 = min_neighbours
        plaques = faceCascade.detectMultiScale(gray, 1.3, 5)

        #удаляем одинаковые плашки
        plaques = only_different(plaques)
        #ПОЧЕМУ ТО ЕСЛИ ЭТО СДЕЛАТЬ, ТО ПРОПАДАЕТ ГЛАВНЫЙ КОНТУР НОМЕРА ПРЯМОУГОЛЬНЫЙ

        # если был найден хотя бы один номер
        if plaques is not None:
            number = len(plaques)
            if number > 0:
                print("{} license plate detected".format(number))

            for plaque in plaques:
                (x,y,w,h) = plaque

                for i, (x, y, w, h) in enumerate(plaques):
                    roi_color = frame[y:y + h, x:x + w]

                    # находим ROI для номера
                    r = 300.0 / roi_color.shape[1]
                    dim = (300, int(roi_color.shape[0] * r))
                    resized = cv2.resize(roi_color, dim, interpolation=cv2.INTER_AREA)



                    # убираем шумы с вырезанного фото
                    no_noize = cv2.GaussianBlur(resized, ksize=(9, 9), sigmaX=0)

                    # переводим в серый формат и применяем цветовой фильтр
                    gray = cv2.cvtColor(no_noize, cv2.COLOR_BGR2GRAY)
                    gray = apply_filter(gray)
                    black_and_white = gray.copy()


                    edged = cv2.Canny(gray, 11, 17, 17)
                    for_lines = resized.copy()

                    # линии Хофа рисуются на цветной фотке
                    lines = cv2.HoughLinesP(edged, rho=1, theta=np.pi / 180, threshold=50, minLineLength=50,maxLineGap=5)


                    for line in lines:
                        x1, y1, x2, y2 = line[0]
                        cv2.line(for_lines, (x1, y1), (x2, y2), (0, 255, 0), 3)


                    # по линиям Хофа находится угол наклона кадра
                    angle = find_angle(edged, lines)


                    # ПОСЛЕ ТОГО, КАК НАШЛИ УГОЛ ПОВОРОТА НЕОБХОДИМО ЕЩЕ РАЗ СДЕЛАТЬ ЧЕРНО-БЕЛУЮ ФОТКУ НО УЖЕ
                    # БЕЗ РАЗМЫТИЯ ГАУССА
                    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
                    gray = apply_filter(gray)
                    black_and_white = gray.copy()


                    # поворачиваем вырезаннную цветную фотографию, чтобы на ней рисовать контуры и ВСЁ
                    rotated_color = rotate(resized, angle, True)




                    for_all_contours = rotated_color.copy()
                    for_sorted_contours = rotated_color.copy()

                    rotated = rotate(gray, angle, False)

                    # на повернутой черно-белой фотографии еще раз ищем грани и контуры
                    edged = cv2.Canny(rotated, 11, 17, 17)
                    # находим контуры среди граней
                    cnts, _ = cv2.findContours(edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_TC89_KCOS)
                    # а рисуем эти контуры на другой фотке
                    cv2.drawContours(for_all_contours, cnts, -1, (0,0,255), 3)


                    # фильтруем контуры по их площади от большего к меньшему
                    cnts = sorted(cnts, key=lambda cnt: cv2.contourArea(cnt), reverse=True)

                    #удаляем все контуры с очень маленькой площадью
                    cnts = sort_by_area(edged, cnts)


                    cv2.drawContours(for_sorted_contours, cnts, -1, (0,255,0), thickness=3)


                    min_area = 0
                    j = None

                    # находим контур с наибольшей площадью - контур номера
                    for i, cnt in enumerate(cnts):
                        rect = cv2.minAreaRect(cnt)
                        box = cv2.boxPoints(rect)  # поиск четырех вершин прямоугольника
                        box = np.intp(box)  # округление координат
                        area = int(rect[1][0] * rect[1][1])  # вычисление площади
                        if area > min_area:  # находим наибольший контур - там и будет номер
                            min_area = area
                            chosen = box
                            j = i
                    x, y, w, h = cv2.boundingRect(chosen)

                    rotated = rotated[y:y + h, x:x + w]




                    ret, thresh = cv2.threshold(rotated, 0, 255, cv2.THRESH_BINARY)
                    img_erode = cv2.erode(thresh, np.ones((3, 3), np.uint8), iterations=1)


                    today = datetime.datetime.today()
                    rndmplate = today.strftime("%H_%M_%S")
                    print(rndmplate)
                    cv2.imwrite(r"license_plates\notfilt\{}_text.jpg".format(rndmplate), rotated)
                    path_to_text = r"license_plates\notfilt\{}_text.jpg".format(rndmplate)
                    global path_plate
                    path_plate = path_to_text

                    text = tes.image_to_string(path_to_text, lang = "rus+eng")
                    print(text)


                    #если хотя бы 3 буквы нашли - хватит
                    if len(text) > 1:
                        if text is not None:
                            if len(text) > 1:
                                print("Unformatted result is: ", text)
                                text = format_text(text)

                        if (text != "") and (text != None):
                            print("\nRESULT: ", text)
                            cv2.imwrite(r"license_plates\{}.jpg".format(rndmplate), rotated_color)
                            normalplate = r"license_plates\{}.jpg".format(rndmplate)
                            cur.execute("INSERT INTO LOGS (date, date_hms, plate_number, plate_path) VALUES (?, ?, ?, ?)", (today.strftime("%Y-%d-%m") ,today.strftime("%H:%M:%S"), text, normalplate))
                            conn.commit()

                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                    else:
                        if len(plaques) > 1: #если плашка всего одна, то мы НЕ можем перейти к следующей
                            print("On this plaque text hasn`t been detected, moving to the next one!")
                        else:
                            print("\nOn this plaque text hasn`t been detected!")
                    
                   
        else:
            print("License plate hasn`t been detected!")
    except:
        pass
    return text

#############################################################################################################


show_steps = False
go = True

print("\nStart!")
cv2.destroyAllWindows()
