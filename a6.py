
#koleksiyonlardan ithalat deque
import numpy as np
import argparse
import imutils
import cv2

#URL’den resim okumak için urllib


# # construct argümanı arse ayrıştırma ve argüman ayrıştırma
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video",
                help="path to the (optional) video file")
ap.add_argument("-b", "--buffer", type=int, default=64,
                help="max buffer size")
args = vars(ap.parse_args())

# HSV renk uzayındaki renklerin alt ve üst sınırlarını tanımlar

lower = {'red': (136, 87, 111), 'green': (66, 122, 129), 'blue': (97, 100, 117), 'yellow': (23, 59, 119)
         }  # assign new item lower['blue'] = (93, 10, 0)
upper = {'red': (180, 255, 255), 'green': (34,139,34), 'blue': (117, 255, 255), 'yellow': (54, 255, 255)
         }


# nesnenin etrafındaki daire için standart renkleri tanımlayın
colors = {'red': (0, 0, 255), 'green': (0, 255, 0), 'blue': (255, 0, 0), 'yellow': (0, 255, 217),
          'orange': (0, 140, 255)}


# pts = deque (maxlen = args ["buffer"])

# Bir video yolu sağlanmadıysa referansı alın
# web kamerasına
if not args.get("video", False):
    camera = cv2.VideoCapture(0)


# başka türlü, video dosyasına referans alın
else:
    camera = cv2.VideoCapture(args["video"])

# döngü devam
while True:

    # mevcut kareyi yakala
    (grabbed, frame) = camera.read()
    # eğer bir video görüntülüyorsak ve bir çerçeve almadık
    # sonra videonun sonuna ulaştık
    if args.get("video") and not grabbed:
        break

    # IP webcam image stream
    # URL = 'http://10.254.254.102:8080/shot.jpg'
    # urllib.urlretrieve(URL, 'shot1.jpg')
    # frame = cv2.imread('shot1.jpg')

    # çerçeveyi yeniden boyutlandırın, bulanıklaştırın ve HSV'ye dönüştürün
    # renk alanı
    frame = imutils.resize(frame, width=600)

    blurred = cv2.GaussianBlur(frame, (11, 11), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    # sözlükteki her renk için # çerçevesindeki nesneyi denetleyin
    for key, value in upper.items():

        # sözlüğe `1 renginden renk için bir maske oluşturun, sonra gerçekleştirin
        # Herhangi bir küçük kaldırmak için bir dizi dilatasyon ve erozyon
        # maskede kalan lekeler

        kernel = np.ones((9, 9), np.uint8)
        mask = cv2.inRange(hsv, lower[key], upper[key])
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # maskedeki kontürleri bul ve akımı başlat
        # (x, y) topun merkezi
        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)[-2]
        center = None

        # sadece en az bir kontur bulunduğunda devam et

        if len(cnts) > 0:

            # maskedeki en büyük konturu bul, sonra kullan
            # minimum çevreleme dairesini hesaplamak için ve
            # centroid
            c = max(cnts, key=cv2.contourArea)
            ((x, y), radius) = cv2.minEnclosingCircle(c)
            M = cv2.moments(c)
            center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

            # sadece yarıçap minimum boyuttaysa devam edin. Böceğinizin büyüklüğü için bu değeri düzeltin
            if radius > 0.5:
                # çerçeveye çember ve centroid çizin,
                # sonra izlenen noktaların listesini güncelle
                cv2.circle(frame, (int(x), int(y)), int(radius), colors[key], 2)
                cv2.putText(frame, key + " ball", (int(x - radius), int(y - radius)), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            colors[key], 2)

    # çerçeveyi ekranımıza göster
    cv2.imshow("Frame", frame)

    key = cv2.waitKey(1) & 0xFF
    # 'q' tuşuna basıldığında döngüyü durdurun
    if key == ord("q"):
        break

# kamerayı temizle ve açık pencereleri kapat
camera.release()
cv2.destroyAllWindows()