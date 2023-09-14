import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import datetime
import pandas as pd

#duygusal analiz modelinin yüklenmesi
model_path = "emotion_model.hdf5"
model = tf.keras.models.load_model(model_path)

#duygusal etiketler
emotions = ["Angry", "Disgust", "Fear", "Surprise", "Sad", "Happy", "Neutral"]

#duygu sayaçları
emotion_counts = {emotion: 0 for emotion in emotions}

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier("haarcascade_eye.xml")


#başlangıç kordinatları
prev_face_x = None
prev_face_y = None
prev_eye_x = None
prev_eye_y = None
prev_hand_pos = None


face_movement_counts = {
    "Sola":0,
    "Saga":0,
    "Yukari":0,
    "Asagi":0
}

eye_movement_counts = {
    "Sola": 0,
    "Saga": 0,
    "Yukari": 0,
    "Asagi": 0
}

frame_counter = 0
hand_count = 0
table_data = []

def format_table(table):
    formatted_table = ""
    for col in table.columns:
        formatted_table+= col+"\n"
        formatted_table+= "-" * len(col) + "\n"

        if isinstance(table[col][0], dict):
            for subcol in table[col][0]:
                formatted_table += subcol + "\n"
                formatted_table += "-" * len(subcol) +"\n"

                for idx, row in table.iterrows():
                    formatted_table += str(row[col][subcol]) +"\n"
                formatted_table += "-" * len(subcol)+ "\n"
        else:
            for idx, row in table.iterrows():
                formatted_table += str(row[col]) +"\n"
            formatted_table += "-" * len(col) +"\n"

        formatted_table += "\n"


    return formatted_table


video_path = "videoplayback (2).mp4"
video = cv2.VideoCapture(video_path)


is_video_finished =False

while True:
    ret, frame = video.read()
    if not ret:
        is_video_finished = True
        break

    frame = cv2.resize(frame, (900, 500))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #el için ayarlamalar
    imageRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(imageRGB)
    hand_pos = None

    #elin tespit ediliip edilmdiğine bak
    if result.multi_hand_landmarks:
        for handLms in result.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark):
                h, w, c = frame.shape
                cx, cy = int(lm.x * w), int(lm.y * h)

            mpDraw.draw_landmarks(frame, handLms, mpHands.HAND_CONNECTIONS)

            #elin orta noktasını belile
            hand_pos = (cx, cy)

    #elin pozisyonu değiştiyse ve önceki pozisyon varsa
    if hand_pos and prev_hand_pos:
        #elin pozisyon değişimini hesapla
        dx = hand_pos[0] - prev_hand_pos[0]
        dy = hand_pos[1] - prev_hand_pos[1]

        #elin pozisyon değişimi sınırlarını belirle
        movement_threshold = 20

        #elin pozisyon değişimi sınırlarını aştıysa
        if abs(dx) > movement_threshold or abs(dy) >movement_threshold:
            hand_count += 1

    prev_hand_pos = hand_pos

    #yüz algılama
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    #algılanan yüz üzerinde döngü
    for(x, y, w, h) in faces:

        #duygusal analiz hazırlık
        detected_faces = frame[int(y):int(y+h):int(x+w)]
        detected_faces = cv2.cvtColor(detected_faces, cv2.COLOR_BGR2GRAY)
        detected_faces = cv2.resize(detected_faces, (64, 64))
        detected_faces = detected_faces.reshape(1, 64, 64, 1).astype("float32") / 255.0
        #duygusal analiz yapma
        emotion_id = model.predict(detected_faces, verbose=0).argmax()
        emotion_label = emotions[emotion_id]
        cv2.putText(frame, emotion_label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        #duygu sayma
        emotion_counts[emotion_label] += 1

        #kafa bölgesini yeşile boyama
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)


        #göz algılama
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.3, 5)

        #kafa kordinatlarını hesapla
        center_x = x+w // 2
        center_y = y+h // 2

        #ilk çerçeve için başlangıç kordinatlarına ata
        if prev_face_x is None or prev_face_y is None:
            prev_face_x = center_x
            prev_face_y =center_y
            continue

        #kafa hareketlerinin yönünü belirleme
        if center_x >prev_face_x:
            face_direction_x = "Saga"
        elif center_x <prev_face_x:
            face_direction_x = "Sola"
        else:
            face_direction_x =""

        if center_y > prev_face_y:
            face_direction_y = "Asagi"
        elif center_y < prev_face_y:
            face_direction_y = "Yukari"
        else:
            face_direction_y = ""

        #kafa hareket sayılarını güncelle
        if face_direction_x:
            face_movement_counts[face_direction_x] += 1
        if face_direction_y:
            face_movement_counts[face_direction_y] += 1

        #kafa kordinatlarını güncelle
        prev_face_x = center_x
        prev_face_y =center_y

        #algolanan gözler üzerinde döngü
        for(ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (255, 0, 0), 2)

            #gözlerin kordinatlarını al
            eye_center_x = x +ex +ew // 2
            eye_center_y = y+ ey + eh // 2


            #ilk çerçeve için başlangıç kordinatlarını al
            if prev_eye_x is None or prev_eye_y is None:
                prev_eye_x = eye_center_x
                prev_eye_y = eye_center_y
                continue

            #göz hareketlerini belirleme
            if eye_center_x > prev_eye_x:
                eye_direction_x = "Saga"
            elif eye_center_x < prev_eye_x:
                eye_direction_x = "Sola"
            else:
                eye_direction_x = ""

            if eye_center_y > prev_eye_y:
                eye_direction_y = "Asagi"
            elif eye_center_y < prev_eye_y:
                eye_direction_y = "Yukari"
            else:
                eye_direction_y = ""

            #göz hareket sayılarını güncelle
            if eye_direction_x:
                eye_movement_counts[eye_direction_x] +=1
            if eye_direction_y:
                eye_movement_counts[eye_direction_y] +=1

            #göz kordinarlarını güncelle
            prev_eye_x = eye_center_x
            prev_eye_y = eye_center_y

    cv2.imshow("frame", frame)

    #duygu oranlarını hesaplama
    total_count = sum(emotion_counts.values())
    emotion_ratios = {emotion: count * 100/total_count for emotion, count in emotion_counts.items()}

    current_time = datetime.datetime.now().strftime("%Y -%m- %d %H:%M:%S")

    #tablo oluşturma ve log işlemleri
    if frame_counter % 200 ==0:
        table_data.append({
            "Time": current_time,
            "Face Movemoet Counts": {
                "Sola": face_movement_counts["Sola"],
                "Saga": face_movement_counts["Saga"],
                "Yukari": face_movement_counts["Yukari"],
                "Asagi": face_movement_counts["Asagi"]
            },
            "Eye Movement Count": {
                "Sola": eye_movement_counts["Sola"],
                "Saga": eye_movement_counts["Saga"],
                "Yukari": eye_movement_counts["Yukari"],
                "Asagi": eye_movement_counts["Asagi"]
            },
            "Hand Movement Count": {
                "Hand Movement": hand_count
            },
            "Emotion Ratios": emotion_ratios
        })
        df = pd.DataFrame(table_data)
        print(format_table(df))

    frame_counter += 1

    k = cv2.waitKey(30) & 0xff

    if k == 27:
        break

cv2.destroyAllWindows()
video.release()

#log yazdırma
if is_video_finished:
    #kafa hareket log dosyasını oluştur ve yaz
    with open("face_movement_log.txt", "w") as face_log_file:
        face_log_file.write("Kişinin kafa hareketi bilgileri:\n")
        for direction, count in face_movement_counts.items():
            face_log_file.write(f"{direction}: {count}\n")

    with open("eye_movement_log.txt", "w") as eye_log_file:
        eye_log_file.write("Kişinin göz hareketi bilgileri:\n")
        for direction, count in eye_movement_counts.items():
            eye_log_file.write(f"{direction}: {count}\n")



    with open("hand_counter_log.txt", "w") as hand_log_file:
        hand_log_file.write("kişinin el ve kol hareket bigileri:\n")
        hand_log_file.write(str(hand_count))


    with open("duygu_analizi_log.txt", "w") as duygu_analizi_file:
        duygu_analizi_file.write("kişinin duygu analizi:\n")
        for emotion, count in emotion_counts.items():
            duygu_analizi_file.write(f"{emotion}: {count}\n")


#bar plot görselleştirme
plt.bar(emotion, emotion_ratios.values())
plt.xlabel('Duygu')
plt.ylabel('Oran')
plt.title('kişinin video boyunca algılanan duygu oranları')
plt.show()


















