import cv2

# Muat model deteksi wajah
face_cascade = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')

# Muat model deteksi jenis kelamin
gender_net = cv2.dnn.readNetFromCaffe('data/deploy_gender.prototxt', 'data/gender_net.caffemodel')

# Muat model deteksi usia
age_net = cv2.dnn.readNetFromCaffe('data/deploy_age.prototxt', 'data/age_net.caffemodel')

# Kelas gender dan usia
GENDER_LIST = ['Pria', 'Wanita']
AGE_LIST = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']

# Buka video capture (webcam)
cap = cv2.VideoCapture(0)

while True:
    # Baca frame dari webcam
    ret, frame = cap.read()
    
    # Ubah ke dalam skala abu-abu
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Deteksi wajah
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    
    for (x, y, w, h) in faces:
        # Gambar persegi di sekitar wajah
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        # Ekstrak wajah
        face = frame[y:y+h, x:x+w]
        blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), (104.0, 177.0, 123.0), swapRB=False, crop=False)
        
        # Prediksi gender
        gender_net.setInput(blob)
        gender_preds = gender_net.forward()
        gender = GENDER_LIST[gender_preds[0].argmax()]
        
        # Prediksi usia
        age_net.setInput(blob)
        age_preds = age_net.forward()
        age = AGE_LIST[age_preds[0].argmax()]
        
        # Tampilkan hasil prediksi gender dan usia
        label = f'{gender}, {age}'
        cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    
    # Tampilkan frame
    cv2.imshow('Gender and Age Detection', frame)
    
    # Break loop saat 'q' ditekan
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Bersihkan segala sesuatunya
cap.release()
cv2.destroyAllWindows()
