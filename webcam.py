import cv2,os

cam = cv2.VideoCapture(0)
cam.set(3,640) #ubah lebar cam
cam.set(6,480)
faceDetector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eyeDetector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
faceDir = 'datawajah'
faceID = input("Masukan Face ID yang akan Direkam Datanya [Kemudian tekan enter: ")
print("Tatap Wajah anda ke depan dalam Webcam. Tunggu proses pengambilan data wajah selesai")
ambilData = 1
while True:
    retv,frame = cam.read()
    grey = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = faceDetector.detectMultiScale(grey,1.3,5) #frame , scaleFactor
    print(faces)

    for (x,y,w,h) in faces:
        frame = cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)
        nameFile = 'wajah.'+str(faceID)+'.'+str(ambilData)+ '.jpg'
        cv2.imwrite(faceDir+ '/' + nameFile,frame)
        ambilData += 1

        roiAbuAbu = grey[y:y+h,x:x+w]
        roiWarna = frame[y:y+h,x:x+w]
        eyes = eyeDetector.detectMultiScale(roiAbuAbu)
        for(xe,ye,we,he) in eyes:
            cv2.rectangle(roiWarna,(xe,ye),(xe+we,ye+he),(0,0,255),1)
    # cv2.imshow("Frame",grey)
    cv2.imshow("Webcam",frame)
    k = cv2.waitKey(1) & 0xFF
    if k == 27 or k == ord('q'):
        break
    elif ambilData>30 :
        break
print('Pengambilan Data')
cam.release()
cv2.destroyAllWindows()
