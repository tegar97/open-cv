import cv2,os,numpy as np
faceDir = 'datawajah'
latihDir = 'latihwajah'

cam = cv2.VideoCapture(0)
cam.set(3,640) #ubah lebar cam
cam.set(4,480)
faceRecognizer = cv2.face.LBPHFaceRecognizer_create()
faceDetector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


faceRecognizer.read(latihDir+'/training.xml')
font = cv2.FONT_HERSHEY_COMPLEX

id = 0
names = ['Tidak Diketahui','Tegar',"tegar2",'Other']

minWidth = 0.1*cam.get(3)
minHeight = 0.1*cam.get(4)

while True:
    retv,frame = cam.read()
    frame = cv2.flip(frame,1) #vertical flio

    grey = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = faceDetector.detectMultiScale(grey,1.2,5,minSize=(round(minWidth),round(minHeight)),) #frame , scaleFactor
    print(faces)
    for (x,y,w,h) in faces:
        frame = cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)
        print(frame)
        id,confident = faceRecognizer.predict(grey[y:y+h,x:x+w]) #confident = 0 cocok sempurna
        if confident<=50:
            nameId= names[id]
            confidentTxt = "{0}%".format(round(100-confident))
        else:
            nameId = names[id]
            confidentTxt = "{0}%".format(round(100-confident))
        cv2.putText(frame,str(nameId),(x+5,y-5),font,1,(255,255,255),2)
        cv2.putText(frame,str(confidentTxt),(x+5,y+h-5),font,1,(255,255,255),2)

    # cv2.imshow("Frame",grey)
    cv2.imshow("Webcam",frame)
    k = cv2.waitKey(1) & 0xFF
    if k == 27 or k == ord('q'):
        break
print('Exit')
cam.release()
cv2.destroyAllWindows()

