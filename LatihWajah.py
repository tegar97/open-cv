import cv2,os,numpy as np
from PIL import Image
faceDir = 'datawajah'
latihDir = 'latihWajah'
def getImageLabel(path):
    imagePaths = [os.path.join(path,f) for f in os.listdir(path)]
    faceSamples = []
    faceIDs = []
    for imagePaths in  imagePaths :
        PILImg = Image.open(imagePaths).convert('L') #convert ke dalam grey
        imgNum = np.array(PILImg,'uint8')


        faceID = int(os.path.split(imagePaths)[-1].split(".")[1])

        faces = faceDetector.detectMultiScale(imgNum)
        for(x,y,w,h) in faces:
            faceSamples.append(imgNum[y:y+h,x:x+w])
            faceIDs.append(faceID)
        return faceSamples,faceIDs
faceRecognizer = cv2.face.LBPHFaceRecognizer_create()
faceDetector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

print('Mesin sedang melakukan training data wajah,Tunggu dalam beberapa detik')
faces,IDs = getImageLabel(faceDir)
faceRecognizer.train(faces,np.array(IDs))

faceRecognizer.write(latihDir+'/training.xml')
print('Sebanyak {0} data wajah telah di trainingkan ke mesin',format(len(np.unique(IDs))))
