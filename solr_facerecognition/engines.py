from django.conf import settings
from sorl.thumbnail.engines.pil_engine import Engine
import cv2.cv as cv
import os

class FaceRecognitionEngine(Engine):

    def create(self, image, geometry, options):
        image,options = self.facerecognition(image, geometry, options)
        image = super(FaceRecognitionEngine, self).create(image, geometry, options)
        return image

    def facerecognition(self, image, geometry, options):
        if 'facerecognition' in options:
            img = cv.CreateImageHeader(image.size, cv.IPL_DEPTH_8U, 3)
            cv.SetData(img, image.tostring())
            grayscale = cv.CreateImage((img.width, img.height), 8, 1)
            cv.CvtColor(img, grayscale, cv.CV_BGR2GRAY)

            storage = cv.CreateMemStorage(0)
            cv.EqualizeHist(grayscale, grayscale)

            cascade = cv.Load(os.path.realpath(__file__)+"/haarcascade_frontalface_alt.xml")
            faces = cv.HaarDetectObjects(grayscale, cascade, storage, 1.2, 2, cv.CV_HAAR_DO_CANNY_PRUNING, (50,50))
            if len(faces):
                face = faces[0]
                a = face[0]
                x1,y1,x2,y2=a[0],a[1],a[0]+a[2],a[1]+a[3]
                x_per = int((x1/float(img.width))*100)
                y_per = int((y1/float(img.height))*100)
                options['crop'] = str(x_per)+"% "+str(y_per)+"%"
        return image,options
