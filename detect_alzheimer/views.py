from django.shortcuts import render
from django.http import HttpResponse
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt

from django.conf import settings
from django.core.files.storage import FileSystemStorage

import numpy as np
import cv2
import os

from keras.preprocessing import image

from . import utils


def index(request):
    return render(request, "detect_alzheimer/home.html")

@csrf_exempt
def predict_json(request):
    labels = ['Alzheimer', 'Non Alzheimer']
    myfile = request.FILES['image']

    #upload image
    fs = FileSystemStorage()
    filename = fs.save(myfile.name, myfile)

    #preprocession
    path = 'media/' + filename
    img = cv2.imread(path)
    img = cv2.resize(img,(150,150))
    img = img[...,::-1]

    #prediction process
    img_name = myfile.name
    prediction = utils.predict(img)
    label = labels[np.argmax(prediction)]
    confidence = "%.2f" % (prediction[np.argmax(prediction)] * 100)
    alzheimer = "%.2f" % (prediction[0] * 100)
    nonalzheimer = "%.2f" % (prediction[1] * 100)
    

    
    os.remove(path)
    data = {
        'label' : label,
        'confidence': confidence,
        'alzheimer': alzheimer,
        'nonalzheimer': nonalzheimer,
        'filename' : img_name
    }

    return JsonResponse(data)


