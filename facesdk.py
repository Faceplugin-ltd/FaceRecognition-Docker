import os

from ctypes import *
from numpy.ctypeslib import ndpointer
from face_util import ResultBox

libPath = os.path.abspath(os.path.dirname(__file__)) + '/libfacesdk.so'
facesdk = cdll.LoadLibrary(libPath)

getMachineCode = facesdk.Faceplugin_get_hardware_id
getMachineCode.argtypes = []
getMachineCode.restype = c_char_p

initSDK = facesdk.Faceplugin_init
initSDK.argtypes = [c_char_p]
initSDK.restype = c_int32

setActivation = facesdk.Faceplugin_activate
setActivation.argtypes = [c_char_p]
setActivation.restype = c_int32

faceDetection = facesdk.Faceplugin_detect
faceDetection.argtypes = [ndpointer(c_ubyte, flags='C_CONTIGUOUS'), c_int32, c_int32, c_int32, POINTER(ResultBox), c_int32]
faceDetection.restype = c_int32

faceRecognition = facesdk.Faceplugin_extract
faceRecognition.argtypes = [ndpointer(c_ubyte, flags='C_CONTIGUOUS'), c_int32, c_int32, c_int32, ResultBox, POINTER(c_float)]
faceRecognition.restype = c_int32

faceSimilarity = facesdk.Faceplugin_similarity
faceSimilarity.argtypes = [POINTER(c_float), POINTER(c_float)]
faceSimilarity.restype = c_float
