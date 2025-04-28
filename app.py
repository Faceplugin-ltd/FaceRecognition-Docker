import sys
sys.path.append('.')

import os
import numpy as np
import ctypes
import base64
import io

from PIL import Image
from flask import Flask, request, jsonify
from facesdk import faceDetection
from facesdk import faceRecognition
from facesdk import faceSimilarity
from facesdk import getMachineCode
from facesdk import setActivation
from facesdk import initSDK
from face_util import ResultBox

livenessThreshold = 0.7
yawThreshold = 10
pitchThreshold = 10
rollThreshold = 10
occlusionThreshold = 0.9
eyeClosureThreshold = 0.8
mouthOpeningThreshold = 0.5
borderRate = 0.05
smallFaceThreshold = 100
lowQualityThreshold = 0.3
hightQualityThreshold = 0.7
luminanceDarkThreshold = 50
luminanceLightThreshold = 200

maxFaceCount = 10

app = Flask(__name__)

is_activated = False

if os.path.exists("license.txt"):

    with open("license.txt", 'r') as file:
        license = file.read()
    
    ret = setActivation(license.encode('utf-8'))
    ret = initSDK("data".encode('utf-8'))

    is_activated = True


@app.route('/get-machine-code', methods=['GET'])
def get_machine_code():
    machine_code = getMachineCode()

    response = jsonify({"machineCode": machine_code.decode("utf-8")})

    response.status_code = 200
    response.headers["Content-Type"] = "application/json; charset=utf-8"
    return response


@app.route('/activate-machine', methods=['POST'])
def activate_machine():
    content = request.get_json()
    license = content['license']

    ret = setActivation(license.encode('utf-8'))
    activate_state = ret
    print("activation: ", ret)
    if ret == 0:
        with open("license.txt", 'w') as file:
            file.write(license)

    ret = initSDK("data".encode('utf-8'))
    init_state = ret
    print("init: ", ret)

    response = jsonify({"activationStatus": activate_state}, {'initStatus': init_state})

    response.status_code = 200
    response.headers["Content-Type"] = "application/json; charset=utf-8"
    return response


@app.route('/face_compare', methods=['POST'])
def match_face():
    faces = []
    isNotFront = None
    isOcclusion = None
    isEyeClosure = None
    isMouthOpening = None
    isBoundary = None
    isSmall = None
    quality = None
    luminance = None
    livenessScore = None

    file1 = request.files['file1']
    file2 = request.files['file2']
    threshold = float(request.form.get('threshold', 0.60))

    try:
        image1 = Image.open(file1)
        image2 = Image.open(file2)
    except:
        result = "Failed to open file"
        faceState = {"is_not_front": isNotFront, "is_boundary_face": isBoundary, "is_small": isSmall, "quality": quality, "result": result, "liveness_score": livenessScore}
        response = jsonify({"face_state": faceState, "faces": faces})

        response.status_code = 200
        response.headers["Content-Type"] = "application/json; charset=utf-8"
        return response


    image_np1 = np.asarray(image1)
    image_np2 = np.asarray(image2)

    faceBoxes = (ResultBox * maxFaceCount)()
    faceCount = faceDetection(image_np1, image_np1.shape[1], image_np1.shape[0], image_np1.shape[1] * 3, faceBoxes, maxFaceCount)

    max_area = 0
    max_area_idx = 0
    for i in range(faceCount):
        area = (faceBoxes[i].x2 - faceBoxes[i].x1) * (faceBoxes[i].y2 - faceBoxes[i].y1)
        if area > max_area:
            max_area = area
            max_area_idx = i

    if faceCount > 0:
        i = max_area_idx
        landmark_68 = []
        for j in range(68):
            landmark_68.append({"x": faceBoxes[i].landmark_68[j * 2], "y": faceBoxes[i].landmark_68[j * 2 + 1]})
        faces.append({'file1': {"x1": faceBoxes[i].x1, "y1": faceBoxes[i].y1, "x2": faceBoxes[i].x2, "y2": faceBoxes[i].y2,
                      "liveness": faceBoxes[i].liveness,
                      "yaw": faceBoxes[i].yaw, "roll": faceBoxes[i].roll, "pitch": faceBoxes[i].pitch, 
                      "face_quality": faceBoxes[i].face_quality, "eye_dist": faceBoxes[i].eye_dist,
                      "landmark_68": landmark_68}})

    feature1 = (ctypes.c_float * 512)()
    ret = faceRecognition(image_np1, image_np1.shape[1], image_np1.shape[0], image_np1.shape[1] * 3, faceBoxes[max_area_idx], feature1)
    if ret != 0:
        # faceState = {"is_not_front": isNotFront, "is_boundary_face": isBoundary, "is_small": isSmall, "quality": quality, "result": result, "liveness_score": livenessScore}
        faceState = {"similarity": 0, "status": None, "message": "Failed to extract feature on image1"}
        response = jsonify({"result": faceState})

        response.status_code = 200
        response.headers["Content-Type"] = "application/json; charset=utf-8"
        return response
    

    faceBoxes = (ResultBox * maxFaceCount)()
    faceCount = faceDetection(image_np2, image_np2.shape[1], image_np2.shape[0], image_np2.shape[1] * 3, faceBoxes, maxFaceCount)

    max_area = 0
    max_area_idx = 0
    for i in range(faceCount):
        area = (faceBoxes[i].x2 - faceBoxes[i].x1) * (faceBoxes[i].y2 - faceBoxes[i].y1)
        if area > max_area:
            max_area = area
            max_area_idx = i

    if faceCount > 0:
        i = max_area_idx
        landmark_68 = []
        for j in range(68):
            landmark_68.append({"x": faceBoxes[i].landmark_68[j * 2], "y": faceBoxes[i].landmark_68[j * 2 + 1]})
        faces.append({'file2': {"x1": faceBoxes[i].x1, "y1": faceBoxes[i].y1, "x2": faceBoxes[i].x2, "y2": faceBoxes[i].y2,
                      "liveness": faceBoxes[i].liveness,
                      "yaw": faceBoxes[i].yaw, "roll": faceBoxes[i].roll, "pitch": faceBoxes[i].pitch, 
                      "face_quality": faceBoxes[i].face_quality, "eye_dist": faceBoxes[i].eye_dist,
                      "landmark_68": landmark_68}})

    feature2 = (ctypes.c_float * 512)()
    ret = faceRecognition(image_np2, image_np2.shape[1], image_np2.shape[0], image_np2.shape[1] * 3, faceBoxes[max_area_idx], feature2)
    if ret != 0:
        # faceState = {"is_not_front": isNotFront, "is_boundary_face": isBoundary, "is_small": isSmall, "quality": quality, "result": result, "liveness_score": livenessScore}
        faceState = {"similarity": 0, "status": None, "message": "Failed to extract feature on image2"}
        response = jsonify({"result": faceState})

        response.status_code = 200
        response.headers["Content-Type"] = "application/json; charset=utf-8"
        return response

    similarity = faceSimilarity(feature1, feature2)

    if similarity > threshold:
        result = "Same Person"
    else:
        result = "Different Person"

    faceState = {"similarity": similarity, "status": result, "message": "Success"}
    response = jsonify({"result": faceState})

    response.status_code = 200
    response.headers["Content-Type"] = "application/json; charset=utf-8"
    return response

@app.route('/face_compare_base64', methods=['POST'])
def match_face_base64():
    faces = []
    isNotFront = None
    isOcclusion = None
    isEyeClosure = None
    isMouthOpening = None
    isBoundary = None
    isSmall = None
    quality = None
    luminance = None
    livenessScore = None

    content = request.get_json()
    threshold = 0.6

    try:
        image1Base64 = content['file1']
        image_data1 = base64.b64decode(image1Base64)
        image1 = Image.open(io.BytesIO(image_data1))

        image2Base64 = content['file2']
        image_data2 = base64.b64decode(image2Base64)
        image2 = Image.open(io.BytesIO(image_data2))
    except:
        result = "Failed to open file"
        faceState = {"is_not_front": isNotFront, "is_boundary_face": isBoundary, "is_small": isSmall, "quality": quality, "result": result, "liveness_score": livenessScore}
        response = jsonify({"face_state": faceState, "faces": faces})

        response.status_code = 200
        response.headers["Content-Type"] = "application/json; charset=utf-8"
        return response


    image_np1 = np.asarray(image1)
    image_np2 = np.asarray(image2)

    faceBoxes = (ResultBox * maxFaceCount)()
    faceCount = faceDetection(image_np1, image_np1.shape[1], image_np1.shape[0], image_np1.shape[1] * 3, faceBoxes, maxFaceCount)

    max_area = 0
    max_area_idx = 0
    for i in range(faceCount):
        area = (faceBoxes[i].x2 - faceBoxes[i].x1) * (faceBoxes[i].y2 - faceBoxes[i].y1)
        if area > max_area:
            max_area = area
            max_area_idx = i

    if faceCount > 0:
        i = max_area_idx
        landmark_68 = []
        for j in range(68):
            landmark_68.append({"x": faceBoxes[i].landmark_68[j * 2], "y": faceBoxes[i].landmark_68[j * 2 + 1]})
        faces.append({'file1': {"x1": faceBoxes[i].x1, "y1": faceBoxes[i].y1, "x2": faceBoxes[i].x2, "y2": faceBoxes[i].y2,
                      "liveness": faceBoxes[i].liveness,
                      "yaw": faceBoxes[i].yaw, "roll": faceBoxes[i].roll, "pitch": faceBoxes[i].pitch, 
                      "face_quality": faceBoxes[i].face_quality, "eye_dist": faceBoxes[i].eye_dist,
                      "landmark_68": landmark_68}})

    feature1 = (ctypes.c_float * 512)()
    ret = faceRecognition(image_np1, image_np1.shape[1], image_np1.shape[0], image_np1.shape[1] * 3, faceBoxes[max_area_idx], feature1)
    if ret != 0:
        # faceState = {"is_not_front": isNotFront, "is_boundary_face": isBoundary, "is_small": isSmall, "quality": quality, "result": result, "liveness_score": livenessScore}
        faceState = {"similarity": 0, "status": None, "message": "Failed to extract feature on image1"}
        response = jsonify({"result": faceState})

        response.status_code = 200
        response.headers["Content-Type"] = "application/json; charset=utf-8"
        return response
    

    faceBoxes = (ResultBox * maxFaceCount)()
    faceCount = faceDetection(image_np2, image_np2.shape[1], image_np2.shape[0], image_np2.shape[1] * 3, faceBoxes, maxFaceCount)

    max_area = 0
    max_area_idx = 0
    for i in range(faceCount):
        area = (faceBoxes[i].x2 - faceBoxes[i].x1) * (faceBoxes[i].y2 - faceBoxes[i].y1)
        if area > max_area:
            max_area = area
            max_area_idx = i

    if faceCount > 0:
        i = max_area_idx
        landmark_68 = []
        for j in range(68):
            landmark_68.append({"x": faceBoxes[i].landmark_68[j * 2], "y": faceBoxes[i].landmark_68[j * 2 + 1]})
        faces.append({'file2': {"x1": faceBoxes[i].x1, "y1": faceBoxes[i].y1, "x2": faceBoxes[i].x2, "y2": faceBoxes[i].y2,
                      "liveness": faceBoxes[i].liveness,
                      "yaw": faceBoxes[i].yaw, "roll": faceBoxes[i].roll, "pitch": faceBoxes[i].pitch, 
                      "face_quality": faceBoxes[i].face_quality, "eye_dist": faceBoxes[i].eye_dist,
                      "landmark_68": landmark_68}})

    feature2 = (ctypes.c_float * 512)()
    ret = faceRecognition(image_np2, image_np2.shape[1], image_np2.shape[0], image_np2.shape[1] * 3, faceBoxes[max_area_idx], feature2)
    if ret != 0:
        # faceState = {"is_not_front": isNotFront, "is_boundary_face": isBoundary, "is_small": isSmall, "quality": quality, "result": result, "liveness_score": livenessScore}
        faceState = {"similarity": 0, "status": None, "message": "Failed to extract feature on image2"}
        response = jsonify({"result": faceState})

        response.status_code = 200
        response.headers["Content-Type"] = "application/json; charset=utf-8"
        return response

    similarity = faceSimilarity(feature1, feature2)

    if similarity > threshold:
        result = "Same Person"
    else:
        result = "Different Person"

    faceState = {"similarity": similarity, "status": result, "message": "Success"}
    response = jsonify({"result": faceState})

    response.status_code = 200
    response.headers["Content-Type"] = "application/json; charset=utf-8"
    return response

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8888))
    app.run(host='0.0.0.0', port=port)