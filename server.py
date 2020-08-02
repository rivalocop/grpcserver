import io
import logging
import pickle
from concurrent import futures
from typing import List

import cv2
import face_recognition
import grpc
import numpy as np
import tensorflow as tf
from PIL import Image
from pydantic.main import BaseModel

import motion_pb2
import motion_pb2_grpc
import settings
from face_detector.face_detect import ImFace
from motion_detector.motion_detect import ImMotion
from repository.singleton import faces

print("[INFO] loading face detector...")
net = cv2.dnn.readNetFromCaffe(settings.DEPLOY_FILE, settings.CAFFE_MODEL)
print("[INFO] loading face detector done...")
print("[INFO] loading motion detector...")
model = tf.keras.models.load_model(settings.MOTION)
le = pickle.loads(open(settings.LABELS, "rb").read())


class User(BaseModel):
    userId: str
    isVerify: bool = False
    encodings: List[float] = []


def preprocess_image(image):
    with io.BytesIO(image) as f:
        # Load it as numpy array
        image = np.array(Image.open(f))
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        image = cv2.resize(image, (300, 300))
        return image


def predict_process(image, expected_label):
    result = {
        'result': False,
        'encoding': None
    }
    with io.BytesIO(image) as f:
        # Load it as numpy array
        image = face_recognition.load_image_file(f)
        # Check if image has face or not
        box = face_recognition.face_locations(image)
        if len(box) > 0:
            # Motion Detection
            extracted_face = ImFace(image, net).face
            motion = ImMotion(extracted_face, le,
                              model, expected_label)
            if motion.result:
                # If motion detection passed then extract face encoding for register face indexes
                encoding = face_recognition.face_encodings(image, box)
                result['result'] = True
                result['encoding'] = encoding[0]
        return result


def get_face_indexes(user_id):
    face_indexes = faces.find_one({'user_id': user_id})
    return face_indexes['encodings']


class User(BaseModel):
    userId: str
    isVerify: bool = False


def preprocess_image(image):
    with io.BytesIO(image) as f:
        # Load it as numpy array
        image = np.array(Image.open(f))
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        image = cv2.resize(image, (300, 300))
        return image


def predict_process(image, expected_label):
    result = {
        'result': False,
        'encoding': None
    }
    with io.BytesIO(image) as f:
        # Load it as numpy array
        image = face_recognition.load_image_file(f)
        # Check if image has face or not
        box = face_recognition.face_locations(image)
        if len(box) > 0:
            # Motion Detection
            extracted_face = ImFace(image, net).face
            motion = ImMotion(extracted_face, le,
                              model, expected_label)
            if motion.result:
                # If motion detection passed then extract face encoding for register face indexes
                encoding = face_recognition.face_encodings(image, box)
                result['result'] = True
                result['encoding'] = encoding[0]
        return result


class MotionServicer(motion_pb2_grpc.MotionServicer):
    def MotionStreaming(self, request_iterator, context):
        face_embeddings = []
        result = False
        user_id = None
        for ri in request_iterator:
            if not face_embeddings:
                face_embeddings = np.array(
                    get_face_indexes(ri.userInfo.userId))
                user_id = ri.userInfo.userId
            # Check face is existed:
            image = preprocess_image(ri.imagePayload)
            box = face_recognition.face_locations(image)
            if len(box) > 0:
                # Handle Registration
                extracted_face = ImFace(image, net).face
                motion = ImMotion(extracted_face, le,
                                  model)
                if motion.result['label'] == ri.expectedLabel and motion.result['confidence'] > 0.5:
                    encoding = face_recognition.face_encodings(image, box)
                    compare_result = face_recognition.compare_faces(
                        face_embeddings, encoding[0])
                    if any(compare_result):
                        face_embeddings.append(encoding[0])
                        result = True
            yield motion_pb2.MotionResponse(result=result)
        if len(face_embeddings) > 0:
            encodings = map(lambda x: x.tolist(), face_embeddings)
            faces.update_one({'user_id': user_id}, {
                             '$set': {'encodings': encodings}})

    def RegisterFaceIndexes(self, request, context):
        registered_user = User(userId=request.userId)
        faces.create(registered_user.dict())
        return motion_pb2.UserFormData(userId=registered_user.userId, isFaceVerify=registered_user.isVerify)


def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    motion_pb2_grpc.add_MotionServicer_to_server(MotionServicer(), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    server.wait_for_termination()


if __name__ == "__main__":
    logging.basicConfig()
    print("[INFO] Starting server...")
    serve()
