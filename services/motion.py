import json
import pickle
import time
import logging
from datetime import datetime

import cv2
import face_recognition
import numpy as np
import tensorflow as tf
from bson import ObjectId

import motion_pb2
import motion_pb2_grpc
import settings
from entities.grpc_models import User, RecentActivity, FaceIndexes
from face_detector.face_detect import ImFace
from motion_detector.motion_detect import ImMotion
from repository.singleton import faces, users, activities, redis_db
from utils.image_utils import preprocess_raw_image, preprocess_png_image

print("[INFO] loading face detector...")
net = cv2.dnn.readNetFromCaffe(settings.DEPLOY_FILE, settings.CAFFE_MODEL)
print("[INFO] loading face detector done...")
print("[INFO] loading motion detector...")
model = tf.keras.models.load_model(settings.MOTION)
le = pickle.loads(open(settings.LABELS, "rb").read())


def get_face_indexes(user_id):
    face_indexes = faces.find({'userId': user_id})
    return [doc["encoding"] for doc in face_indexes]


def predict_process(image, location):
    try:
        extracted_face = ImFace(image, net).face
        motion = ImMotion(extracted_face, le,
                          model)
        encodings = face_recognition.face_encodings(image, location)
        new_embedding = FaceIndexes(
            confidence=motion.result['confidence'],
            label=motion.result['label'],
            encoding=encodings[0].tolist())
        return new_embedding
    except Exception as err:
        print(err)


class MotionServicer(motion_pb2_grpc.MotionServicer):
    def MotionStreaming(self, request_iterator, context):
        for ri in request_iterator:
            if ri.is_ping_msg:
                print("Pinged")
                yield motion_pb2.MotionResponse(is_pong_msg=True)
            else:
                returned_model = motion_pb2.MotionResponse(
                    result=False, confidence=0.0, id='None')
                try:
                    image = preprocess_raw_image(ri.imagePayload)
                    box = face_recognition.face_locations(image)
                    if len(box) > 0:
                        result = predict_process(image, box)
                        print(
                            f"predict {ri.expectedLabel} got {result.label} with confidence: {result.confidence}")
                        if result.label == ri.expectedLabel and result.confidence > 0.5:
                            encoding_id = faces.insert_one(
                                result.dict()).inserted_id
                            returned_model = motion_pb2.MotionResponse(
                                result=True, confidence=result.confidence, id=str(encoding_id))
                except Exception as e:
                    print(e)
                finally:
                    time.sleep(0.5)
                    print('predicted')
                    yield returned_model

    # create user information when register a face
    def RegisterFaceIndexes(self, request, context):
        registered_user = User(userId=request.user_id)
        users.insert_one(registered_user.dict())
        return motion_pb2.UserFormData(user_id=registered_user.userId)

    # update face embedding list for current user
    def UpdateFaceIndexes(self, request, context):
        returned_model = motion_pb2.FaceIndexesResponse(is_success=False)
        try:
            if len(request.imageIds) > 0:
                for i in request.image_ids:
                    faces.update_one({'_id': ObjectId(i)}, {
                        '$set': {'userId': request.user_inf.user_id}})
                new_activity = RecentActivity(
                    isSuccess=True,
                    title='Update face indexes',
                    content='Update face indexes from main account',
                    causeId=request.user_inf.user_id
                )
                activity_id = activities.insert_one(
                    new_activity.dict()).inserted_id
                returned_model = motion_pb2.ActivityRecent(
                    is_success=new_activity.isSuccess,
                    activity_id=str(activity_id),
                    title=new_activity.title,
                    content=new_activity.content,
                    created_time=new_activity.createdTime.strftime(
                        "%Y-%m-%d %H:%M:%S"),
                    modified_time=new_activity.modifiedTime.strftime(
                        "%Y-%m-%d %H:%M:%S")
                )
        except Exception as err:
            print(err)
        finally:
            return returned_model

    # motion detection + face recognition processing
    def FaceRecognizeStreaming(self, request_iterator, context):
        face_embedding = []
        for ri in request_iterator:
            if ri.is_ping_msg:
                face_embedding = get_face_indexes(ri.user_id)
                logging.info('Pinged from client')
                yield motion_pb2.MotionResponse(is_pong_msg=True)
                logging.info('Ponged back to client')
            else:
                returned_model = motion_pb2.MotionResponse(
                    result=False, confidence=0.0, id='None')
                try:
                    # image = preprocess_raw_image(ri.image_payload)
                    image = preprocess_png_image(ri.image_payload)
                    box = face_recognition.face_locations(image)
                    if len(box) > 0:
                        result = predict_process(image, box)
                        print(
                            f"predict {ri.expected_label} got {result.label} with confidence: {result.confidence}")
                        if result.label == ri.expected_label and result.confidence > 0.5:
                            unknown_face_encoding = np.array(result.encoding)
                            match_results = face_recognition.compare_faces(
                                face_embedding, unknown_face_encoding)
                            if match_results[0]:
                                returned_model = motion_pb2.MotionResponse(
                                    result=True, confidence=result.confidence)
                except Exception as err:
                    logging.exception(err)
                finally:
                    yield returned_model

    # request face recognize from external system
    def RequireFaceRecognizeRequest(self, request, context):
        new_activity = RecentActivity(
            title=request.titleRequest,
            content=request.contentRequest,
            causeId=request.userId
        )
        activity_id = activities.insert_one(new_activity.dict()).inserted_id
        while True:
            output = redis_db.get(str(activity_id))
            if output is not None:
                output = output.decode("utf-8")
                output = json.loads(output)
                if output['isSuccess']:
                    is_recognized = True
                else:
                    is_recognized = False
                redis_db.delete(str(activity_id))
                break
        if is_recognized:
            return motion_pb2.FaceRecognizeResponse(activity_id=str(activity_id),
                                                    result=motion_pb2.FaceRecognizeResponse.Result.RESULT_SUCCESS)
        else:
            return motion_pb2.FaceRecognizeResponse(activity_id=str(activity_id),
                                                    result=motion_pb2.FaceRecognizeResponse.Result.RESULT_FAILURE)

    def GetActivityRecentList(self, request, context):
        list_activities = activities.find({'causeId': request.user_id}) \
            .sort([('createdTime', -1)])
        for a in list_activities:
            yield motion_pb2.ActivityRecent(
                activity_id=str(a['_id']),
                is_success=a['isSuccess'],
                title=a['title'],
                content=a['content'],
                cause_id=a['causeId'],
                created_time=a['createdTime'].strftime("%Y-%m-%d %H:%M:%S"),
                modified_time=a['modifiedTime'].strftime("%Y-%m-%d %H:%M:%S")
            )

    def UpdateActivityRecent(self, request, context):
        stored_activity = activities.find_one(
            {'_id': ObjectId(request.activity_id)})
        stored_activity_model = RecentActivity(**stored_activity)
        update_data = {'isSuccess': False,
                       'modifiedTime': datetime.now()}
        update_activity = stored_activity_model.copy(update=update_data)
        activities.update_one({'_id': ObjectId(request.activity_id)}, {
            '$set': update_activity.dict()})
        redis_db.set(request.activity_id, update_activity.json())
        return motion_pb2.ActivityRecent(
            is_success=update_activity.isSuccess,
            title=update_activity.title,
            content=update_activity.content,
            cause_id=update_activity.causeId,
            created_time=update_activity.createdTime.strftime(
                "%Y-%m-%d %H:%M:%S"),
            modified_time=update_activity.modifiedTime.strftime(
                "%Y-%m-%d %H:%M:%S")
        )
