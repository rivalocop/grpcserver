from pymongo import MongoClient

mongo_client = MongoClient('mongodb://localhost:27017/')
db = mongo_client['FaceVerifySystem']
faces = db.face_indexes
