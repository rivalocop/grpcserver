import redis
from pymongo import MongoClient

import settings

redis_db = redis.StrictRedis(host=settings.REDIS_HOST,
                             port=settings.REDIS_PORT, db=settings.REDIS_DB)
mongo_client = MongoClient('mongodb://localhost:27017/')
db = mongo_client['FaceVerifySystem']
faces = db.face_indexes
users = db.users
activities = db.activities
