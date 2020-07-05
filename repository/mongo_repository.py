class MongoRepository:
    def __init__(self, collection):
        self.collection = collection

    def create(self, doc):
        if doc is not None:
            self.collection.insert(doc)
        else:
            raise Exception("Nothing to save, because insert doc parameter is None")

    def get(self, user_id):
        if user_id is not None:
            return self.collection.find_one({"user_id": user_id})
