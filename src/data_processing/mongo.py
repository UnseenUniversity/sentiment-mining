
import pymongo
from pymongo import MongoClient

class MongoORM():

    def __init__(self, db_name="yelp"):
        self.client = MongoClient()
        self.db     = self.client[db_name]

    # method returns a cursor (only accesses information when needed)
    def get_collection(self, name):
        return self.db[name].find()

    def get_item(self, collection, item):
        return self.db[collection].find(item)

    def get_item_by_key(self, collection, key, value):
        return self.db[collection].find_one({key: value})

    def get_item_by_id(self, collection, _id):
        return self.get_item_by_key(collection, "_id", _id)

    def get_item_count(self, collection, item):
        return self.db[collection].find(item).count()

    def get_collection_count(self, collection):
        return self.db[collection].count()

    #items is a dictionary which selects from the collection the interesting fields
    #item_filter is a dictionary which tells mongo what fields should the method return
    #sort by index tells mongo what field(s) should be user for sorting
    #consider creating an index on a field if you plan to sort frequently
    #see get_user_sorted_by_reviews()
    def get_sorted_collection(self, collection, items, item_filter, sort_by_index, direction=pymongo.DESCENDING):
        return self.db[collection].find(items, item_filter).sort(sort_by_index, direction)

    def insert_item(self, collection, item):
        self.db[collection].insert(item)

    # If upsert is set to true, creates a new document when no document matches
    # the query criteria. The default value is false, which does not insert a new document
    # when no match is found.
    def upsert_item(self, collection, item, upsert=True):
        self.db[collection].update({"_id": item["_id"]}, {"$set": item}, upsert=upsert)

    def update_item(self, collection, item, update_blob):
        return self.db[collection].update(item, update_blob)

    def update_item_array(self, collection, item, field, values):
        return self.db[collection].update(item, {"$addToSet": {field: {"$each": values}}})

    def increment_field(self, collection, item_id, field, amount):
        return self.db[collection].update({"_id", item_id},
                                          {"$inc": {field: amount}},
                                          upsert=True)

    def get_users(self):
        return self.get_collection("users")

    def get_user_field(self, user_id, field):
        return self.get_item_by_key("users", "user_id", user_id)[field]

    def get_user_sorted_by_reviews(self, limit=10):
        return self.get_sorted_collection("users", {},
                                          {"user_id": 1, "review_count": 1},
                                          "review_count").limit(limit)


def _test():

    #PyMongo Documentation: http://api.mongodb.org/python/current/index.html

    #Usage example
    db = MongoORM()
    print db.get_collection_count("users")

    # get 10 random users
    count_users = 10
    users = db.get_collection("users")

    idx = 0
    for user in users:
        print "User: ", user["user_id"], user["name"]
        idx += 1
        if idx >= count_users:
            break

    # get the number of reviews from a random user
    user_id = "fHtTaujcyKvXglE33Z5yIw"
    print db.get_user_field(user_id, "name"), \
          db.get_user_field(user_id, "review_count")

    # return top 10 user by number of reviews
    for user in db.get_user_sorted_by_reviews():
        print "User: ", user["user_id"], " | Review count: ", user["review_count"]

if __name__ == "__main__":
    _test()