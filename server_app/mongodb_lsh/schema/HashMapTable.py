'''
Defining lsh table type
'''

from mongoengine import Document, ListField, IntField, ReferenceField, StringField, CASCADE, PULL
from Tenant import Tenant
from Item import Item


class HashMapTable(Document):
    tenantRef = ReferenceField(
        Tenant, required=True, reverse_delete_rule=CASCADE)
    itemType = StringField(required=True)
    contentType = StringField(required=True)
    embeddingType = StringField(required=True)
    hash_dim = IntField(required=True)
    seed = IntField(required=True)
    hashBucketRefs = ListField(ReferenceField("HashBucket"))


class HashBucket(Document):
    tenantRef = ReferenceField(
        Tenant, required=True, reverse_delete_rule=CASCADE)
    hashKey = StringField(unique_with=["hashMapTableRef"], required=True)
    hashMapTableRef = ReferenceField(
        HashMapTable, required=True, reverse_delete_rule=CASCADE)
    itemRefs = ListField(ReferenceField(Item, reverse_delete_rule=PULL))


HashBucket.register_delete_rule(HashMapTable, "hashBucketRefs", PULL)
