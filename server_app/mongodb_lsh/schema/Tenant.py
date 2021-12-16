'''
Define Tenant
'''
from mongoengine import DynamicDocument, StringField, ListField, ReferenceField


class Tenant(DynamicDocument):
    tenantId = StringField(primary_key=True)
    userRefs = ListField(ReferenceField("User"))
    itemRefs = ListField(ReferenceField("Item"))
    hashMapTableRefs = ListField(ReferenceField("HashMapTable"))
