'''
Define users
'''
from mongoengine import DynamicDocument, StringField, PULL
from Tenant import Tenant


class User(DynamicDocument):
    email = StringField(primary_key=True)


User.register_delete_rule(Tenant, "userRefs", PULL)
