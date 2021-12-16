
'''
Document item type
'''
from mongoengine import StringField, ReferenceField, DynamicDocument, PULL
from mongoengine.queryset.base import CASCADE
from Tenant import Tenant


class Item(DynamicDocument):
    '''
    item type. Each item consists of its own id and content embeddings
    '''
    tenantRef = ReferenceField(
        Tenant, reverse_delete_rule=CASCADE, required=True)
    itemId = StringField(required=True, unique_with=[
                         "itemType", "contentType", "tenantRef"])
    itemType = StringField(requried=True)
    contentType = StringField(required=True)
    content = StringField(default="")


Item.register_delete_rule(Tenant, "itemRefs", PULL)
