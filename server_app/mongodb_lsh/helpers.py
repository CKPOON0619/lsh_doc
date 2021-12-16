from schema import Tenant, User, Item, HashMapTable, HashBucket
import tensorflow as tf


def create_hashes(hashes):
    return tf.strings.reduce_join(hashes, axis=-1)


def fetchTenant(tenantId):
    tenantRecord = Tenant.objects(tenantId=tenantId)
    if not tenantRecord:
        return False
    else:
        return tenantRecord[0]


def fetchUser(userEmail):
    userRecord = User.objects(email=userEmail)
    if not userRecord:
        return False
    else:
        return userRecord[0]


def calculate_hashes(projections, embeddings):
    hashes_array = tf.map_fn(create_hashes, tf.where(
        tf.matmul(embeddings, projections) > 0, "1", "0"))
    return hashes_array.numpy().astype("str")


def get_items(tables, hashes_queries):
    res = dict()
    for table, hashes in zip(tables, hashes_queries):
        for hash in hashes:
            res.update([(i, True) for i in table.get(hash, [])])
    return list(res.keys())
