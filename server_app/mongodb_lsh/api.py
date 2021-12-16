import numpy as np
from fastapi import Body, FastAPI
from interfaces import UserRegistry, TenantRegistry, TenantUserRegistry, TenantItemRegistry, TenantItemDelete, TenantTableRefresh, TenantTableItemRegistry, TenantItemQueryWithIds, TenantItemQueryWithContents
from helpers import fetchUser, fetchTenant, calculate_hashes, get_items
from schema import Item, User, Tenant, HashMapTable, HashBucket
from pymongo import UpdateOne

app = FastAPI()


@app.post("/add/user")
async def add_user(userRegistry: UserRegistry):
    userEmail = userRegistry.email
    user = fetchUser(userEmail)
    if user:
        return {"state": "Error", "message": "user already exist."}
    try:
        user = User(email=userEmail)
        user.save()
        return {"state": "Success"}
    except Exception as errorMsg:
        return {"state": "Error", "message": "User creation failed upon saving: {}".format(errorMsg)}


@app.post("/add/tenant")
async def add_tenant(tenantRegistry: TenantRegistry):
    tenant = fetchTenant(tenantRegistry.tenantId)
    if tenant:
        return {"state": "Error", "message": "Tenant already exist."}
    else:
        try:
            tenant = Tenant(tenantId=tenantRegistry.tenantId,
                            userRefs=[], itemRefs=[])
            tenant.save()
            return {"state": "Success"}
        except Exception as errorMsg:
            return {"state": "Error", "message": "tenant creation failed upon saving: {}".format(errorMsg)}


@app.post("/tenant/add/user")
async def tenant_addUser(tenantUserRegistry: TenantUserRegistry):
    user = fetchUser(tenantUserRegistry.userEmail)
    if not user:
        return {"state": "Error", "message": "user does not exist."}
    else:
        try:
            update = Tenant.objects(tenantId=tenantUserRegistry.tenantId).update(
                add_to_set__userRefs=user)
            if not update:
                return {"state": "Error", "message": "Tenant not found."}
            return {"state": "Success", "message": "User added.".format(update)}
        except Exception as errorMsg:
            return {"state": "Error", "message": "User addition failed upon saving: {}".format(errorMsg)}


@app.post("/tenant/add/items")
async def tenant_Items(tenantItemRegistry: TenantItemRegistry):
    tenantUserRecords = Tenant.objects(
        tenantId=tenantItemRegistry.tenantId, userRefs=tenantItemRegistry.userEmail)
    if not tenantUserRecords:
        return {"state": "Error", "message": "User not found."}
    tenant = tenantUserRecords[0]
    try:
        itemInstances = [Item(tenantRef=tenant, **dict(registryItem))
                         for idx, registryItem in enumerate(tenantItemRegistry.items)]
        Item.objects.insert(itemInstances)
        tenantUserRecords.update(add_to_set__itemRefs=itemInstances)
        return {"state": "Success", "message": "Inserted {} items.".format(len(itemInstances))}
    except Exception as errorMsg:
        return {"state": "Error", "message": "Insertion failed: {}".format(errorMsg)}


@app.post("/tenant/delete/items")
async def tenant_deleteItems(tenantItemDelete: TenantItemDelete):
    tenantUserRecords = Tenant.objects(
        tenantId=tenantItemDelete.tenantId, userRefs=tenantItemDelete.userEmail)
    if not tenantUserRecords:
        return {"state": "Error", "message": "User not found."}
    try:
        itemDeletedNum = Item.objects(itemId__in=tenantItemDelete.itemIds,
                                      contentType=tenantItemDelete.contentType, itemType=tenantItemDelete.itemType).delete()
        return {"state": "Success", "message": "Deleted {} items.".format(itemDeletedNum)}
    except Exception as errorMsg:
        return {"state": "Error", "message": "Deletion failed: {}".format(errorMsg)}

# Review if we should allow updating items or instead deleting and add again


@app.post("/tenant/update/items")
async def tenant_Items(tenantItemRegistry: TenantItemRegistry):
    tenantUserRecords = Tenant.objects(
        tenantId=tenantItemRegistry.tenantId, userRefs=tenantItemRegistry.userEmail)
    if not tenantUserRecords:
        return {"state": "Error", "message": "User not found."}
    updated = 0
    for idx, registryItem in enumerate(tenantItemRegistry.items):
        try:
            dbItemRecords = Item.objects(
                tenantRef=tenantUserRecords[0],
                itemId=registryItem.itemId,
                itemType=registryItem.itemType,
                contentType=registryItem.contentType
            )
            if dbItemRecords:
                dbItemRecords.update(content=registryItem.content)
                updated += 1
        except Exception as errorMsg:
            return {"state": "Error", "message": "Failed updating item id {}, error: {}".format(registryItem.itemId, errorMsg)}
    if not updated:
        return {"state": "Error", "message": "No items found."}
    return {"state": "Success", "message": "Updated {} item(s)".format(updated)}


@app.post("/tenant/refresh/table")
async def tenant_refreshTable(tenantTableRefresh: TenantTableRefresh):
    tenantUserRecords = Tenant.objects(
        tenantId=tenantTableRefresh.tenantId, userRefs=tenantTableRefresh.userEmail)
    if not tenantUserRecords:
        return {"state": "Error", "message": "No user tenant found."}
    tenant = tenantUserRecords[0]
    encoderModel = ENCODERS[tenantTableRefresh.embeddingType]
    items = tenant.itemRefs
    contents = [i.content for i in items]

    hashMapTableRecords = HashMapTable.objects(
        tenantRef=tenantUserRecords[0],
        contentType=tenantTableRefresh.contentType,
        itemType=tenantTableRefresh.itemType,
        embeddingType=tenantTableRefresh.embeddingType,
        hash_dim=tenantTableRefresh.hash_dim
    )

    if hashMapTableRecords:
        try:
            hashMapTableRecords.delete()
        except Exception as errorMsg:
            return {"state": "Error", "message": "Failed in removing existing table: {}".format(errorMsg)}

    newHashMapTableInstances = [
        HashMapTable(
            tenantRef=tenant,
            contentType=tenantTableRefresh.contentType,
            itemType=tenantTableRefresh.itemType,
            embeddingType=tenantTableRefresh.embeddingType,
            hash_dim=tenantTableRefresh.hash_dim,
            hashBucketRefs=[],
            seed=np.random.randint(0, 99999999)
        ) for i in range(tenantTableRefresh.num_tables)
    ]
    try:
        HashMapTable.objects.insert(newHashMapTableInstances)
        return {"state": "Success", "message": "Table refreshed."}
    except Exception as errorMsg:
        return {"state": "Error", "message": "Table insertion fail: {}".format(errorMsg)}


# TODO: test performance if not using bulk write
@app.post("/tenant/table/add/items")
async def tenant_addItems(tenantTableItemRegistry: TenantTableItemRegistry):
    tenantUserRecords = Tenant.objects(
        tenantId=tenantTableItemRegistry.tenantId,
        userRefs=tenantTableItemRegistry.userEmail,
    )
    if not tenantUserRecords:
        return {"state": "Error", "message": "User not found."}
    tenant = tenantUserRecords[0]
    hashMapTableRecords = HashMapTable.objects(
        tenantRef=tenant,
        contentType=tenantTableItemRegistry.contentType,
        itemType=tenantTableItemRegistry.itemType,
        embeddingType=tenantTableItemRegistry.embeddingType,
        hash_dim=tenantTableItemRegistry.hash_dim,
    )

    encoderModel = ENCODERS[tenantTableItemRegistry.embeddingType]
    if not hashMapTableRecords:
        return {"state": "Error", "message": "Tables not found."}

    itemRecords = Item.objects(tenantRef=tenant, itemId__in=tenantTableItemRegistry.itemIds,
                               itemType=tenantTableItemRegistry.itemType, contentType=tenantTableItemRegistry.contentType)
    if not itemRecords:
        return {"state": "Error", "message": "Items not found."}
    itemValues = list(itemRecords.aggregate([
        {"$group": {
            "_id": None,
            "ids": {
                "$push": "$_id",
            },
            "contents": {
                "$push": "$content",
            }
        }
        }
    ]))[0]
    itemIds = itemValues['ids']
    contents = itemValues['contents']
    hashMapTableValues = list(hashMapTableRecords.aggregate([
        {
            '$lookup': {
                'from': 'hash_bucket',
                'localField': 'hashBucketRefs',
                'foreignField': '_id',
                'as': 'hashBuckets'
            }
        },
        {
            "$project": {
                "seed": 1,
                "hashBucketLookup": {
                    "$map":
                    {
                        "input": "$hashBuckets",
                        "as": "hashBucket",
                        "in": {"k": "$$hashBucket.hashKey", "v": "$$hashBucket._id"}
                    }
                }
            }
        },
        {
            "$project": {
                "seed": 1,
                "hashBucketLookup": {"$arrayToObject": "$hashBucketLookup"}
            }
        }
    ]))

    embeddings = ENCODERS[tenantTableItemRegistry.embeddingType]["encode"](
        itemValues["contents"])
    seeds = [hashTable["seed"] for hashTable in hashMapTableValues]
    projections = []
    for seed in seeds:
        np.random.seed(seed)
        projection = np.random.randn(
            encoderModel["embedding_dim"], tenantTableItemRegistry.hash_dim)
        projections.append(projection)

    hashes_array = calculate_hashes(projections, embeddings)
    newhashBucketInstances = []
    hashBucketUpdates = []
    hashMapTableUpdateRecords = []

    for hashes, hashMapTable in zip(hashes_array, hashMapTableValues):
        hashBucketLookup = hashMapTable["hashBucketLookup"]
        hashBucketChangesLookup = dict()
        for itemId, hash in zip(itemIds, hashes):
            recorded_change = hashBucketChangesLookup.get(hash)
            if not recorded_change:
                hashBucketChangesLookup[hash] = [itemId]
            else:
                hashBucketChangesLookup[hash].append(itemId)

        for hash, itemList in hashBucketChangesLookup.items():
            if hash in hashBucketLookup:
                hashBucketUpdates.append(
                    (hashBucketLookup[hash], hashBucketChangesLookup[hash]))
            else:
                bucket = HashBucket(
                    tenantRef=tenant,
                    hashKey=hash,
                    hashMapTableRef=hashMapTable["_id"],
                    itemRefs=hashBucketChangesLookup[hash]
                )
                newhashBucketInstances.append(bucket)
                hashMapTableUpdateRecords.append(
                    (hashMapTable["_id"], hash, bucket))

    if newhashBucketInstances:

        HashBucket.objects.insert(newhashBucketInstances)
    if hashBucketUpdates:

        HashBucket._get_collection().bulk_write([UpdateOne({'_id': bucketRef}, {'$addToSet': {
            'itemRefs': {'$each': itemRefs}}}) for bucketRef, itemRefs in hashBucketUpdates], ordered=False)
    if hashMapTableUpdateRecords:
        HashMapTable._get_collection().bulk_write([UpdateOne({'_id': tableId}, {'$addToSet': {
            'hashBucketRefs': bucket.id}}) for tableId, hash, bucket in hashMapTableUpdateRecords], ordered=False)

    return {"state": "Success", "message": "Successful. Added {} new hash buckets to tables. {} updates to existing hash buckets.".format(len(newhashBucketInstances), len(hashBucketUpdates))}


@app.get("/tenant/table/query/items")
async def tenant_queryItems(tenantTableItemQuery: Union[TenantItemQueryWithIds, TenantItemQueryWithContents]):
    tenantUserRecords = Tenant.objects(
        tenantId=tenantTableItemQuery.tenantId,
        userRefs=tenantTableItemQuery.userEmail,
    )
    query_dict = dict(tenantTableItemQuery)
    if query_dict.get("itemIds"):
        itemRecords = Item.objects(itemId__in=tenantTableItemQuery.itemIds)
        if not itemRecords:
            return {"state": "Error", "message": "Item not found."}
        query_embeddings = USE_embed(itemRecords.values_list("content"))
    else:
        query_embeddings = USE_embed(tenantTableItemQuery.contents)

    if not tenantUserRecords:
        return {"state": "Error", "message": "User not found."}

    hashMapTableRecords = HashMapTable.objects(
        tenantRef=tenantUserRecords[0],
        contentType=tenantTableItemQuery.contentType,
        itemType=tenantTableItemQuery.itemType,
        embeddingType=tenantTableItemQuery.embeddingType,
        hash_dim=tenantTableItemQuery.hash_dim
    ).only("seed", "hashBucketRefs")

    if not hashMapTableRecords:
        return {"state": "Error", "message": "Tables not found."}

    existingTableNum = len(hashMapTableRecords)

    if tenantTableItemQuery.queryTableNum > existingTableNum:
        return {"state": "Error", "message": "queryTableNum({}) exceeds the number of existing tables({}).".format(tenantTableItemQuery.queryTableNum, existingTableNum)}

    encoderModel = ENCODERS[tenantTableItemQuery.embeddingType]
    hashMapTableValues = list(hashMapTableRecords[:tenantTableItemQuery.queryTableNum].aggregate([
        {
            '$lookup': {
                'from': 'hash_bucket',
                'localField': 'hashBucketRefs',
                'foreignField': '_id',
                'as': 'hashBuckets'
            }
        },
        {
            "$project": {
                "seed": 1,
                "hashBucketLookup":
                {"$map":
                 {
                     "input": "$hashBuckets",
                     "as": "hashBucket",
                     "in": {"k": "$$hashBucket.hashKey", "v": "$$hashBucket.itemRefs"}
                 }
                 }
            }
        },
        {
            "$project": {
                "seed": 1,
                "hashBucketLookup": {"$arrayToObject": "$hashBucketLookup"}
            }
        }
    ]))

    projections = []
    tables = []
    for result in hashMapTableValues:
        np.random.seed(result['seed'])
        projections.append(np.random.randn(
            encoderModel["embedding_dim"], tenantTableItemQuery.hash_dim))
        tables.append(result['hashBucketLookup'])
    tableQueryHashes = calculate_hashes(projections, query_embeddings)
    result = Item.objects(id__in=get_items(
        tables, tableQueryHashes)).distinct(field="itemId")
    return {"state": "Success", "result": result}
