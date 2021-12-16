'''
Init mongo database
'''
from mongoengine import register_connection


def gloabl_init():
    # Register a core database.
    alias_core = 'core'
    db = 'item_db'
    data = dict(
        # username=USERNAME,
        # password=PASSWORD,
        # host=HOST,
        # port=PORT,
        # authenication_source='admin',
        # authentication_mechanism='SCRAM-SHA-1',
        # ssl=True,
        # ssl_cert_reqs=ssl.CERT_NONE,
    )
    register_connection(alias=alias_core, name=db, **data)
