# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""MongoDB handler."""

import functools
import time
import pymongo
import os
from urllib import parse
from pymongo.errors import WriteError, AutoReconnect
import logging

# Configure logging
TAO_LOG_LEVEL = os.getenv('TAO_LOG_LEVEL', 'INFO').upper()
tao_log_level = getattr(logging, TAO_LOG_LEVEL, logging.INFO)
logging.basicConfig(
    level=logging.WARNING,  # Root logger: suppress third-party DEBUG logs
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logging.getLogger('nvidia_tao_core').setLevel(tao_log_level)
logger = logging.getLogger(__name__)

# MongoDB connection setup
mongo_client = None  # Initialize to None
mongo_secret = None
mongo_namespace = None
mongo_operator_enabled = None
_client_initialized = False


def _init_mongo_client():
    """Initialize MongoDB client on first use (lazy initialization).

    This is called automatically when MongoHandler is created. It checks environment
    variables to determine whether to use mongomock (test mode) or real MongoDB (production).
    """
    # pylint: disable=global-statement
    global mongo_client, mongo_secret, mongo_namespace, mongo_operator_enabled, _client_initialized

    if _client_initialized:
        return

    # Check for test mode FIRST
    if os.getenv("TAO_TEST_MODE", "").lower() in ["true", "1", "yes"]:
        # Use mongomock for testing (no Docker/K8s required)
        logger.info("Running in TEST_MODE - using mongomock for MongoDB operations")
        try:
            import mongomock
            mongo_client = mongomock.MongoClient()
            mongo_namespace = "test"
            mongo_operator_enabled = False
            _client_initialized = True
            logger.info("Successfully initialized mongomock client for testing")
        except ImportError as exc:
            raise ImportError(
                "mongomock is required for test mode. Install with: pip install mongomock"
            ) from exc
    elif os.getenv("BACKEND"):
        # Production setup (Docker/K8s with real MongoDB)
        if os.getenv("HOST_PLATFORM") == "local-docker":
            mongo_secret = os.getenv("MONGOSECRET", "")
            if not mongo_secret:
                raise ValueError(
                    "MONGOSECRET environment variable is required but not set. "
                    "Please ensure MongoDB credentials are properly configured."
                )
            mongo_namespace = "default"
            mongo_operator_enabled = False
            encoded_secret = parse.quote(mongo_secret, safe='')
            mongo_uri_prefix = "mongodb"
            mongo_connection_string = (
                f"{mongo_uri_prefix}://default-user:{encoded_secret}@mongodb:27017/tao"
                "?authSource=admin"
            )
            mongo_client = pymongo.MongoClient(mongo_connection_string, tz_aware=True)
            _client_initialized = True
        else:  # k8s
            mongo_secret = os.getenv("MONGOSECRET", "")
            if not mongo_secret:
                raise ValueError(
                    "MONGOSECRET environment variable is required but not set. "
                    "Please ensure MongoDB credentials are properly configured."
                )
            mongo_namespace = os.getenv("NAMESPACE", "default")
            mongo_operator_enabled = os.getenv('MONGO_OPERATOR_ENABLED', 'true') == 'true'
            encoded_secret = parse.quote(mongo_secret, safe='')
            mongo_uri_prefix = "mongodb+srv" if mongo_operator_enabled else "mongodb"
            mongo_connection_string = (
                f"{mongo_uri_prefix}://default-user:{encoded_secret}@mongodb-svc.{mongo_namespace}"
                ".svc.cluster.local/tao?replicaSet=mongodb&ssl=false&authSource=admin"
            )
            mongo_client = pymongo.MongoClient(mongo_connection_string, tz_aware=True)
            _client_initialized = True


NUM_RETRY = 5


def reset_test_db():
    """Reset the test database (only works in TEST_MODE).

    This function drops all databases in the mongomock client,
    useful for cleaning up between tests.

    Raises:
        RuntimeError: If called outside of TEST_MODE
    """
    if os.getenv("TAO_TEST_MODE", "").lower() not in ["true", "1", "yes"]:
        raise RuntimeError("reset_test_db() can only be called in TEST_MODE")

    _init_mongo_client()

    if mongo_client is None:
        raise RuntimeError("MongoDB client not initialized")

    # Drop all databases except admin, local, and config
    for db_name in mongo_client.list_database_names():
        if db_name not in ['admin', 'local', 'config']:
            mongo_client.drop_database(db_name)
    logger.info("Test database reset completed")


def get_test_client():
    """Get the MongoDB client (useful for testing).

    Returns:
        The mongo_client instance

    Raises:
        RuntimeError: If client is not initialized
    """
    _init_mongo_client()
    if mongo_client is None:
        raise RuntimeError("MongoDB client not initialized")
    return mongo_client


def retry_method(func):
    """Decorator to retry DB methods for a specified number of times."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        retries = kwargs.pop('retries', NUM_RETRY)
        for i in range(retries):
            try:
                return func(*args, **kwargs)
            except AutoReconnect as e:
                logger.error("AutoReconnect exception in %s: %s", func.__name__, e)
            except WriteError as e:
                logger.error(
                    "WriteError exception in %s: %s \n"
                    "With arguments %s and %s",
                    func.__name__, e, args, kwargs
                )
            except Exception as e:
                # Log or handle the exception as needed
                logger.error("Exception in %s: %s", func.__name__, e)
            if i != retries - 1:
                time.sleep(30)
        # If all retries fail, raise an exception or handle it accordingly
        raise ValueError(f"Failed to execute {func.__name__} after multiple retries")
    return wrapper


class MongoHandler:
    """Handler class for MongoDB operations with retry capability."""

    @retry_method
    def __init__(self, db_name, collection_name):
        """Initialize MongoHandler with specified database and collection.

        Args:
            db_name (str): Name of the database to connect to.
            collection_name (str): Name of the collection within the database.
        """
        _init_mongo_client()

        if mongo_client is None:
            error_msg = (
                "MongoDB client not initialized. "
                "For production: Set BACKEND environment variable. "
                "For testing: Set TAO_TEST_MODE=true and ensure mongomock is installed."
            )
            raise RuntimeError(error_msg)
        self.mongo_client = mongo_client  # pylint: disable=E0606
        self.db = self.mongo_client[db_name]
        self.collection = self.db[collection_name]
        # Create unique index on 'id' field
        # The retry_method decorator will handle AutoReconnect exceptions
        self.create_unique_index("id")

    @retry_method
    def upsert(self, query, new_data):
        """Insert or update a document based on the query.

        Args:
            query (dict): Query criteria for selecting the document.
            new_data (dict): Data to insert or update in the document.
        """
        self.collection.update_one(query, {'$set': new_data}, upsert=True)

    @retry_method
    def update_many(self, query, new_data):
        """Update multiple documents based on the query.

        Args:
            query (dict): Query criteria for selecting the documents.
            new_data (dict): Data to update in the documents.
        """
        self.collection.update_many(query, {'$set': new_data})

    @retry_method
    def upsert_append(self, query, new_data):
        """Append new data to the 'status' field in a document.

        If a document matching the query exists, the new data is appended to the 'status' array.
        If no document matches, a new one is created with the given query and 'status' field.

        Args:
            query (dict): The filter criteria to find the document.
            new_data (Any): The data to append to the 'status' field.

        Returns:
            None
        """
        append_data = {'$push': {'status': new_data}}
        self.collection.update_one(query, append_data, upsert=True)

    def delete_one(self, query):
        """Delete a single document matching the query.

        Args:
            query (dict): Query criteria for selecting the document.
        """
        self.collection.delete_one(query)

    def delete_many(self, query):
        """Delete multiple documents matching the query.

        Args:
            query (dict): Query criteria for selecting documents.
        """
        self.collection.delete_many(query)

    def find(self, query):
        """Find documents that match the query.

        Args:
            query (dict): Query criteria for selecting documents.

        Returns:
            list: List of documents that match the query.
        """
        if not query:
            result = list(self.collection.find())
        else:
            result = list(self.collection.find(query))

        return result if result else []

    def find_one(self, query=None):
        """Find a single document that matches the query.

        Args:
            query (dict, optional): Query criteria for selecting the document. Defaults to None.

        Returns:
            dict: The first document that matches the query.
        """
        if not query:
            result = self.collection.find_one()
        else:
            result = self.collection.find_one(query)

        return result if result else {}

    def find_latest(self):
        """Retrieve the latest document from the collection.

        The latest document is determined based on the '_id' field in descending order.

        Returns:
            dict: The latest document if found, otherwise an empty dictionary.
        """
        result = self.collection.find_one(sort=[('_id', pymongo.DESCENDING)])
        return result if result else {}

    def create_unique_index(self, index):
        """Create a unique index on the specified field.

        Args:
            index (str): Field to create the unique index on.
        """
        self.collection.create_index(index, unique=True)

    def create_text_index(self, index):
        """Create a text index on the specified field.

        Args:
            index (str): Field to create the text index on.
        """
        self.collection.create_index([(index, pymongo.TEXT)])

    def create_ttl_index(self, index, ttl_time_in_seconds):
        """Create a TTL (time-to-live) index on the specified field.

        Args:
            index (str): Field to create the TTL index on.
            ttl_time_in_seconds (int): Time in seconds after which the document will expire.
        """
        self.collection.create_index(index, expireAfterSeconds=ttl_time_in_seconds)
