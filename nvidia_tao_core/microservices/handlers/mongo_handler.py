# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""MongoDB backup and restore handler module"""
import os
import logging
import traceback
import json
import gzip
from datetime import datetime
from bson import ObjectId

from nvidia_tao_core.microservices.utils.handler_utils import Code
from nvidia_tao_core.microservices.utils.stateless_handler_utils import get_root
from nvidia_tao_core.microservices.utils.cloud_utils import create_cs_instance

if os.getenv("BACKEND"):
    from nvidia_tao_core.microservices.utils.mongo_utils import mongo_client, MongoHandler
else:
    # Define stub for when BACKEND is not set
    MongoHandler = None


# Configure logging
logger = logging.getLogger(__name__)


class MongoBackupHandler:
    """Handles MongoDB backup and restore operations."""

    @staticmethod
    def _create_workspace_metadata_from_cli(access_key, secret_key, s3_bucket_name, region=None, endpoint_url=None):
        """Create workspace metadata dictionary from CLI parameters.

        Args:
            access_key (str): AWS access key
            secret_key (str): AWS secret key
            s3_bucket_name (str): S3 bucket name
            region (str, optional): AWS region
            endpoint_url (str, optional): Custom S3 endpoint URL

        Returns:
            dict: Workspace metadata dictionary
        """
        return {
            "cloud_type": "aws",
            "cloud_details": {
                "aws_access_key_id": access_key,
                "aws_secret_access_key": secret_key,
                "s3_bucket": s3_bucket_name,
                "aws_default_region": region or "us-east-1",
                "endpoint_url": endpoint_url
            }
        }

    @staticmethod
    def _serialize_document(doc):
        """Convert MongoDB document to JSON-serializable format."""
        if isinstance(doc, dict):
            return {k: MongoBackupHandler._serialize_document(v) for k, v in doc.items()}
        if isinstance(doc, list):
            return [MongoBackupHandler._serialize_document(item) for item in doc]
        if isinstance(doc, ObjectId):
            return {"$oid": str(doc)}
        if isinstance(doc, datetime):
            return {"$date": doc.isoformat()}
        return doc

    @staticmethod
    def _deserialize_document(doc):
        """Convert JSON document back to MongoDB format."""
        if isinstance(doc, dict):
            if "$oid" in doc:
                return ObjectId(doc["$oid"])
            if "$date" in doc:
                return datetime.fromisoformat(doc["$date"])
            return {k: MongoBackupHandler._deserialize_document(v) for k, v in doc.items()}
        if isinstance(doc, list):
            return [MongoBackupHandler._deserialize_document(item) for item in doc]
        return doc

    @staticmethod
    def _setup_backup_file_path(backup_file_name="mongodb_backup..gz"):
        """Setup backup file path and ensure directory exists.

        Returns:
            str: Full path to backup file
        """
        root = get_root()
        dump_dir = os.path.join(root, "dump", "archive")
        os.makedirs(dump_dir, exist_ok=True)
        return os.path.join(dump_dir, backup_file_name)

    @staticmethod
    def _perform_mongodb_backup():
        """Perform the actual MongoDB backup operation.

        Returns:
            dict: Backup data structure
        """
        # Get all database names (excluding admin, config, local)
        db_names = [name for name in mongo_client.list_database_names()
                    if name not in ['admin', 'config', 'local']]

        backup_data = {
            "backup_timestamp": datetime.utcnow().isoformat(),
            "databases": {}
        }

        # Backup each database
        for db_name in db_names:
            logger.info("Backing up database: %s", db_name)
            database = mongo_client[db_name]
            backup_data["databases"][db_name] = {
                "collections": {},
                "indexes": {}
            }

            # Backup collections and their documents
            for collection_name in database.list_collection_names():
                logger.info("Backing up collection: %s.%s", db_name, collection_name)
                collection = database[collection_name]

                # Backup documents
                documents = list(collection.find())
                serialized_docs = [MongoBackupHandler._serialize_document(doc) for doc in documents]
                backup_data["databases"][db_name]["collections"][collection_name] = serialized_docs

                # Backup indexes
                indexes = list(collection.list_indexes())
                backup_data["databases"][db_name]["indexes"][collection_name] = [
                    MongoBackupHandler._serialize_document(idx) for idx in indexes
                ]

        return backup_data

    @staticmethod
    def _save_and_upload_backup(backup_data, backup_file_path, cs_instance):
        """Save backup data to file and upload to cloud storage.

        Args:
            backup_data (dict): The backup data to save
            backup_file_path (str): Local file path to save to
            cs_instance: Cloud storage instance for upload
        """
        # Write compressed backup file
        with gzip.open(backup_file_path, 'wt', encoding='utf-8') as f:
            json.dump(backup_data, f, indent=2)

        # Upload to cloud storage
        cs_instance.upload_file(backup_file_path, backup_file_path)

        # Clean up local file
        os.remove(backup_file_path)

    @staticmethod
    def mongo_backup(cs_instance, backup_file_name=None):
        """Backup MongoDB data using PyMongo.

        Args:
            cs_instance: Cloud storage instance for uploading backup.
            backup_file_name (str, optional): Name of the backup file.

        Returns:
            Response: A response indicating the outcome of the operation.
        """
        try:
            logger.info("Starting PyMongo-based MongoDB backup")

            # Setup backup file path
            backup_file_path = MongoBackupHandler._setup_backup_file_path(backup_file_name)

            # Perform MongoDB backup
            backup_data = MongoBackupHandler._perform_mongodb_backup()

            # Save and upload backup
            MongoBackupHandler._save_and_upload_backup(backup_data, backup_file_path, cs_instance)

            logger.info("Successfully backed up MongoDB using PyMongo to cloud storage")
            return Code(200, {"message": "MongoDB backup successful"}, "MongoDB backup successful")

        except Exception as e:
            logger.error("Exception thrown in mongo_backup: %s", str(e))
            logger.error(traceback.format_exc())
            return Code(400, {}, "Error in MongoDB backup")

    @staticmethod
    def backup_for_workspace(workspace_metadata, backup_file_name=None):
        """Backup MongoDB data using workspace metadata (high-level method for app.py).

        Args:
            workspace_metadata (dict): Workspace metadata containing cloud credentials
            backup_file_name (str, optional): Name of the backup file

        Returns:
            Response: A response indicating the outcome of the operation
        """
        try:
            # Validate workspace metadata
            if not workspace_metadata:
                return Code(400, {}, "Workspace metadata not provided")

            cloud_type = workspace_metadata.get("cloud_type")
            if cloud_type not in ["aws", "azure"]:
                return Code(400, {}, "MongoDB backup/restore is only supported for AWS and Azure workspaces")

            # Create cloud storage instance
            cs_instance, _ = create_cs_instance(workspace_metadata)
            if not cs_instance:
                return Code(404, {}, "Unable to create cloud storage instance")

            # Perform backup
            return MongoBackupHandler.mongo_backup(cs_instance, backup_file_name)

        except Exception as e:
            logger.error("Exception in backup_for_workspace: %s", str(e))
            logger.error(traceback.format_exc())
            return Code(400, {}, f"Error in workspace backup: {str(e)}")

    @staticmethod
    def backup_from_cli(access_key, secret_key, s3_bucket_name, region=None, endpoint_url=None):
        """CLI-friendly backup method using PyMongo.

        Args:
            access_key (str): AWS access key
            secret_key (str): AWS secret key
            s3_bucket_name (str): S3 bucket name
            region (str, optional): AWS region
            endpoint_url (str, optional): Custom S3 endpoint URL

        Returns:
            bool: True if backup successful, False otherwise
        """
        try:
            logger.info("Starting CLI backup")

            # Create workspace metadata from CLI parameters
            workspace_metadata = MongoBackupHandler._create_workspace_metadata_from_cli(
                access_key, secret_key, s3_bucket_name, region, endpoint_url
            )

            # Use the workspace backup method
            result = MongoBackupHandler.backup_for_workspace(workspace_metadata)
            return result.status_code == 200

        except Exception as e:
            logger.error("Exception in backup_from_cli: %s", str(e))
            logger.error(traceback.format_exc())
            return False

    @staticmethod
    def mongo_restore(cs_instance, backup_file_name=None):
        """Restore MongoDB data using PyMongo.

        Args:
            cs_instance: Cloud storage instance for downloading backup.
            backup_file_name (str, optional): Name of the backup file.

        Returns:
            Response: A response indicating the outcome of the operation.
        """
        try:
            logger.info("Starting PyMongo-based MongoDB restore")

            backup_file_path = MongoBackupHandler._setup_backup_file_path(backup_file_name)

            # Download backup file from cloud storage
            cs_instance.download_file(backup_file_path, backup_file_path)
            logger.info("Downloaded backup file to %s", backup_file_path)

            # Read and decompress backup file
            with gzip.open(backup_file_path, 'rt', encoding='utf-8') as f:
                backup_data = json.load(f)

            # Restore each database
            for db_name, db_data in backup_data["databases"].items():
                database = mongo_client[db_name]

                # Restore collections
                for collection_name, documents in db_data["collections"].items():
                    collection = database[collection_name]

                    # Clear existing data (optional - you might want to make this configurable)
                    collection.drop()

                    if documents:  # Only insert if there are documents
                        # Deserialize documents
                        deserialized_docs = [MongoBackupHandler._deserialize_document(doc) for doc in documents]
                        collection.insert_many(deserialized_docs)

                # Restore indexes (skip the default _id index)
                if "indexes" in db_data:
                    for collection_name, indexes in db_data["indexes"].items():
                        collection = database[collection_name]
                        for index_info in indexes:
                            try:
                                if index_info.get("name") != "_id_":  # Skip default index
                                    deserialized_index = MongoBackupHandler._deserialize_document(index_info)
                                    # Extract index specification
                                    if "key" in deserialized_index:
                                        index_spec = list(deserialized_index["key"].items())
                                        index_options = {k: v for k, v in deserialized_index.items()
                                                         if k not in ["key", "v", "ns"]}
                                        collection.create_index(index_spec, **index_options)
                            except Exception as idx_error:
                                logger.warning("Failed to restore index %s: %s",
                                               index_info.get("name", "unknown"), str(idx_error))

            # Clean up backup file
            os.remove(backup_file_path)

            logger.info("Restored MongoDB from backup file using PyMongo")
            return Code(200, {"message": "MongoDB restore successful"}, "MongoDB restore successful")

        except Exception as e:
            logger.error("Exception thrown in mongo_restore: %s", str(e))
            logger.error(traceback.format_exc())
            return Code(400, {}, "Error in MongoDB restore")

    @staticmethod
    def restore_for_workspace(workspace_metadata, backup_file_name=None):
        """Restore MongoDB data using workspace metadata (high-level method for app.py).

        Args:
            workspace_metadata (dict): Workspace metadata containing cloud credentials
            backup_file_name (str, optional): Name of the backup file

        Returns:
            Response: A response indicating the outcome of the operation
        """
        try:
            # Validate workspace metadata
            if not workspace_metadata:
                return Code(400, {}, "Workspace metadata not provided")

            cloud_type = workspace_metadata.get("cloud_type")
            if cloud_type not in ["aws", "azure"]:
                return Code(400, {}, "MongoDB backup/restore is only supported for AWS and Azure workspaces")

            # Create cloud storage instance
            cs_instance, _ = create_cs_instance(workspace_metadata)
            if not cs_instance:
                return Code(404, {}, "Unable to create cloud storage instance")

            # Perform restore
            return MongoBackupHandler.mongo_restore(cs_instance, backup_file_name)

        except Exception as e:
            logger.error("Exception in restore_for_workspace: %s", str(e))
            logger.error(traceback.format_exc())
            return Code(400, {}, f"Error in workspace restore: {str(e)}")


if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="MongoDB backup using MongoBackupHandler (PyMongo-based)")
    parser.add_argument("--access-key", required=True, help="AWS S3 access key")
    parser.add_argument("--secret-key", required=True, help="AWS S3 secret key")
    parser.add_argument("--s3-bucket-name", required=True, help="AWS S3 bucket name")
    parser.add_argument("--region", help="AWS S3 region", default="us-east-1")
    parser.add_argument("--endpoint-url", help="AWS S3 endpoint URL", default=None)

    args = parser.parse_args()

    success = MongoBackupHandler.backup_from_cli(
        args.access_key,
        args.secret_key,
        args.s3_bucket_name,
        args.region,
        args.endpoint_url
    )

    if not success:
        sys.exit(1)

    logger.info("Backup completed successfully")


# Export both classes for external imports
__all__ = ['MongoBackupHandler', 'MongoHandler']
