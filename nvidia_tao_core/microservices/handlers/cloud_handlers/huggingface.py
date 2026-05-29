# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Hugging Face model and dataset handler functions"""
import logging
from huggingface_hub import snapshot_download

logger = logging.getLogger(__name__)


def download_huggingface_dataset(download_url, destination_folder, token=None):
    """Download a dataset from the Hugging Face dataset hub.

    Args:
        download_url (str): The URL or repository ID of the dataset on Hugging Face.
                           Can be either a URL like 'https://huggingface.co/datasets/user/dataset'
                           or a repository ID like 'user/dataset'.
        destination_folder (str): The destination folder where the dataset will be saved.
        token (str): The token for accessing private datasets (optional).

    Returns:
        str: The path to the downloaded dataset directory.
    """
    # Extract repository ID from URL if needed
    repo_id = download_url
    if download_url.startswith("https://huggingface.co/datasets/"):
        # Extract repo_id from URL: https://huggingface.co/datasets/user/dataset
        repo_id = download_url.replace("https://huggingface.co/datasets/", "").rstrip("/")

    try:
        # Use snapshot_download with repo_type="dataset"
        if token:
            dataset_dir = snapshot_download(
                repo_id=repo_id,
                repo_type="dataset",
                local_dir=destination_folder,
                token=token
            )
        else:
            dataset_dir = snapshot_download(
                repo_id=repo_id,
                repo_type="dataset",
                local_dir=destination_folder
            )

        logger.info("Successfully downloaded HuggingFace dataset %s to %s", repo_id, destination_folder)
        return dataset_dir

    except Exception as e:
        logger.error("Failed to download HuggingFace dataset %s. Error: %s", repo_id, e)
        raise


def download_huggingface_model(download_url, destination_folder, token):
    """Download a model from the Hugging Face model hub.

    Args:
        download_url (str): The URL of the model on the Hugging Face model hub.
        destination_folder (str): The destination folder where the model will be saved.
        token (str): The token for accessing private models (optional).

    Returns:
        str: The path to the downloaded model directory.
    """
    if token:
        model_dir = snapshot_download(
            repo_id=download_url,
            local_dir=destination_folder,
            token=token
        )
    else:
        model_dir = snapshot_download(
            repo_id=download_url,
            local_dir=destination_folder)
    return model_dir


def get_huggingface_model_size_info(repo_id, token=None):
    """Get model size information from Hugging Face model hub.

    Args:
        repo_id (str): The repository ID of the model on Hugging Face (e.g., 'microsoft/DialoGPT-small').
        token (str): The token for accessing private models (optional).

    Returns:
        tuple: (total_size_bytes, file_count) or (None, None) if failed.
    """
    return _get_huggingface_repo_size_info(repo_id, repo_type="model", token=token)


def get_huggingface_dataset_size_info(repo_id, token=None):
    """Get dataset size information from Hugging Face dataset hub.

    Args:
        repo_id (str): The repository ID of the dataset on Hugging Face (e.g., 'squad', 'glue').
        token (str): The token for accessing private datasets (optional).

    Returns:
        tuple: (total_size_bytes, file_count) or (None, None) if failed.
    """
    return _get_huggingface_repo_size_info(repo_id, repo_type="dataset", token=token)


def _get_huggingface_repo_size_info(repo_id, repo_type="model", token=None):
    """Get repository size information from Hugging Face hub (internal helper).

    Args:
        repo_id (str): The repository ID on Hugging Face.
        repo_type (str): Type of repository - "model" or "dataset".
        token (str): The token for accessing private repositories (optional).

    Returns:
        tuple: (total_size_bytes, file_count) or (None, None) if failed.
    """
    if not repo_id:
        logger.error("Invalid repo_id provided")
        return None, None

    try:
        from huggingface_hub import HfApi  # pylint: disable=C0415

        # Create HfApi instance
        api = HfApi(token=token) if token else HfApi()

        # Get repository info with file metadata
        if repo_type == "dataset":
            repo_info = api.dataset_info(repo_id, files_metadata=True)
        else:
            repo_info = api.model_info(repo_id, files_metadata=True)

        # Calculate total size from individual files (more accurate than usedStorage)
        # usedStorage includes Git history and LFS overhead, not just downloadable files
        total_size_bytes = 0
        file_count = 0

        if hasattr(repo_info, 'siblings') and repo_info.siblings:
            for sibling in repo_info.siblings:
                if sibling.size is not None:
                    total_size_bytes += sibling.size
                    file_count += 1

        # Fallback if no siblings or no size info
        if file_count == 0:
            # Get file count from list_repo_files
            files = api.list_repo_files(repo_id, repo_type=repo_type)
            file_count = len(files)

            # If we couldn't calculate size from siblings, try usedStorage as last resort
            if hasattr(repo_info, 'usedStorage') and repo_info.usedStorage:
                total_size_bytes = repo_info.usedStorage
            else:
                total_size_bytes = None

        logger.info("HuggingFace %s %s: size=%s bytes, files=%s", repo_type, repo_id, total_size_bytes, file_count)
        return total_size_bytes, file_count

    except Exception as e:
        logger.error("Failed to get HuggingFace %s size info for %s. Error: %s", repo_type, repo_id, e)
        return None, None
