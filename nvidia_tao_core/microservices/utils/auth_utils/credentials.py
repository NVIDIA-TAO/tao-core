# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Authentication utils credential modules"""
import copy
import datetime
import os
import uuid
import traceback
import jwt
import logging

from .session import (
    __SESSION_EXPIRY_SECONDS__,
    _SESSION_REFRESH_SECONDS__,
    set_session,
    get_user_metadata_from_ngc_response
)
from nvidia_tao_core.microservices.utils.ngc_utils import (
    get_user_key,
    get_user_info,
)
from nvidia_tao_core.microservices.utils.encrypt_utils import NVVaultEncryption
from nvidia_tao_core.microservices.utils.mongo_utils import MongoHandler

# Configure logging
TAO_LOG_LEVEL = os.getenv('TAO_LOG_LEVEL', 'INFO').upper()
tao_log_level = getattr(logging, TAO_LOG_LEVEL, logging.INFO)
logging.basicConfig(
    level=logging.WARNING,  # Root logger: suppress third-party DEBUG logs
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logging.getLogger('nvidia_tao_core').setLevel(tao_log_level)
logger = logging.getLogger(__name__)


def get_from_ngc(key, org_name: str, enable_telemetry: bool | None = None) -> tuple[dict, str]:
    """Get signing key from token"""
    err = None
    creds = None
    try:
        # Get token
        if not org_name:
            err = f'Org Name {org_name} not valid'
            return creds, err
        if key.startswith("nvapi"):
            logger.info("Scoped key passed with and org %s", org_name)
            token = key
            r = get_user_info(key, accept_encoding="identity")
        else:
            err = ('Credentials error: Invalid NGC_PERSONAL_KEY, NGC_API_KEYs are no longer valid, '
                   'generate a personal key with Cloud Functions, NGC Catalog and Private registry services '
                   'https://org.ngc.nvidia.com/setup/personal-keys')
            return creds, err
        if r.status_code != 200:
            err = 'Credentials error: Invalid NGC_PERSONAL_KEY'
            return creds, err
        ngc_user_id = r.json().get('user', {}).get('id')
        ngc_user_name = r.json().get('user', {}).get('name')
        ngc_user_email = r.json().get('user', {}).get('email')
        if not ngc_user_id:
            err = 'Credentials error: Unknown NGC user ID'
            return creds, err
        user_id = str(uuid.uuid5(uuid.UUID(int=0), str(ngc_user_id)))
        creds = {'user_id': user_id, 'user_name': ngc_user_name, 'user_email': ngc_user_email, 'token': token}
        mongo = MongoHandler("tao", "users")
        encrypted_key = key
        config_path = os.getenv("VAULT_SECRET_PATH", None)
        if config_path:
            encryption = NVVaultEncryption(config_path)
            if encryption.check_config()[0]:
                encrypted_key = encryption.encrypt(key)
            elif not os.getenv("DEV_MODE", "False").lower() in ("true", "1"):
                err = "Vault service does not work, can't store API key"
                return creds, err

        user_query = {'id': user_id}
        user = mongo.find_one(user_query)
        user_metadata = {'id': user_id}
        if 'key' not in user or encrypted_key != user['key'].get(org_name, ""):
            user_metadata['key'] = {org_name: encrypted_key}
        if 'jwt_token' not in user or not is_token_valid(user['jwt_token']):
            logger.info("Creating new JWT Token")
            token = create_jwt_token(user_id, org_name, key)
            user_metadata['jwt_token'] = token
            creds['token'] = token
        else:
            logger.info("Using old JWT Token")
            creds['token'] = user['jwt_token']

        # If user didn't set the telemetry preference, disable for NVAIE user
        if enable_telemetry is None:
            is_nvaie_user = False
            roles = r.json().get('user', {}).get('roles', [])
            for role in roles:
                if org_name == role.get('org', {}).get('name', ''):
                    if 'NVIDIA_AI_ENTERPRISE_VIEWER' in role.get('orgRoles', []):
                        is_nvaie_user = True
                        break
            if is_nvaie_user:
                enable_telemetry = False
            else:
                enable_telemetry = True

        user_metadata['settings'] = copy.copy(user.get('settings', {}))
        user_metadata['settings'][org_name] = {'enable_telemetry': enable_telemetry}
        mongo.upsert(user_query, user_metadata)

        # Add JWT token to token_info array for session management
        extra_user_metadata = get_user_metadata_from_ngc_response(r)
        set_session(user_id, org_name, creds['token'], extra_user_metadata)
        if key.startswith("nvapi"):
            set_session(user_id, org_name, key, extra_user_metadata)

    except Exception as e:
        logger.error(traceback.format_exc())
        err = 'Credentials error: ' + str(e)
    return creds, err


def create_jwt_token(user_id, org_name, user_key):
    """Create new JWT Token for userId"""
    payload = {
        "user_id": user_id,
        "org_name": org_name,
        "exp": datetime.datetime.now(tz=datetime.timezone.utc) + datetime.timedelta(seconds=__SESSION_EXPIRY_SECONDS__)
    }
    token = jwt.encode(payload, user_key)
    return token


def decode_jwt_token(token):
    """Decode JWT Token"""
    payload, err = {}, None
    try:
        raw_payload = jwt.decode(token, options={'verify_signature': False}, algorithms=["HS256"])
        user_id, org_name = raw_payload.get('user_id'), raw_payload.get('org_name')
        user_key = get_user_key(user_id, org_name)
        if not user_key:
            err = 'Unable to retrieve user key for token'
            return {}, err
        payload = jwt.decode(token, user_key, algorithms=["HS256"])
        payload['user_key'] = user_key
    except jwt.exceptions.InvalidTokenError as e:
        err = e
    return payload, err


def is_token_valid(token):
    """Returns true if token is not yet expired, else false"""
    try:
        payload, err = decode_jwt_token(token)
        if err:
            return False
        if "exp" in payload:
            exp_time = datetime.datetime.fromtimestamp(payload["exp"], tz=datetime.timezone.utc)
            refresh_time = exp_time - datetime.timedelta(seconds=_SESSION_REFRESH_SECONDS__)
            if datetime.datetime.now(tz=datetime.timezone.utc) >= refresh_time:
                return False
            return True
        return False
    except Exception as e:
        logger.error("Exception thrown in is_token_valid is %s", str(e))
        return False
