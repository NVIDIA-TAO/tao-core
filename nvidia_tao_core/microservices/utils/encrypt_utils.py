# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""NV Vault encryption."""

import os
import json
import logging

from Crypto.Cipher import AES
from base64 import b64encode, b64decode

# Configure logging
TAO_LOG_LEVEL = os.getenv('TAO_LOG_LEVEL', 'INFO').upper()
tao_log_level = getattr(logging, TAO_LOG_LEVEL, logging.INFO)
logging.basicConfig(
    level=logging.WARNING,  # Root logger: suppress third-party DEBUG logs
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logging.getLogger('nvidia_tao_core').setLevel(tao_log_level)
logger = logging.getLogger(__name__)


class NVVaultEncryption:
    """A class to encrypt/decrypt messages uses nv vault.

    When a NV vault sidecar is set,
    it will inject secrets to the file specified by env variable VAULT_SECRET_PATH.
    And the content of the file is like:
    ```
    {"kv": {
        "k8s": {
            "aes_key": "123456"
        }
    }
    ```
    This class uses the AES algorithm to encrypt the message, since the encrypt/decrypt only
    occurs inside this k8s cluster and doesn't need to exchange any info with external servers.
    """

    AES_ENCRYPT_HEADER = b"tao_service_k8s_header"
    AES_ENCRYPT_TEXT_KEYS = ['nonce', 'header', 'ciphertext', 'tag']

    def __init__(self, config_path):
        """Create a encryption class with the given config path."""
        self.config_path = config_path

    def _get_config_content(self):
        """Get the content from the config file."""
        if self.config_path is None or not os.path.exists(self.config_path):
            return None
        try:
            with open(self.config_path, "r", encoding="utf8") as fp:
                config_content = json.load(fp)
            return config_content
        except Exception as e:
            logger.error("Exception thrown in get_config_content is %s", str(e))
            return None

    def _get_k8s_keys(self):
        """Get keys for the k8s clusters."""
        config_content = self._get_config_content()
        kv_secrets = config_content.get("kv", None) if isinstance(config_content, dict) else None
        k8s_secrets = kv_secrets.get("k8s", None) if isinstance(kv_secrets, dict) else None
        return k8s_secrets

    def _get_key(self):
        """Get the key to encrypt messages by the AES algorithm."""
        k8s_secrets = self._get_k8s_keys()
        aes_key = k8s_secrets.get("aes_key", None) if isinstance(k8s_secrets, dict) else None
        return b64decode(aes_key)

    def check_config(self):
        """Check if the given config format is correct."""
        msg = ""
        config_is_ok = False

        if self._get_config_content() is None:
            msg = f"Cannot open file {self.config_path}."
            return config_is_ok, msg

        if self._get_key() is None:
            msg = f"Cannot find the AES key from {self.config_path}."
            return config_is_ok, msg

        config_is_ok = True
        msg = "The config file format is correct."
        return config_is_ok, msg

    def encrypt(self, message):
        """Encrypt the given message and return the encrypted message."""
        key = self._get_key()
        message = message.encode("utf8")
        header = self.AES_ENCRYPT_HEADER
        cipher = AES.new(key, AES.MODE_EAX)
        cipher.update(header)
        ciphertext, tag = cipher.encrypt_and_digest(message)

        json_k = self.AES_ENCRYPT_TEXT_KEYS
        json_v = [b64encode(x).decode('utf-8') for x in (cipher.nonce, header, ciphertext, tag)]
        result = json.dumps(dict(zip(json_k, json_v)))
        return result

    def decrypt(self, message):
        """Decrypt the given message and return the decrypted message."""
        message = json.loads(message)
        key = self._get_key()
        json_k = self.AES_ENCRYPT_TEXT_KEYS
        jv = {k: b64decode(message[k]) for k in json_k}
        nonce, header, ciphertext, tag = self.AES_ENCRYPT_TEXT_KEYS
        cipher = AES.new(key, AES.MODE_EAX, nonce=jv[nonce])
        cipher.update(jv[header])
        plaintext = cipher.decrypt_and_verify(jv[ciphertext], jv[tag])
        return plaintext.decode("utf8")
