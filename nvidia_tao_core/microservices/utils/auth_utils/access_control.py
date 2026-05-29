# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# http://<server>:<port>/<namespace>/api/v1/orgs/<org_name>/experiments?<params>
# ['', '<namespace', 'api', 'v1', 'orgs', '<org_name>', 'experiments']

"""Authentication utils access control modeules"""
from nvidia_tao_core.microservices.utils.core_utils import get_admin_key
# import re
# from nvidia_tao_core.microservices.utils.mongo_utils import MongoHandler


class AccessControlError(Exception):
    """Access Control Error"""

    pass


def validate(user_id, org_name, url, token):
    """Validate org_name requested is accessible to the provided user"""
    user_id = str(user_id)
    err = None
    if url.endswith(":status_update") or url.endswith(":log_update"):
        admin_key = get_admin_key()
        if admin_key != token:
            err = AccessControlError("Invalid token")
            return err
    if not org_name:
        err = AccessControlError("Invalid Org requested")
    # if err is None:
    #     # TODO: Implement TAO NGC user role checking here
    #     mongo = MongoHandler("tao", "users")
    #     user_metadata = mongo.find_one({'id': user_id})
    #     member_of = user_metadata.get('member_of', [])
    #     pattern = fr"{org_name}/.*:TAO_USER"
    #     if not any(re.match(pattern, member) for member in member_of):
    #         err = AccessControlError("No access granted for user in org " + org_name)
    return err
