# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""API response filtering modules"""
from datetime import datetime


def parse_datetime(d):
    """Parse the created_on field to a datetime object"""
    created_on = d.get('created_on')
    if isinstance(created_on, datetime):
        return created_on
    try:
        return datetime.fromisoformat(created_on.replace('Z', '+00:00'))
    except ValueError:
        return datetime.fromisoformat(created_on)


def apply(args, data):
    """Filter results based on the arguments provided"""
    filter_sort = args.get('sort')
    filter_name = args.get('name')
    filter_type = args.get('type')
    filter_arch = args.get('network_arch')
    filter_read_only = args.get('read_only')
    filter_shared = args.get('shared')
    filter_format = args.get('format')
    filter_cloud_type = args.get('cloud_type')
    filter_tag = args.get('tags')
    filter_status = args.get('status')
    filter_search = args.get('search')

    if filter_search is not None:
        data = list(filter(
            lambda d: (
                filter_search.lower() in d.get('name', '').lower() or
                filter_search.lower() in d.get('description', '').lower() or
                filter_search in d.get('id', '')
            ),
            data
        ))

    if filter_name is not None:
        if filter_name.startswith('!'):
            filter_name = filter_name[1:]
            data = list(filter(lambda d: d.get('name') != filter_name, data))
        else:
            data = list(filter(lambda d: d.get('name') == filter_name, data))
    if filter_type is not None:
        if filter_type.startswith('!'):
            filter_type = filter_type[1:]
            data = list(filter(lambda d: d.get('type') != filter_type, data))
        else:
            data = list(filter(lambda d: d.get('type') == filter_type, data))
    if filter_arch is not None:
        if filter_arch.startswith('!'):
            filter_arch = filter_arch[1:]
            data = list(filter(lambda d: d.get('network_arch') != filter_arch, data))
        else:
            data = list(filter(lambda d: d.get('network_arch') == filter_arch, data))

    if filter_read_only is not None:
        filter_read_only_as_boolean = filter_read_only == 'true'
        data = list(filter(lambda d: d.get('read_only') == filter_read_only_as_boolean, data))

    if filter_shared is not None:
        filter_shared_as_boolean = filter_shared == 'true'
        data = list(filter(lambda d: d.get('shared') == filter_shared_as_boolean, data))

    if filter_format is not None:
        if filter_format.startswith('!'):
            filter_format = filter_format[1:]
            data = list(filter(lambda d: d.get('format') != filter_format, data))
        else:
            data = list(filter(lambda d: d.get('format') == filter_format, data))

    if filter_cloud_type is not None:
        if filter_cloud_type.startswith('!'):
            filter_cloud_type = filter_cloud_type[1:]
            data = list(filter(lambda d: d.get('cloud_type') != filter_cloud_type, data))
        else:
            data = list(filter(lambda d: d.get('cloud_type') == filter_cloud_type, data))

    if filter_tag is not None:
        if filter_tag.startswith('!'):
            filter_tag = filter_tag[1:]
            data = list(filter(lambda d: filter_tag not in d.get('tags', []), data))
        else:
            data = list(filter(lambda d: filter_tag in d.get('tags', []), data))

    if filter_status is not None:
        if filter_status.startswith('!'):
            filter_status = filter_status[1:]
            data = list(filter(lambda d: d.get('status') != filter_status, data))
        else:
            data = list(filter(lambda d: d.get('status') == filter_status, data))

    if filter_sort == 'name-ascending':
        data = sorted(data, key=lambda d: '' + d.get('name') + ':' + d.get('version'), reverse=False)
    elif filter_sort == 'name-descending':
        data = sorted(data, key=lambda d: '' + d.get('name') + ':' + d.get('version'), reverse=True)
    elif filter_sort == 'status-ascending':
        data = sorted(data, key=lambda d: '' + d.get('status'), reverse=False)
    elif filter_sort == 'status-descending':
        data = sorted(data, key=lambda d: '' + d.get('status'), reverse=True)
    elif filter_sort == 'date-ascending':
        data = sorted(data, key=parse_datetime, reverse=False)
    else:  # elif filter_sort == 'date-descending':
        data = sorted(data, key=parse_datetime, reverse=True)
    return data
