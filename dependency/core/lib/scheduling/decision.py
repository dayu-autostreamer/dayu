"""Stable identities for per-task scheduling decisions."""

import hashlib
import json
import uuid


def canonical_digest(value):
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(',', ':'),
    )
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()


def build_schedule_decision(
    request,
    plan,
    deployment_version,
    runtime_directory_revision,
):
    request = request if isinstance(request, dict) else {}
    task_context = request.get('task_context')
    task_context = task_context if isinstance(task_context, dict) else {}
    source_id = request.get('source_id', task_context.get('source_id'))
    task_id = task_context.get('task_id')
    root_uuid = str(task_context.get('root_uuid') or '')
    plan_digest = canonical_digest({
        'plan': plan,
        'deployment_version': deployment_version,
        'runtime_directory_revision': runtime_directory_revision,
    })
    identity = root_uuid or str(uuid.uuid4())
    decision_id = str(uuid.uuid5(
        uuid.NAMESPACE_URL,
        'dayu:schedule:{}:{}:{}'.format(source_id, identity, plan_digest),
    ))
    return {
        'decision_id': decision_id,
        'plan_digest': plan_digest,
        'source_id': source_id,
        'task_id': task_id,
        'root_uuid': root_uuid,
    }
