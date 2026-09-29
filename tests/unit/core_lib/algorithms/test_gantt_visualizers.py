import ast

import pytest

from core.lib.algorithms.result_visualizer.service_device_gantt_visualizer import ServiceDeviceGanttVisualizer
from core.lib.algorithms.result_visualizer.service_gantt_visualizer import ServiceGanttVisualizer
from core.lib.common import ClassFactory, ClassType, TaskConstant, YamlOps
from core.lib.content import Task


def service_entry(name, *, execute_device='', next_nodes=None):
    return {
        'service': {
            'service_name': name,
            'execute_device': execute_device,
        },
        'next_nodes': next_nodes or [],
    }


def build_task(*, task_id=0, deployment=None):
    dag = Task.extract_dag_from_dict({
        'detector': service_entry('detector', execute_device='edge-a', next_nodes=['classifier']),
        'classifier': service_entry('classifier', execute_device='cloud-a'),
    })
    return Task(
        source_id=1,
        task_id=task_id,
        source_device='edge-a',
        all_edge_devices=['edge-a', 'edge-b'],
        dag=dag,
        deployment=deployment,
    )


@pytest.mark.unit
def test_service_gantt_defaults_to_business_services_and_uses_execute_timestamps():
    task = build_task(task_id=0)
    task.get_service('detector').set_tmp_data({
        'execute_start': 100.25,
        'execute_end': 101.75,
        'real_execute_start': 100.75,
        'real_execute_end': 101.0,
    })
    task.get_service('classifier').set_tmp_data({
        'execute_start': 102.0,
        'execute_end': 103.5,
        'real_execute_start': 102.4,
        'real_execute_end': 103.0,
    })
    task.get_service(TaskConstant.START.value).set_tmp_data({
        'execute_start': 99.0,
        'execute_end': 99.5,
    })
    task.get_service(TaskConstant.END.value).set_tmp_data({
        'execute_start': 104.0,
        'execute_end': 104.5,
    })

    visualizer = ServiceGanttVisualizer(variables=['timeline'])
    result = visualizer(task)

    assert result['timeline']['lanes'] == ['detector', 'classifier']
    assert result['timeline']['segments'] == [
        {
            'task_id': 0,
            'lane': 'detector',
            'service': 'detector',
            'device': 'edge-a',
            'start_time': 100.25,
            'end_time': 101.75,
        },
        {
            'task_id': 0,
            'lane': 'classifier',
            'service': 'classifier',
            'device': 'cloud-a',
            'start_time': 102.0,
            'end_time': 103.5,
        },
    ]
    assert visualizer(Task.deserialize(task.serialize())) == result


@pytest.mark.unit
def test_service_gantt_filters_explicit_services_and_skips_invalid_intervals():
    task = build_task(task_id=7)
    task.get_service('classifier').set_tmp_data({'execute_start': 10, 'execute_end': 12})
    task.get_service('detector').set_tmp_data({'execute_start': 13, 'execute_end': 11})

    result = ServiceGanttVisualizer(
        variables=['timeline'],
        services=[
            'classifier',
            TaskConstant.START.value,
            'missing',
            'detector',
            TaskConstant.END.value,
            'classifier',
        ],
    )(task)['timeline']

    assert result['lanes'] == ['classifier', 'detector']
    assert [segment['service'] for segment in result['segments']] == ['classifier']


@pytest.mark.unit
@pytest.mark.parametrize(
    'tmp_data',
    [
        {'execute_end': 2},
        {'execute_start': 1},
        {'execute_start': 'invalid', 'execute_end': 2},
        {'execute_start': 1, 'execute_end': float('nan')},
        {'execute_start': float('inf'), 'execute_end': 2},
        {'execute_start': True, 'execute_end': 2},
        None,
    ],
)
def test_service_gantt_keeps_lane_but_skips_incomplete_or_non_finite_intervals(tmp_data):
    task = build_task()
    task.get_service('detector').set_tmp_data(tmp_data)

    payload = ServiceGanttVisualizer(variables=['timeline'], services=['detector'])(task)['timeline']

    assert payload == {'lanes': ['detector'], 'segments': []}


@pytest.mark.unit
def test_service_device_gantt_uses_deployment_lanes_and_actual_execute_device():
    task = build_task(
        task_id=4,
        deployment={
            'detector': ['edge-a', 'edge-b'],
            'classifier': ['cloud-a'],
        },
    )
    task.get_service('detector').set_execute_device('edge-c')
    task.get_service('detector').set_tmp_data({'execute_start': 20.5, 'execute_end': 22.0})
    task.get_service('classifier').set_tmp_data({'execute_start': 30.0, 'execute_end': 31.0})

    payload = ServiceDeviceGanttVisualizer(variables=['timeline'], service='detector')(task)['timeline']

    assert payload['lanes'] == ['edge-a', 'edge-b', 'edge-c']
    assert payload['segments'] == [
        {
            'task_id': 4,
            'lane': 'edge-c',
            'service': 'detector',
            'device': 'edge-c',
            'start_time': 20.5,
            'end_time': 22.0,
        }
    ]


@pytest.mark.unit
def test_service_device_gantt_supports_legacy_deployment_shape_and_missing_service():
    task = build_task(deployment={'detector': ['detector'], 'cloud-a': ['classifier']})
    task.get_service('detector').set_tmp_data({'execute_start': 1.0, 'execute_end': 1.0})

    payload = ServiceDeviceGanttVisualizer(variables=['timeline'], service='detector')(task)['timeline']
    missing_payload = ServiceDeviceGanttVisualizer(variables=['timeline'], service='missing')(task)['timeline']

    assert payload['lanes'] == ['detector', 'edge-a']
    assert payload['segments'][0]['lane'] == 'edge-a'
    assert missing_payload == {'lanes': [], 'segments': []}


@pytest.mark.unit
@pytest.mark.parametrize('service', [TaskConstant.START.value, TaskConstant.END.value])
def test_service_device_gantt_rejects_dag_sentinel_services(service):
    task = build_task(deployment={service: ['edge-a']})

    payload = ServiceDeviceGanttVisualizer(variables=['timeline'], service=service)(task)['timeline']

    assert payload == {'lanes': [], 'segments': []}


@pytest.mark.unit
def test_driving_perception_config_exposes_both_gantt_hooks():
    config = YamlOps.read_yaml('config/visualization_configs/driving_perception_visualization_config.yaml')
    gantt_configs = {
        item['hook_name']: item
        for item in config
        if item.get('type') == 'gantt'
    }

    assert set(gantt_configs) == {'service_gantt', 'service_device_gantt'}
    assert gantt_configs['service_gantt']['variables'] == ['Service Task Timeline']
    assert 'hook_params' not in gantt_configs['service_gantt']
    assert gantt_configs['service_device_gantt']['variables'] == ['Service Device Task Timeline']
    assert ast.literal_eval(gantt_configs['service_device_gantt']['hook_params']) == {
        'service': 'traffic-detection'
    }

    for hook_name, item in gantt_configs.items():
        assert ClassFactory.is_exists(ClassType.RESULT_VISUALIZER, hook_name)
        assert item['size'] == 3
        assert item['x_axis'] == 'Time'
        assert item['y_axis'] in {'Service', 'Device'}
        assert not item.get('save_expense', False)


@pytest.mark.unit
def test_gantt_hooks_are_implemented_in_separate_modules():
    assert ServiceGanttVisualizer.__module__.endswith('.service_gantt_visualizer')
    assert ServiceDeviceGanttVisualizer.__module__.endswith('.service_device_gantt_visualizer')
