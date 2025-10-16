import json


class Service:
    def __init__(self, service_name,
                 execute_device='',
                 transmit_time=0,
                 execute_time=0,
                 real_execute_time=0,
                 content: object = None,
                 temp=None):
        self.__service_name = service_name
        self.__execute_device = execute_device
        self.__transmit_time = transmit_time

        # execute_time means duration of service execution from controller to controller
        # includes waiting time in queue and other system costs
        self.__execute_time = execute_time

        # real_execute_time only includes real execute time of service in processor
        self.__real_execute_time = real_execute_time

        # result data of service
        self.__content = content

        # temporary data (main for time tickets)
        self.__tmp_data = temp if temp else {}

    def get_service_name(self):
        return self.__service_name

    def set_service_name(self, service_name):
        self.__service_name = service_name

    def get_execute_device(self):
        return self.__execute_device

    def set_execute_device(self, execute_device):
        self.__execute_device = execute_device

    def set_transmit_time(self, transmit_time):
        assert transmit_time >= 0, f'transmit time of service "{self.__service_name}" is negative: {transmit_time}!'
        self.__transmit_time = transmit_time

    def set_execute_time(self, execute_time):
        assert execute_time >= 0, f'execute time of service "{self.__service_name}" is negative: {execute_time}!'
        self.__execute_time = execute_time

    def set_real_execute_time(self, real_execute_time):
        assert real_execute_time >= 0, \
            f' real execute time of service "{self.__service_name}" is negative: {real_execute_time}!'
        self.__real_execute_time = real_execute_time

    def get_transmit_time(self):
        return self.__transmit_time

    def get_execute_time(self):
        return self.__execute_time

    def get_real_execute_time(self):
        return self.__real_execute_time

    def get_service_total_time(self):
        return self.__transmit_time + self.__execute_time

    def get_content_data(self):
        return self.__content

    def set_content_data(self, content):
        self.__content = content

    def get_tmp_data(self):
        return self.__tmp_data

    def set_tmp_data(self, tmp):
        self.__tmp_data = tmp

    def record_time_ticket(self, tag: str, duration: float):
        """
        Record a time ticket in the temporary data.

        :param tag: The name of the time ticket.
        :param duration: The duration to record for the time ticket.
        """
        assert tag not in self.__tmp_data, f'Time ticket "{tag}" already exists in temporary data!'
        self.__tmp_data[tag] = duration

    def to_dict(self):
        return {
            'service_name': self.get_service_name(),
            'execute_device': self.get_execute_device(),
            'execute_data': {'transmit_time': self.get_transmit_time(),
                             'execute_time': self.get_execute_time(),
                             'real_execute_time': self.get_real_execute_time()},
            'content': self.get_content_data(),
            'tmp_data': self.get_tmp_data()
        }

    @classmethod
    def from_dict(cls, dag_dict: dict):
        service = Service(service_name=dag_dict['service_name'])

        service.set_execute_device(dag_dict['execute_device']) if 'execute_device' in dag_dict else None
        service.set_transmit_time(dag_dict['execute_data']['transmit_time']) \
            if 'execute_data' in dag_dict and 'transmit_time' in dag_dict['execute_data'] else None
        service.set_execute_time(dag_dict['execute_data']['execute_time']) \
            if 'execute_data' in dag_dict and 'execute_time' in dag_dict['execute_data'] else None
        service.set_real_execute_time(dag_dict['execute_data']['real_execute_time']) \
            if 'execute_data' in dag_dict and 'real_execute_time' in dag_dict['execute_data'] else None
        service.set_content_data(dag_dict['content']) if 'content' in dag_dict else None
        service.set_tmp_data(dag_dict['tmp_data']) if 'tmp_data' in dag_dict else None
        return service

    def serialize(self):
        return json.dumps(self.to_dict())

    @classmethod
    def deserialize(cls, data: str):
        return cls.from_dict(json.loads(data))

    def __hash__(self):
        return hash(self.__service_name)

    def __eq__(self, other):
        return isinstance(other, Service) and self.__service_name == other.__service_name

    def __repr__(self):
        return self.__service_name
