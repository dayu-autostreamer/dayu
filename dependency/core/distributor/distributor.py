import os
import sqlite3
from datetime import datetime

from core.lib.content import Task
from core.lib.estimation import TimeEstimator
from core.lib.common import LOGGER, FileNameConstant, FileOps, SystemConstant
from core.lib.network import http_request, NodeInfo, merge_address, NetworkAPIMethod, NetworkAPIPath, PortInfo


class Distributor:
    def __init__(self):

        self.scheduler_hostname = NodeInfo.get_cloud_node()
        self.scheduler_port = PortInfo.get_component_port(SystemConstant.SCHEDULER.value)
        self.scheduler_address = merge_address(NodeInfo.hostname2ip(self.scheduler_hostname),
                                               port=self.scheduler_port,
                                               path=NetworkAPIPath.SCHEDULER_SCENARIO)

        self.record_path = FileNameConstant.DISTRIBUTOR_RECORD.value

    def distribute_data(self, cur_task: Task):
        assert cur_task, 'Current task is None'

        LOGGER.info(f'[Distribute Data] source: {cur_task.get_source_id()}  task: {cur_task.get_task_id()}')

        self.save_task_record(cur_task)
        self.send_scenario_to_scheduler(cur_task)

    def save_task_record(self, cur_task: Task):
        self.record_total_end_ts(cur_task)
        task_source_id = cur_task.get_source_id()
        task_task_id = cur_task.get_task_id()
        task_ctime = datetime.now().timestamp()

        conn = sqlite3.connect(self.record_path)
        c = conn.cursor()

        create_table_sql = '''
            CREATE TABLE IF NOT EXISTS records (
                source_id INTEGER,
                task_id INTEGER,
                ctime REAL,
                json TEXT,
                is_visited BOOL,
                PRIMARY KEY (source_id, task_id)
            );
        '''
        c.execute(create_table_sql)

        try:
            c.execute('INSERT INTO records VALUES (?, ?, ?, ?, ?)',
                      (task_source_id, task_task_id, task_ctime, cur_task.serialize(), False))
            conn.commit()
        except sqlite3.IntegrityError:
            LOGGER.warning(f'[Task Name Conflict] source_id: {task_source_id}, task_id: {task_task_id} '
                           f'has already existed in database.')
        finally:
            conn.close()

    @staticmethod
    def record_total_end_ts(cur_task):
        TimeEstimator.record_task_ts(cur_task,
                                     'total_end_time',
                                     is_end=False)

    def send_scenario_to_scheduler(self, cur_task):
        assert cur_task, 'Current task is None'

        LOGGER.info(f'[Send Scenario] source: {cur_task.get_source_id()}  task: {cur_task.get_task_id()}')
        http_request(url=self.scheduler_address,
                     method=NetworkAPIMethod.SCHEDULER_SCENARIO,
                     data={'data': cur_task.serialize()})

    @staticmethod
    def record_transmit_ts(cur_task):
        assert cur_task, 'Current task is None'

        duration = TimeEstimator.record_dag_ts(cur_task, is_end=True, sub_tag='transmit')

        cur_task.save_transmit_time(duration)
        LOGGER.info(f'[Source {cur_task.get_source_id()} / Task {cur_task.get_task_id()}] '
                    f'record transmit time of stage {cur_task.get_flow_index()}: {duration:.3f}s')

    def query_result(self, time_ticket, size):
        if self.is_database_empty():
            return {
                'result': [],
                'time_ticket': time_ticket,
                'size': 0
            }

        conn = sqlite3.connect(self.record_path)
        c = conn.cursor()

        # query unvisited records
        query_sql = f'''
            SELECT source_id, task_id, ctime, json 
            FROM records 
            WHERE ctime > {time_ticket} AND is_visited = 0
            ORDER BY ctime ASC;
        '''
        c.execute(query_sql)
        results = c.fetchall()

        n = len(results)
        flat_ids = []
        for row in results:
            flat_ids.append(row[0])
            flat_ids.append(row[1])
        ctime_results = [row[2] for row in results]
        json_results = [row[3] for row in results]

        if n > 0:
            # mark visited records
            update_sql = f'''
                UPDATE records
                SET is_visited = 1
                WHERE (source_id, task_id) IN (VALUES {','.join(['(?,?)'] * n)});
            '''
            c.execute(update_sql, flat_ids)
            conn.commit()

            conn.close()

        # prepare response
        if len(ctime_results) > 0:
            new_time_ticket = ctime_results[-1]
        else:
            new_time_ticket = time_ticket
        LOGGER.debug(f'last file time: {new_time_ticket}')
        if size > 0:
            json_results = json_results[:size]

        return {'result': json_results,
                'time_ticket': new_time_ticket,
                'size': len(json_results)
                }

    def query_all_result(self):
        if self.is_database_empty():
            return {
                'result': [],
                'size': 0
            }
        conn = sqlite3.connect(self.record_path)
        c = conn.cursor()

        # query records, ordered by create time
        query_sql = f'''
            SELECT json 
            FROM records 
            ORDER BY source_id ASC, task_id ASC;
        '''
        c.execute(query_sql)
        results = c.fetchall()

        json_results = [row[0] for row in results]

        conn.close()

        return {'result': json_results,
                'size': len(json_results)
                }

    def clear_database(self):
        FileOps.remove_file(self.record_path)
        LOGGER.info('[Distributor] Database Cleared')

    def is_database_empty(self):
        return not os.path.exists(self.record_path)
