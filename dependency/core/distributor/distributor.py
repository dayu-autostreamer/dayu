import os
import sqlite3
from datetime import datetime

from core.lib.content import Task
from core.lib.estimation import TimeEstimator
from core.lib.common import LOGGER, FileNameConstant, FileOps, SystemConstant
from core.lib.network import http_request, NodeInfo, merge_address, NetworkAPIMethod, NetworkAPIPath, PortInfo


class Distributor:
    """
    Distributor with SQLite persistence.
    - Removed 'is_visited' column. Incremental reads are driven solely by time_ticket.
    - Uses WAL and busy timeouts to wait on locks instead of raising 'database is locked'.
    - All SQL is parameterized; connections are context-managed to avoid leaks.
    """

    # ---- Connection/SQLite tuning parameters ----
    _CONNECT_TIMEOUT_SECS = 5.0  # sqlite3.connect timeout: how long to wait on a locked database handle
    _BUSY_TIMEOUT_MS = 5000  # PRAGMA busy_timeout: how long SQLite will wait for locks inside a connection
    _JOURNAL_MODE = "WAL"  # Better read/write concurrency
    _SYNCHRONOUS = "NORMAL"  # Reasonable durability with good throughput (can be "FULL" if you prefer)

    def __init__(self):
        self.scheduler_hostname = NodeInfo.get_cloud_node()
        self.scheduler_port = PortInfo.get_component_port(SystemConstant.SCHEDULER.value)
        self.scheduler_address = merge_address(
            NodeInfo.hostname2ip(self.scheduler_hostname),
            port=self.scheduler_port,
            path=NetworkAPIPath.SCHEDULER_SCENARIO
        )
        self.record_path = FileNameConstant.DISTRIBUTOR_RECORD.value

        # Initialize DB schema and indexes
        self._init_db()

    def _connect(self, *, autocommit=False):
        """
        Create a new SQLite connection with:
        - timeout: waits for the specified seconds if the DB is locked
        - busy_timeout: additional in-connection wait for locks
        - WAL mode & tuned synchronous for better concurrency
        Set autocommit with isolation_level=None if you want BEGIN/COMMIT explicitly.
        """
        isolation_level = None if autocommit else ""  # None => autocommit on; "" => sqlite default (implicit transactions)
        conn = sqlite3.connect(
            self.record_path,
            timeout=self._CONNECT_TIMEOUT_SECS,
            isolation_level=isolation_level,
            detect_types=0,
            check_same_thread=True,  # set False only if you truly share the connection across threads
        )
        cur = conn.cursor()
        # Apply pragmas every time (safe & ensures settings survive across new connections)
        cur.execute(f"PRAGMA busy_timeout={self._BUSY_TIMEOUT_MS};")
        cur.execute(f"PRAGMA journal_mode={self._JOURNAL_MODE};")
        cur.execute(f"PRAGMA synchronous={self._SYNCHRONOUS};")
        # Slightly bigger page cache can help for repeated scans
        cur.execute("PRAGMA cache_size=-8000;")  # ~8MB cache; negative means KB
        conn.commit()
        return conn

    def _init_db(self):
        """Create table and indexes if not present."""
        # Ensure DB directory exists if a directory component is present
        dirpath = os.path.dirname(self.record_path)
        if dirpath:
            os.makedirs(dirpath, exist_ok=True)

        with self._connect(autocommit=True) as conn:
            c = conn.cursor()
            # Primary key is (source_id, task_id).
            c.execute("""
                      CREATE TABLE IF NOT EXISTS records
                      (
                          source_id
                          INTEGER,
                          task_id
                          INTEGER,
                          ctime
                          REAL,
                          json
                          TEXT,
                          PRIMARY
                          KEY
                      (
                          source_id,
                          task_id
                      )
                          );
                      """)
            # Index to accelerate incremental scans by time
            c.execute("""
                      CREATE INDEX IF NOT EXISTS idx_records_ctime
                          ON records(ctime);
                      """)
            conn.commit()

    def distribute_data(self, cur_task: Task):
        assert cur_task, 'Current task is None'

        LOGGER.info(f'[Distribute Data] source: {cur_task.get_source_id()}  task: {cur_task.get_task_id()}')

        self.save_task_record(cur_task)
        self.send_scenario_to_scheduler(cur_task)

    def save_task_record(self, cur_task: Task):
        """
        Insert or update a record for the task.
        NOTE: Try INSERT first, and on conflict log a warning (same behavior as original).
        """
        self.record_total_end_ts(cur_task)
        task_source_id = cur_task.get_source_id()
        task_task_id = cur_task.get_task_id()
        task_ctime = datetime.now().timestamp()

        try:
            with self._connect() as conn:
                c = conn.cursor()
                # Explicit transaction.
                c.execute("BEGIN;")
                c.execute(
                    "INSERT INTO records (source_id, task_id, ctime, json) VALUES (?, ?, ?, ?)",
                    (task_source_id, task_task_id, task_ctime, cur_task.serialize())
                )
                conn.commit()
        except sqlite3.IntegrityError:
            LOGGER.warning(
                f'[Task Name Conflict] source_id: {task_source_id}, task_id: {task_task_id} already exists.'
            )

    @staticmethod
    def record_total_end_ts(cur_task):
        TimeEstimator.record_task_ts(cur_task, 'total_end_time', is_end=False)

    def send_scenario_to_scheduler(self, cur_task: Task):
        """
        Send scenario to scheduler with simple retries and short timeouts.
        Network errors are logged and retried; DB is unaffected.
        """
        assert cur_task, 'Current task is None'
        LOGGER.info(f'[Send Scenario] source: {cur_task.get_source_id()}  task: {cur_task.get_task_id()}')

        try:
            http_request(
                url=self.scheduler_address,
                method=NetworkAPIMethod.SCHEDULER_SCENARIO,
                data={'data': cur_task.serialize()})

        except Exception as e:
            LOGGER.warning(f"Send scenario to scheduler failed: {e}")
            LOGGER.exception(e)

    @staticmethod
    def record_transmit_ts(cur_task):
        assert cur_task, 'Current task is None'
        duration = TimeEstimator.record_dag_ts(cur_task, is_end=True, sub_tag='transmit')
        cur_task.save_transmit_time(duration)
        LOGGER.info(f'[Source {cur_task.get_source_id()} / Task {cur_task.get_task_id()}] '
                    f'record transmit time of stage {cur_task.get_flow_index()}: {duration:.3f}s')

    def query_result(self, time_ticket, size):
        """
        Incremental query by time_ticket.
        - Returns records with ctime > time_ticket ordered ASC.
        - If size > 0, apply LIMIT at SQL level for efficiency.
        - new_time_ticket equals the last returned record's ctime (or remains unchanged if no records).
        """
        if self.is_database_empty():
            return {'result': [], 'time_ticket': time_ticket, 'size': 0}

        # Read-only transaction is sufficient; we still want consistent snapshot
        with self._connect() as conn:
            c = conn.cursor()

            if size and size > 0:
                c.execute(
                    """
                    SELECT source_id, task_id, ctime, json
                    FROM records
                    WHERE ctime > ?
                    ORDER BY ctime DESC LIMIT ?
                    """,
                    (time_ticket, size)
                )
                rows = c.fetchall()
                rows = rows[::-1]
            else:
                c.execute(
                    """
                    SELECT source_id, task_id, ctime, json
                    FROM records
                    WHERE ctime > ?
                    ORDER BY ctime ASC
                    """,
                    (time_ticket,)
                )
                rows = c.fetchall()

        if not rows:
            LOGGER.debug(f'No new records, last file time unchanged: {time_ticket}')
            return {'result': [], 'time_ticket': time_ticket, 'size': 0}

        # Prepare response
        json_results = [r[3] for r in rows]
        new_time_ticket = rows[-1][2]  # ctime of the last returned row
        LOGGER.debug(f'Last file time updated: {new_time_ticket}')

        return {
            'result': json_results,
            'time_ticket': new_time_ticket,
            'size': len(json_results)
        }

    def query_results_by_time(self, start_time, end_time, source_id=None):
        """
        Query records within a specific time range, optionally filtered by source_id.
        """
        if self.is_database_empty():
            return {'result': [], 'size': 0}

        with self._connect() as conn:
            c = conn.cursor()
            if source_id is not None:
                c.execute(
                    """
                    SELECT json
                    FROM records
                    WHERE ctime BETWEEN ? AND ? AND source_id = ?
                    ORDER BY ctime ASC
                    """,
                    (start_time, end_time, source_id)
                )
            else:
                c.execute(
                    """
                    SELECT json
                    FROM records
                    WHERE ctime BETWEEN ? AND ?
                    ORDER BY ctime ASC
                    """,
                    (start_time, end_time)
                )
            results = [row[0] for row in c.fetchall()]

        return {'result': results, 'size': len(results)}

    def query_all_result(self):
        """
        Return all records ordered by (source_id, task_id).
        """
        if self.is_database_empty():
            return {'result': [], 'size': 0}

        with self._connect() as conn:
            c = conn.cursor()
            c.execute(
                """
                SELECT json
                FROM records
                ORDER BY source_id ASC, task_id ASC
                """
            )
            results = [row[0] for row in c.fetchall()]

        return {'result': results, 'size': len(results)}

    def clear_database(self):
        """Remove the DB file entirely."""
        FileOps.remove_file(self.record_path)
        LOGGER.info('[Distributor] Database Cleared')
        self._init_db()

    def is_database_empty(self):
        """Quick existence check."""
        return not os.path.exists(self.record_path)
