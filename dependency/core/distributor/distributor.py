import os
import json
import gzip
import sqlite3
import tempfile
import time
from datetime import datetime

from core.lib.content import Task
from core.lib.estimation import TimeEstimator
from core.lib.common import LOGGER, FileNameConstant, FileOps, Context
from core.lib.network import http_request, NetworkAPIMethod, NetworkAPIPath
from core.lib.runtime import RuntimeContext, RuntimeLeaseClient, RuntimeLeaseError


def _indent_json_block(text, prefix='    '):
    return '\n'.join(f'{prefix}{line}' if line else prefix for line in text.splitlines())


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
    _INIT_RETRY_SECONDS = 20.0
    _INIT_RETRY_INTERVAL_SECONDS = 0.05
    _DEFAULT_RESULT_LOG_EXPORT_BATCH_SIZE = 500 # Batch size used when generating compressed export files
    _DEFAULT_RESULT_LOG_RETENTION_RECORDS = 0 # Keep the latest N task results in distributor storage to avoid unbounded growth
    _DEFAULT_RESULT_LOG_RETENTION_PRUNE_INTERVAL = 200 # Prune stale result records every N writes
    _SCHEDULER_REQUEST_TIMEOUT_SECONDS = 5.0

    def __init__(self):
        self.runtime_context = RuntimeContext.get_default()
        self.runtime_lease_client = RuntimeLeaseClient(
            self.runtime_context,
            requester=http_request,
        )
        self.scheduler_endpoint = self.runtime_context.resolve_static_endpoint('scheduler')
        self.scheduler_address = self.scheduler_endpoint.url(NetworkAPIPath.SCHEDULER_SCENARIO)
        self.record_path = FileNameConstant.DISTRIBUTOR_RECORD.value
        self.result_log_export_batch_size = max(
            1,
            int(Context.get_parameter(
                'RESULT_LOG_EXPORT_BATCH_SIZE',
                self._DEFAULT_RESULT_LOG_EXPORT_BATCH_SIZE,
                direct=False
            ))
        )
        self.result_log_retention_records = max(
            0,
            int(Context.get_parameter(
                'RESULT_LOG_RETENTION_RECORDS',
                self._DEFAULT_RESULT_LOG_RETENTION_RECORDS,
                direct=False
            ))
        )
        self.result_log_retention_prune_interval = max(
            1,
            int(Context.get_parameter(
                'RESULT_LOG_RETENTION_PRUNE_INTERVAL',
                self._DEFAULT_RESULT_LOG_RETENTION_PRUNE_INTERVAL,
                direct=False
            ))
        )
        self._writes_since_prune = 0

        # Initialize DB schema and indexes
        self._init_db()

    def _connect(self, *, autocommit=False):
        """
        Create a new SQLite connection with:
        - timeout: waits for the specified seconds if the DB is locked
        - busy_timeout: additional in-connection wait for locks
        - tuned synchronous/cache settings for better concurrency
        WAL mode is established once by `_init_db` because changing persistent
        journal metadata on every request races with other workers.
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
        # ``journal_mode`` is persistent database metadata and requires an
        # exclusive lock when it changes.  Setting it on every connection lets
        # multiple Gunicorn workers race during boot and can kill the whole Pod
        # with ``database is locked``.  `_init_db` establishes WAL once under a
        # bounded retry loop; ordinary request connections only apply local
        # connection settings.
        cur.execute(f"PRAGMA busy_timeout={self._BUSY_TIMEOUT_MS};")
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

        deadline = time.monotonic() + self._INIT_RETRY_SECONDS
        while True:
            try:
                with self._connect(autocommit=True) as conn:
                    c = conn.cursor()
                    mode = c.execute(
                        f"PRAGMA journal_mode={self._JOURNAL_MODE};"
                    ).fetchone()
                    if not mode or str(mode[0]).lower() != self._JOURNAL_MODE.lower():
                        raise RuntimeError(
                            "failed to establish Distributor SQLite WAL mode"
                        )
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
                    # Index to accelerate incremental scans by time.
                    c.execute("""
                      CREATE INDEX IF NOT EXISTS idx_records_ctime
                          ON records(ctime);
                      """)
                    conn.commit()
                return
            except sqlite3.OperationalError as exc:
                if (
                    "locked" not in str(exc).lower()
                    and "busy" not in str(exc).lower()
                ) or time.monotonic() >= deadline:
                    raise
                time.sleep(self._INIT_RETRY_INTERVAL_SECONDS)

    def distribute_data(self, cur_task: Task):
        assert cur_task, 'Current task is None'

        LOGGER.info(f'[Distribute Data] source: {cur_task.get_source_id()}  task: {cur_task.get_task_id()}')

        existing = self.get_task_record(cur_task.get_source_id(), cur_task.get_task_id())
        if existing is not None:
            if existing.get_root_uuid() != cur_task.get_root_uuid():
                raise RuntimeError(
                    f"task identity conflict for source={cur_task.get_source_id()} "
                    f"task={cur_task.get_task_id()}"
                )
            LOGGER.debug(
                f'[Distribute Data] Idempotent duplicate accepted. '
                f'source={cur_task.get_source_id()} task={cur_task.get_task_id()}'
            )
            return True

        try:
            # Fence late results before they become durable. A task whose
            # directory revision has retired must not reappear after Backend
            # has released the rollout gate at its deadline.
            self.runtime_lease_client.renew(cur_task)
        except RuntimeLeaseError as exc:
            LOGGER.warning(
                f'[Runtime Task Lease] Drop unowned result before persistence. '
                f'source={cur_task.get_source_id()} task={cur_task.get_task_id()}: {exc}'
            )
            return False

        inserted = self.save_task_record(cur_task)
        if not inserted:
            return True
        if not self.send_scenario_to_scheduler(cur_task):
            LOGGER.warning(
                f'[Scheduler Scenario] Durable task result was accepted but scenario feedback was not. '
                f'source={cur_task.get_source_id()} task={cur_task.get_task_id()}'
            )
        try:
            self.runtime_lease_client.release(cur_task)
        except RuntimeLeaseError as exc:
            # A failed release is deliberately not retried or emulated.  The
            # scheduler lease remains until its TTL or the immutable retirement
            # deadline, whichever comes first.
            LOGGER.warning(
                f'[Runtime Task Lease] Release failed; retain until TTL. '
                f'source={cur_task.get_source_id()} task={cur_task.get_task_id()}: {exc}'
            )
        return True

    def get_task_record(self, source_id, task_id):
        with self._connect() as conn:
            row = conn.execute(
                "SELECT json FROM records WHERE source_id = ? AND task_id = ?",
                (source_id, task_id),
            ).fetchone()
        return Task.deserialize(row[0]) if row else None

    def save_task_record(self, cur_task: Task):
        """Persist once and treat a repeated delivery of the same root task as idempotent."""
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
            existing = self.get_task_record(task_source_id, task_task_id)
            if existing is None or existing.get_root_uuid() != cur_task.get_root_uuid():
                raise RuntimeError(
                    f"task identity conflict for source={task_source_id} task={task_task_id}"
                )
            LOGGER.debug(
                f'[Distribute Data] Concurrent duplicate accepted. '
                f'source={task_source_id} task={task_task_id}'
            )
            return False

        self._writes_since_prune += 1
        if self.result_log_retention_records and self._writes_since_prune >= self.result_log_retention_prune_interval:
            self._prune_old_records()
            self._writes_since_prune = 0
        return True

    @staticmethod
    def record_total_end_ts(cur_task):
        TimeEstimator.record_task_ts(cur_task, 'total_end_time', is_end=False)

    def send_scenario_to_scheduler(self, cur_task: Task):
        """Send one bounded best-effort scenario update after durable storage."""
        assert cur_task, 'Current task is None'
        LOGGER.info(f'[Send Scenario] source: {cur_task.get_source_id()}  task: {cur_task.get_task_id()}')

        try:
            response = http_request(
                url=self.scheduler_address,
                method=NetworkAPIMethod.SCHEDULER_SCENARIO,
                timeout=self._SCHEDULER_REQUEST_TIMEOUT_SECONDS,
                data={'data': cur_task.serialize()})
            if not isinstance(response, dict) or response.get('accepted') is not True:
                LOGGER.warning('Scheduler did not acknowledge the scenario update.')
                return False
            return True

        except Exception as e:
            LOGGER.warning(f"Send scenario to scheduler failed: {e}")
            LOGGER.exception(e)
            return False

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

    def create_result_log_export_file(self):
        snapshot_path = self._create_result_log_snapshot()
        export_handle = tempfile.NamedTemporaryFile(
            prefix='dayu-result-log-',
            suffix='.json.gz',
            delete=False
        )
        export_path = export_handle.name
        export_handle.close()

        try:
            self._write_result_log_export(snapshot_path, export_path)
        except Exception:
            FileOps.remove_file(export_path)
            raise
        finally:
            FileOps.remove_file(snapshot_path)

        return export_path

    def clear_database(self):
        """Remove the DB file entirely."""
        FileOps.remove_file(self.record_path)
        LOGGER.info('[Distributor] Database Cleared')
        self._init_db()

    def is_database_empty(self):
        """Quick existence check."""
        return not os.path.exists(self.record_path)

    def _create_result_log_snapshot(self):
        snapshot_handle = tempfile.NamedTemporaryFile(
            prefix='dayu-result-log-snapshot-',
            suffix='.db',
            delete=False
        )
        snapshot_path = snapshot_handle.name
        snapshot_handle.close()

        try:
            with self._connect() as source_conn:
                with sqlite3.connect(snapshot_path) as snapshot_conn:
                    source_conn.backup(snapshot_conn)
                    snapshot_conn.commit()
        except Exception:
            FileOps.remove_file(snapshot_path)
            raise

        return snapshot_path

    def _iter_snapshot_records(self, snapshot_path):
        with sqlite3.connect(snapshot_path) as conn:
            c = conn.cursor()
            c.execute(
                """
                SELECT json
                FROM records
                ORDER BY ctime ASC, source_id ASC, task_id ASC
                """
            )

            while True:
                rows = c.fetchmany(self.result_log_export_batch_size)
                if not rows:
                    break
                for row in rows:
                    yield row[0]

    def _write_result_log_export(self, snapshot_path, export_path):
        with gzip.open(export_path, 'wt', encoding='utf-8') as fh:
            fh.write('[\n')
            first = True
            for payload in self._iter_snapshot_records(snapshot_path):
                try:
                    record = json.loads(payload)
                except json.JSONDecodeError:
                    LOGGER.warning('[Distributor] Skip malformed result log record during export.')
                    continue

                if not first:
                    fh.write(',\n')
                fh.write(_indent_json_block(json.dumps(record, ensure_ascii=False, indent=4)))
                first = False

            if not first:
                fh.write('\n')
            fh.write(']\n')

    def _prune_old_records(self):
        try:
            with self._connect() as conn:
                c = conn.cursor()
                c.execute(
                    """
                    DELETE FROM records
                    WHERE rowid IN (
                        SELECT rowid
                        FROM records
                        ORDER BY ctime DESC, source_id DESC, task_id DESC
                        LIMIT -1 OFFSET ?
                    )
                    """,
                    (self.result_log_retention_records,)
                )
                deleted_rows = c.rowcount if c.rowcount and c.rowcount > 0 else 0
                conn.commit()
                if deleted_rows:
                    LOGGER.info(f'[Distributor] Pruned {deleted_rows} stale result log records.')
        except Exception as e:
            LOGGER.warning(f'[Distributor] Prune old result log records failed: {e}')
            LOGGER.exception(e)
