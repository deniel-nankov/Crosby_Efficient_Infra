"""
Integration Tests for Compliance RAG System
============================================
Tests database connectivity and basic operations.

Run with: pytest tests/test_integration.py -v -m integration
"""

import pytest
import os
import json
from datetime import date, datetime, timezone


# =============================================================================
# DATABASE HELPERS
# =============================================================================

def get_postgres_connection():
    """Try to create a PostgreSQL connection."""
    try:
        import psycopg2
        conn = psycopg2.connect(
            host=os.environ.get("POSTGRES_HOST", "localhost"),
            port=int(os.environ.get("POSTGRES_PORT", "5432")),
            database=os.environ.get("POSTGRES_DB", "compliance"),
            user=os.environ.get("POSTGRES_USER", "compliance_user"),
            password=os.environ.get("POSTGRES_PASSWORD", "compliance_dev_password_123"),
        )
        return conn
    except Exception:
        return None


def get_redis_connection():
    """Try to create a Redis connection."""
    try:
        import redis
        r = redis.Redis(
            host=os.environ.get("REDIS_HOST", "localhost"),
            port=int(os.environ.get("REDIS_PORT", "6379")),
            db=0,
        )
        r.ping()
        return r
    except Exception:
        return None


# Check if packages are installed
try:
    import psycopg2
    PSYCOPG2_AVAILABLE = True
except ImportError:
    PSYCOPG2_AVAILABLE = False

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture(scope="module")
def postgres_conn():
    """Provide PostgreSQL connection for tests."""
    if not PSYCOPG2_AVAILABLE:
        pytest.skip("psycopg2 not installed")
    
    conn = get_postgres_connection()
    if conn is None:
        pytest.skip("PostgreSQL not available")
    
    yield conn
    conn.close()


@pytest.fixture(scope="module")
def redis_conn():
    """Provide Redis connection for tests."""
    if not REDIS_AVAILABLE:
        pytest.skip("redis not installed")
    
    conn = get_redis_connection()
    if conn is None:
        pytest.skip("Redis not available")
    
    yield conn


# =============================================================================
# POSTGRESQL TESTS
# =============================================================================

@pytest.mark.integration
class TestPostgreSQLConnection:
    """Test PostgreSQL connectivity and schema."""
    
    def test_connection_works(self, postgres_conn):
        """Should be able to connect to PostgreSQL."""
        cursor = postgres_conn.cursor()
        cursor.execute("SELECT 1")
        result = cursor.fetchone()
        cursor.close()
        assert result[0] == 1
    
    def test_required_tables_exist(self, postgres_conn):
        """Required tables should exist."""
        required_tables = [
            "control_runs",
            "control_results", 
            "control_definitions",
            "exceptions",
            "approvals",
            "audit_log",
        ]
        
        cursor = postgres_conn.cursor()
        cursor.execute("""
            SELECT table_name FROM information_schema.tables 
            WHERE table_schema = 'public' AND table_type = 'BASE TABLE'
        """)
        existing_tables = [row[0] for row in cursor.fetchall()]
        cursor.close()
        
        for table in required_tables:
            assert table in existing_tables, f"Missing table: {table}"
    
    def test_control_runs_table_structure(self, postgres_conn):
        """control_runs table should have required columns."""
        required_columns = [
            "run_id", "run_code", "run_type", "run_date", "status", "config_hash"
        ]
        
        cursor = postgres_conn.cursor()
        cursor.execute("""
            SELECT column_name FROM information_schema.columns 
            WHERE table_name = 'control_runs'
        """)
        existing_columns = [row[0] for row in cursor.fetchall()]
        cursor.close()
        
        for col in required_columns:
            assert col in existing_columns, f"Missing column: {col}"


@pytest.mark.integration
class TestPostgreSQLOperations:
    """Test PostgreSQL CRUD operations."""
    
    def test_insert_and_read_control_run(self, postgres_conn):
        """Should be able to insert and read control run."""
        import uuid
        
        run_id = str(uuid.uuid4())
        run_code = f"TEST-{run_id[:8]}"
        
        cursor = postgres_conn.cursor()
        
        try:
            # Insert
            cursor.execute("""
                INSERT INTO control_runs (
                    run_id, run_code, run_type, run_date,
                    snowflake_snapshot_id, snowflake_snapshot_ts,
                    status, config_hash, executor_service, executor_version
                ) VALUES (%s, %s, 'ad-hoc', CURRENT_DATE,
                    'SNAP-TEST', NOW(), 'running', 'test-hash', 'test', '1.0.0')
            """, (run_id, run_code))
            postgres_conn.commit()
            
            # Read
            cursor.execute("SELECT run_code, status FROM control_runs WHERE run_id = %s", (run_id,))
            result = cursor.fetchone()
            
            assert result is not None
            assert result[0] == run_code
            assert result[1] == "running"
            
        finally:
            cursor.execute("DELETE FROM control_runs WHERE run_id = %s", (run_id,))
            postgres_conn.commit()
            cursor.close()
    
    def test_update_control_run_status(self, postgres_conn):
        """Should be able to update control run status."""
        import uuid
        
        run_id = str(uuid.uuid4())
        run_code = f"TEST-UPD-{run_id[:8]}"
        
        cursor = postgres_conn.cursor()
        
        try:
            # Insert
            cursor.execute("""
                INSERT INTO control_runs (
                    run_id, run_code, run_type, run_date,
                    snowflake_snapshot_id, snowflake_snapshot_ts,
                    status, config_hash, executor_service, executor_version
                ) VALUES (%s, %s, 'ad-hoc', CURRENT_DATE,
                    'SNAP-TEST', NOW(), 'running', 'test-hash', 'test', '1.0.0')
            """, (run_id, run_code))
            
            # Update
            cursor.execute("""
                UPDATE control_runs SET status = 'completed', 
                run_timestamp_end = NOW() WHERE run_id = %s
            """, (run_id,))
            postgres_conn.commit()
            
            # Verify
            cursor.execute("SELECT status FROM control_runs WHERE run_id = %s", (run_id,))
            result = cursor.fetchone()
            
            assert result[0] == "completed"
            
        finally:
            cursor.execute("DELETE FROM control_runs WHERE run_id = %s", (run_id,))
            postgres_conn.commit()
            cursor.close()
    
    def test_status_constraint_enforced(self, postgres_conn):
        """Invalid status should be rejected."""
        import uuid
        
        run_id = str(uuid.uuid4())
        cursor = postgres_conn.cursor()
        
        try:
            with pytest.raises(Exception):
                cursor.execute("""
                    INSERT INTO control_runs (
                        run_id, run_code, run_type, run_date,
                        snowflake_snapshot_id, snowflake_snapshot_ts,
                        status, config_hash, executor_service, executor_version
                    ) VALUES (%s, 'TEST', 'ad-hoc', CURRENT_DATE,
                        'SNAP', NOW(), 'invalid_status', 'hash', 'test', '1.0.0')
                """, (run_id,))
                postgres_conn.commit()
        finally:
            postgres_conn.rollback()
            cursor.close()
    
    def test_hash_stored_correctly(self, postgres_conn):
        """Hash values should be stored and retrieved correctly."""
        import uuid
        import hashlib
        
        run_id = str(uuid.uuid4())
        original_hash = hashlib.sha256(b"test data").hexdigest()
        
        cursor = postgres_conn.cursor()
        
        try:
            cursor.execute("""
                INSERT INTO control_runs (
                    run_id, run_code, run_type, run_date,
                    snowflake_snapshot_id, snowflake_snapshot_ts,
                    status, config_hash, executor_service, executor_version
                ) VALUES (%s, 'HASH-TEST', 'ad-hoc', CURRENT_DATE,
                    'SNAP', NOW(), 'completed', %s, 'test', '1.0.0')
            """, (run_id, original_hash))
            postgres_conn.commit()
            
            cursor.execute("SELECT config_hash FROM control_runs WHERE run_id = %s", (run_id,))
            stored_hash = cursor.fetchone()[0]
            
            assert stored_hash == original_hash
            assert len(stored_hash) == 64
            
        finally:
            cursor.execute("DELETE FROM control_runs WHERE run_id = %s", (run_id,))
            postgres_conn.commit()
            cursor.close()


# =============================================================================
# REDIS TESTS
# =============================================================================

@pytest.mark.integration
class TestRedisConnection:
    """Test Redis connectivity."""
    
    def test_connection_works(self, redis_conn):
        """Should be able to ping Redis."""
        assert redis_conn.ping() == True
    
    def test_set_and_get(self, redis_conn):
        """Should be able to set and get values."""
        key = "test:compliance:value"
        value = "test_123"
        
        try:
            redis_conn.set(key, value)
            result = redis_conn.get(key)
            assert result.decode() == value
        finally:
            redis_conn.delete(key)
    
    def test_json_storage(self, redis_conn):
        """Should be able to store JSON data."""
        key = "test:compliance:json"
        data = {"run_id": "test-123", "status": "running"}
        
        try:
            redis_conn.set(key, json.dumps(data))
            result = json.loads(redis_conn.get(key))
            assert result["run_id"] == "test-123"
        finally:
            redis_conn.delete(key)
    
    def test_ttl_works(self, redis_conn):
        """TTL should be set correctly."""
        key = "test:compliance:ttl"
        
        try:
            redis_conn.setex(key, 60, "ephemeral")
            ttl = redis_conn.ttl(key)
            assert 0 < ttl <= 60
        finally:
            redis_conn.delete(key)


# =============================================================================
# CONTROL DEFINITIONS TESTS
# =============================================================================

@pytest.mark.integration
class TestControlDefinitions:
    """Test control definitions table."""
    
    def test_can_query_control_definitions(self, postgres_conn):
        """Should be able to query control definitions."""
        cursor = postgres_conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM control_definitions")
        count = cursor.fetchone()[0]
        cursor.close()
        
        # Should have some controls defined
        assert count >= 0
    
    def test_can_insert_control_definition(self, postgres_conn):
        """Should be able to insert a control definition."""
        import uuid
        
        control_id = str(uuid.uuid4())
        cursor = postgres_conn.cursor()
        
        try:
            cursor.execute("""
                INSERT INTO control_definitions (
                    control_id, control_code, control_name, control_category,
                    description, threshold_type, threshold_value, threshold_operator,
                    frequency, is_active, effective_date, created_by, updated_by
                ) VALUES (
                    %s, 'TEST_CTRL_001', 'Test Control', 'concentration',
                    'Test control description', 'percentage', 10.0, 'gt',
                    'daily', true, CURRENT_DATE, 'integration_test', 'integration_test'
                )
            """, (control_id,))
            postgres_conn.commit()
            
            cursor.execute("SELECT control_name FROM control_definitions WHERE control_id = %s", (control_id,))
            result = cursor.fetchone()
            
            assert result[0] == "Test Control"
            
        finally:
            cursor.execute("DELETE FROM control_definitions WHERE control_id = %s", (control_id,))
            postgres_conn.commit()
            cursor.close()


# =============================================================================
# END-TO-END WORKFLOW TEST
# =============================================================================

@pytest.mark.integration
class TestEndToEndWorkflow:
    """Test complete workflow."""
    
    def test_full_control_run_workflow(self, postgres_conn):
        """Test complete control run from start to finish."""
        import uuid
        
        run_id = str(uuid.uuid4())
        control_id = str(uuid.uuid4())
        result_id = str(uuid.uuid4())
        run_code = f"E2E-{run_id[:8]}"
        
        cursor = postgres_conn.cursor()
        
        try:
            # 1. Create control definition
            cursor.execute("""
                INSERT INTO control_definitions (
                    control_id, control_code, control_name, control_category,
                    description, threshold_type, threshold_value, threshold_operator,
                    frequency, is_active, effective_date, created_by, updated_by
                ) VALUES (%s, 'E2E_TEST', 'E2E Test Control', 'concentration',
                    'End-to-end test', 'percentage', 10.0, 'gt', 'daily', true, CURRENT_DATE, 'integration_test', 'integration_test')
            """, (control_id,))
            
            # 2. Create control run
            cursor.execute("""
                INSERT INTO control_runs (
                    run_id, run_code, run_type, run_date,
                    snowflake_snapshot_id, snowflake_snapshot_ts,
                    status, config_hash, executor_service, executor_version
                ) VALUES (%s, %s, 'scheduled', CURRENT_DATE,
                    'SNAP-E2E', NOW(), 'running', 'e2e-hash', 'e2e-test', '1.0.0')
            """, (run_id, run_code))
            
            # 3. Create control result
            cursor.execute("""
                INSERT INTO control_results (
                    result_id, run_id, control_id,
                    calculated_value, threshold_value, threshold_operator,
                    result_status, evidence_query_hash, evidence_row_count,
                    computation_sql, computation_duration_ms
                ) VALUES (%s, %s, %s, 8.5, 10.0, 'gt', 'pass', 
                    'test-hash', 1, 'SELECT 1', 100)
            """, (result_id, run_id, control_id))
            
            # 4. Update run to completed
            cursor.execute("""
                UPDATE control_runs SET status = 'completed',
                    total_controls = 1, controls_passed = 1,
                    run_timestamp_end = NOW()
                WHERE run_id = %s
            """, (run_id,))
            
            postgres_conn.commit()
            
            # 5. Verify
            cursor.execute("""
                SELECT cr.status, cr.controls_passed, res.result_status
                FROM control_runs cr
                JOIN control_results res ON cr.run_id = res.run_id
                WHERE cr.run_id = %s
            """, (run_id,))
            
            result = cursor.fetchone()
            
            assert result[0] == "completed"
            assert result[1] == 1
            assert result[2] == "pass"
            
        finally:
            # Cleanup
            cursor.execute("DELETE FROM control_results WHERE run_id = %s", (run_id,))
            cursor.execute("DELETE FROM control_runs WHERE run_id = %s", (run_id,))
            cursor.execute("DELETE FROM control_definitions WHERE control_id = %s", (control_id,))
            postgres_conn.commit()
            cursor.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "integration"])
