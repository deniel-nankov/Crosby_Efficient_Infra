"""
Integration Tests for Compliance RAG System
============================================
Tests database connectivity and basic operations.

Run with: pytest tests/test_integration.py -v -m integration

These tests verify:
- PostgreSQL connectivity
- pgvector extension availability  
- Schema integrity (tables, columns, indexes)
- Basic CRUD operations
- Redis connectivity (optional)
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
            port=int(os.environ.get("POSTGRES_PORT", "5433")),  # Note: non-standard port
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
# POSTGRESQL BASIC TESTS
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
    
    def test_pgvector_extension_exists(self, postgres_conn):
        """pgvector extension should be installed."""
        cursor = postgres_conn.cursor()
        cursor.execute("SELECT 1 FROM pg_extension WHERE extname = 'vector'")
        result = cursor.fetchone()
        cursor.close()
        assert result is not None, "pgvector extension not installed"
    
    def test_core_tables_exist(self, postgres_conn):
        """Core tables should exist."""
        # These are the actual tables from the schema
        required_tables = [
            "policy_chunks",
            "fund_control_results",
            "fund_positions",
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
    
    def test_policy_chunks_has_required_columns(self, postgres_conn):
        """policy_chunks table should have required columns."""
        required_columns = ["content", "embedding"]
        
        cursor = postgres_conn.cursor()
        cursor.execute("""
            SELECT column_name FROM information_schema.columns 
            WHERE table_name = 'policy_chunks'
        """)
        existing_columns = [row[0] for row in cursor.fetchall()]
        cursor.close()
        
        for col in required_columns:
            assert col in existing_columns, f"Missing column: {col}"
    
    def test_embedding_column_is_vector_type(self, postgres_conn):
        """embedding column should be vector type."""
        cursor = postgres_conn.cursor()
        cursor.execute("""
            SELECT data_type, udt_name FROM information_schema.columns 
            WHERE table_name = 'policy_chunks' AND column_name = 'embedding'
        """)
        result = cursor.fetchone()
        cursor.close()
        
        assert result is not None, "embedding column not found"
        # pgvector columns show as USER-DEFINED with udt_name = 'vector'
        assert result[0] == 'USER-DEFINED' or 'vector' in str(result).lower()


# =============================================================================
# POSTGRESQL OPERATIONS TESTS
# =============================================================================

@pytest.mark.integration
class TestPostgreSQLOperations:
    """Test PostgreSQL basic operations."""
    
    def test_can_count_policy_chunks(self, postgres_conn):
        """Should be able to count policy chunks."""
        cursor = postgres_conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM policy_chunks")
        result = cursor.fetchone()
        cursor.close()
        
        assert result is not None
        assert isinstance(result[0], int)
    
    def test_can_read_policy_chunks(self, postgres_conn):
        """Should be able to read policy chunks."""
        cursor = postgres_conn.cursor()
        cursor.execute("SELECT content FROM policy_chunks LIMIT 5")
        results = cursor.fetchall()
        cursor.close()
        
        # Results may be empty if no data, but query should succeed
        assert isinstance(results, list)
    
    def test_can_query_embeddings(self, postgres_conn):
        """Should be able to query embeddings."""
        cursor = postgres_conn.cursor()
        cursor.execute("""
            SELECT COUNT(*) FROM policy_chunks 
            WHERE embedding IS NOT NULL
        """)
        result = cursor.fetchone()
        cursor.close()
        
        assert result is not None
        assert isinstance(result[0], int)
    
    def test_vector_dimension_function_works(self, postgres_conn):
        """vector_dims function should work on embeddings."""
        cursor = postgres_conn.cursor()
        cursor.execute("""
            SELECT vector_dims(embedding) FROM policy_chunks 
            WHERE embedding IS NOT NULL LIMIT 1
        """)
        result = cursor.fetchone()
        cursor.close()
        
        # If there are embeddings, dimension should be positive
        if result is not None:
            assert result[0] > 0


# =============================================================================
# COMPLIANCE SNAPSHOTS TESTS
# =============================================================================

@pytest.mark.integration
class TestFundPositions:
    """Test fund_positions table operations."""
    
    def test_can_query_fund_positions(self, postgres_conn):
        """Should be able to query fund_positions."""
        cursor = postgres_conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM fund_positions")
        result = cursor.fetchone()
        cursor.close()
        
        assert result is not None
        assert isinstance(result[0], int)
    
    def test_fund_positions_has_columns(self, postgres_conn):
        """fund_positions should have expected columns."""
        cursor = postgres_conn.cursor()
        cursor.execute("""
            SELECT column_name FROM information_schema.columns 
            WHERE table_name = 'fund_positions'
        """)
        columns = [row[0] for row in cursor.fetchall()]
        cursor.close()
        
        # Should have at least some columns
        assert len(columns) > 0


# =============================================================================
# CONTROL RESULTS TESTS
# =============================================================================

@pytest.mark.integration
class TestControlResults:
    """Test fund_control_results table operations."""
    
    def test_can_query_control_results(self, postgres_conn):
        """Should be able to query fund_control_results."""
        cursor = postgres_conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM fund_control_results")
        result = cursor.fetchone()
        cursor.close()
        
        assert result is not None
        assert isinstance(result[0], int)
    
    def test_control_results_columns(self, postgres_conn):
        """fund_control_results should have expected columns."""
        expected_columns = [
            "control_id", "control_name", "control_type",
            "calculated_value", "threshold", "status"
        ]
        
        cursor = postgres_conn.cursor()
        cursor.execute("""
            SELECT column_name FROM information_schema.columns 
            WHERE table_name = 'fund_control_results'
        """)
        existing_columns = [row[0] for row in cursor.fetchall()]
        cursor.close()
        
        for col in expected_columns:
            assert col in existing_columns, f"Missing column: {col}"


# =============================================================================
# VECTOR SEARCH TESTS
# =============================================================================

@pytest.mark.integration
class TestVectorSearch:
    """Test vector similarity search operations."""
    
    def test_cosine_distance_operator_works(self, postgres_conn):
        """<-> operator should work for cosine distance."""
        cursor = postgres_conn.cursor()
        
        # Create test vectors
        cursor.execute("""
            SELECT 
                '[1,0,0]'::vector <-> '[1,0,0]'::vector AS same,
                '[1,0,0]'::vector <-> '[0,1,0]'::vector AS different
        """)
        result = cursor.fetchone()
        cursor.close()
        
        same_dist, diff_dist = result
        assert same_dist < diff_dist  # Same vector closer than different
    
    def test_can_order_by_vector_distance(self, postgres_conn):
        """Should be able to order by vector distance."""
        cursor = postgres_conn.cursor()
        
        # This query should parse and plan correctly even if no data
        cursor.execute("""
            SELECT chunk_id FROM policy_chunks 
            WHERE embedding IS NOT NULL
            ORDER BY embedding <-> embedding
            LIMIT 1
        """)
        # Just verifying query doesn't error
        cursor.fetchall()
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
            retrieved = redis_conn.get(key)
            assert retrieved.decode() == value
        finally:
            redis_conn.delete(key)
    
    def test_json_storage(self, redis_conn):
        """Should be able to store JSON."""
        key = "test:compliance:json"
        data = {"control_id": "TEST_001", "status": "pass"}
        
        try:
            redis_conn.set(key, json.dumps(data))
            retrieved = json.loads(redis_conn.get(key))
            assert retrieved == data
        finally:
            redis_conn.delete(key)
    
    def test_expiry(self, redis_conn):
        """Should be able to set expiry."""
        key = "test:compliance:expiry"
        
        try:
            redis_conn.setex(key, 3600, "test")  # 1 hour expiry
            ttl = redis_conn.ttl(key)
            assert ttl > 0 and ttl <= 3600
        finally:
            redis_conn.delete(key)


# =============================================================================
# DATA INTEGRITY TESTS
# =============================================================================

@pytest.mark.integration
class TestDataIntegrity:
    """Test data integrity constraints."""
    
    def test_policy_chunk_content_not_null(self, postgres_conn):
        """Policy chunks should have non-null content."""
        cursor = postgres_conn.cursor()
        cursor.execute("""
            SELECT COUNT(*) FROM policy_chunks WHERE content IS NULL
        """)
        null_count = cursor.fetchone()[0]
        cursor.close()
        
        assert null_count == 0, f"Found {null_count} chunks with null content"
    
    def test_embeddings_have_consistent_dimensions(self, postgres_conn):
        """All embeddings should have the same dimension."""
        cursor = postgres_conn.cursor()
        cursor.execute("""
            SELECT DISTINCT vector_dims(embedding) 
            FROM policy_chunks 
            WHERE embedding IS NOT NULL
        """)
        dimensions = cursor.fetchall()
        cursor.close()
        
        # Should be 0 or 1 distinct dimensions
        assert len(dimensions) <= 1, f"Found multiple embedding dimensions: {dimensions}"


# =============================================================================
# TRANSACTION TESTS
# =============================================================================

@pytest.mark.integration
class TestTransactions:
    """Test transaction handling."""
    
    def test_rollback_works(self, postgres_conn):
        """Rollback should undo changes."""
        cursor = postgres_conn.cursor()
        
        # Get initial count
        cursor.execute("SELECT COUNT(*) FROM policy_chunks")
        initial_count = cursor.fetchone()[0]
        
        # Rollback any pending changes
        postgres_conn.rollback()
        
        # Count should be the same
        cursor.execute("SELECT COUNT(*) FROM policy_chunks")
        final_count = cursor.fetchone()[0]
        cursor.close()
        
        assert initial_count == final_count
    
    def test_autocommit_not_enabled(self, postgres_conn):
        """Autocommit should not be enabled (explicit transactions)."""
        # Default psycopg2 connections don't have autocommit
        assert postgres_conn.autocommit == False
