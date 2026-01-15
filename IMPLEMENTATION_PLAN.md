# 🎯 CEI Implementation Plan: Step-by-Step Build & Test Approach

## Philosophy: Build → Test → Validate → Integrate → Repeat

**Core Principles:**
1. **No feature ships without tests**
2. **Each component is bulletproof before integration**
3. **Always know what's working and what's broken**
4. **Start with the smallest valuable piece**
5. **Measure everything - track your progress**

---

## 📊 Progress Tracking System

Before we build anything, let's set up a way to track where we are:

```
STATUS LEGEND:
⬜ Not Started
🟦 In Progress
✅ Completed & Tested
🔄 Needs Revision
🚫 Blocked
```

### **Current Phase Dashboard**
```yaml
project_status:
  current_phase: "Phase 0 - Setup"
  overall_progress: 0%
  
  phases:
    - phase_0_setup: 0%
    - phase_1_foundation: 0%
    - phase_2_validation: 0%
    - phase_3_monitoring: 0%
    - phase_4_intelligence: 0%
    - phase_5_scale: 0%
```

We'll update this after each milestone.

---

## 🏗️ PHASE 0: Setup & Environment (Week 1)

**Goal**: Get development environment working perfectly before writing any code

### Step 0.1: Local Development Setup ⬜
```bash
# Create project structure
mkdir -p cei/{src,tests,config,docs,scripts}
mkdir -p cei/src/{validators,monitors,integrations,reports}
mkdir -p cei/tests/{unit,integration,e2e}

# Initialize Python environment
cd cei
python3 -m venv venv
source venv/bin/activate
```

**Test Criteria:**
- [ ] Virtual environment activates without errors
- [ ] Python version is 3.11+
- [ ] Directory structure created correctly

### Step 0.2: Install Core Dependencies ⬜
```bash
# requirements.txt
cat > requirements.txt << EOF
# Core
pandas==2.1.4
numpy==1.26.2
python-dotenv==1.0.0

# Testing
pytest==7.4.3
pytest-cov==4.1.0
pytest-mock==3.12.0
black==23.12.1
flake8==7.0.0
mypy==1.8.0

# Data Quality
great-expectations==0.18.8

# Development
ipython==8.19.0
pre-commit==3.6.0
EOF

pip install -r requirements.txt
```

**Test Criteria:**
- [ ] All packages install without errors
- [ ] `pytest --version` works
- [ ] `python -c "import great_expectations"` works

### Step 0.3: Setup Testing Infrastructure ⬜
```python
# tests/conftest.py
import pytest
import pandas as pd
from pathlib import Path

@pytest.fixture
def sample_data():
    """Fixture for test data"""
    return pd.DataFrame({
        'timestamp': pd.date_range('2026-01-01', periods=100, freq='1H'),
        'symbol': ['AAPL'] * 100,
        'price': [150.0 + i * 0.5 for i in range(100)],
        'volume': [1000000 + i * 1000 for i in range(100)]
    })

@pytest.fixture
def temp_config_dir(tmp_path):
    """Fixture for temporary config directory"""
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    return config_dir

# Run test to verify fixtures work
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
```

**Test Criteria:**
- [ ] `pytest tests/conftest.py -v` passes
- [ ] Fixtures load correctly
- [ ] Test data generates as expected

### Step 0.4: Setup Git & Pre-commit Hooks ⬜
```bash
# Initialize git (already done, but verify)
git init
git branch -M main

# Setup pre-commit hooks
cat > .pre-commit-config.yaml << EOF
repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.5.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-json
      - id: check-added-large-files

  - repo: https://github.com/psf/black
    rev: 23.12.1
    hooks:
      - id: black

  - repo: https://github.com/pycqa/flake8
    rev: 7.0.0
    hooks:
      - id: flake8
        args: [--max-line-length=100]
EOF

pre-commit install
```

**Test Criteria:**
- [ ] Pre-commit hooks install successfully
- [ ] Create dummy file, run `pre-commit run --all-files` - passes
- [ ] Black auto-formats code correctly

### Step 0.5: Create Project Configuration ⬜
```python
# config/config.yaml
project:
  name: "Crosby Efficient Infrastructure"
  version: "0.1.0"
  environment: "development"

logging:
  level: "DEBUG"
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  file: "logs/cei.log"

data_quality:
  fail_on_error: true
  null_tolerance: 0.0
  enable_caching: true

testing:
  generate_reports: true
  coverage_threshold: 80
```

```python
# src/config.py
import yaml
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

@dataclass
class Config:
    """Application configuration"""
    project_name: str
    version: str
    environment: str
    log_level: str
    
    @classmethod
    def from_yaml(cls, config_path: str) -> 'Config':
        """Load config from YAML file"""
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        return cls(
            project_name=config_dict['project']['name'],
            version=config_dict['project']['version'],
            environment=config_dict['project']['environment'],
            log_level=config_dict['logging']['level']
        )

# Test configuration loading
if __name__ == "__main__":
    config = Config.from_yaml('config/config.yaml')
    print(f"✅ Config loaded: {config.project_name} v{config.version}")
```

**Test Criteria:**
- [ ] Config file loads without errors
- [ ] Can access all config values
- [ ] Test with invalid config - raises appropriate error

### **Phase 0 Checkpoint** ✋
Before proceeding to Phase 1, verify:
- [x] ✅ Development environment is fully functional
- [x] ✅ All tests run with `pytest`
- [x] ✅ Pre-commit hooks work
- [x] ✅ Configuration system works
- [x] ✅ You can commit and push to GitHub

**Estimated Time: 1-2 days**

---

## 🧱 PHASE 1: Build First Data Validator (Week 2-3)

**Goal**: Create ONE bulletproof data quality validator that we can trust completely

### Step 1.1: Define Validation Schema ⬜
```python
# src/validators/schema.py
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from enum import Enum

class ColumnType(Enum):
    """Supported column types"""
    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    DATETIME = "datetime"
    BOOLEAN = "boolean"

@dataclass
class ColumnSchema:
    """Schema definition for a single column"""
    name: str
    dtype: ColumnType
    nullable: bool = False
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    allowed_values: Optional[List[Any]] = None
    
    def validate_value(self, value: Any) -> bool:
        """Validate a single value against schema"""
        # Null check
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return self.nullable
        
        # Range check
        if self.min_value is not None and value < self.min_value:
            return False
        if self.max_value is not None and value > self.max_value:
            return False
        
        # Allowed values check
        if self.allowed_values is not None and value not in self.allowed_values:
            return False
        
        return True

@dataclass
class DataSchema:
    """Complete schema for a dataset"""
    name: str
    columns: List[ColumnSchema]
    row_count_min: Optional[int] = None
    row_count_max: Optional[int] = None
    
    def get_column(self, name: str) -> Optional[ColumnSchema]:
        """Get column schema by name"""
        for col in self.columns:
            if col.name == name:
                return col
        return None
```

**Test Step 1.1:**
```python
# tests/unit/test_schema.py
import pytest
from src.validators.schema import ColumnSchema, ColumnType, DataSchema

def test_column_schema_nullable():
    """Test nullable column validation"""
    col = ColumnSchema(name="price", dtype=ColumnType.FLOAT, nullable=False)
    assert col.validate_value(100.0) == True
    assert col.validate_value(None) == False

def test_column_schema_range():
    """Test range validation"""
    col = ColumnSchema(
        name="price",
        dtype=ColumnType.FLOAT,
        min_value=0.0,
        max_value=1000.0
    )
    assert col.validate_value(500.0) == True
    assert col.validate_value(-1.0) == False
    assert col.validate_value(1001.0) == False

def test_column_schema_allowed_values():
    """Test allowed values validation"""
    col = ColumnSchema(
        name="status",
        dtype=ColumnType.STRING,
        allowed_values=["active", "inactive", "pending"]
    )
    assert col.validate_value("active") == True
    assert col.validate_value("deleted") == False

def test_data_schema_get_column():
    """Test retrieving column from schema"""
    schema = DataSchema(
        name="market_data",
        columns=[
            ColumnSchema(name="symbol", dtype=ColumnType.STRING),
            ColumnSchema(name="price", dtype=ColumnType.FLOAT)
        ]
    )
    assert schema.get_column("symbol") is not None
    assert schema.get_column("invalid") is None

# Run tests
pytest.main([__file__, "-v", "--cov=src.validators.schema"])
```

**Validation Checkpoint:**
- [ ] All schema tests pass (100% coverage)
- [ ] Run `pytest tests/unit/test_schema.py -v`
- [ ] Coverage report shows 100% for schema.py

### Step 1.2: Build Core Validator ⬜
```python
# src/validators/validator.py
import pandas as pd
from typing import List, Dict, Any
from dataclasses import dataclass
from datetime import datetime
from .schema import DataSchema, ColumnSchema

@dataclass
class ValidationResult:
    """Result of a validation check"""
    passed: bool
    schema_name: str
    timestamp: datetime
    errors: List[str]
    warnings: List[str]
    row_count: int
    failed_rows: List[int]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'passed': self.passed,
            'schema_name': self.schema_name,
            'timestamp': self.timestamp.isoformat(),
            'errors': self.errors,
            'warnings': self.warnings,
            'row_count': self.row_count,
            'failed_row_count': len(self.failed_rows)
        }
    
    def __str__(self) -> str:
        """Human-readable string representation"""
        status = "✅ PASSED" if self.passed else "❌ FAILED"
        return f"{status} - {self.schema_name} ({self.row_count} rows)\n" + \
               f"Errors: {len(self.errors)}, Warnings: {len(self.warnings)}"

class DataValidator:
    """Validates data against schema definitions"""
    
    def __init__(self, schema: DataSchema):
        self.schema = schema
    
    def validate(self, df: pd.DataFrame) -> ValidationResult:
        """
        Validate a DataFrame against the schema
        
        Args:
            df: DataFrame to validate
            
        Returns:
            ValidationResult with pass/fail and details
        """
        errors = []
        warnings = []
        failed_rows = []
        
        # Check 1: Column presence
        missing_cols = set(col.name for col in self.schema.columns) - set(df.columns)
        if missing_cols:
            errors.append(f"Missing required columns: {missing_cols}")
        
        extra_cols = set(df.columns) - set(col.name for col in self.schema.columns)
        if extra_cols:
            warnings.append(f"Extra columns found: {extra_cols}")
        
        # Check 2: Row count
        if self.schema.row_count_min and len(df) < self.schema.row_count_min:
            errors.append(f"Row count {len(df)} below minimum {self.schema.row_count_min}")
        if self.schema.row_count_max and len(df) > self.schema.row_count_max:
            errors.append(f"Row count {len(df)} exceeds maximum {self.schema.row_count_max}")
        
        # Check 3: Column-level validation
        for col_schema in self.schema.columns:
            if col_schema.name not in df.columns:
                continue  # Already flagged as missing
            
            column_data = df[col_schema.name]
            
            # Null check
            null_count = column_data.isna().sum()
            if not col_schema.nullable and null_count > 0:
                errors.append(f"Column '{col_schema.name}' has {null_count} null values (not allowed)")
                failed_rows.extend(df[column_data.isna()].index.tolist())
            
            # Value-level validation
            for idx, value in column_data.items():
                if not col_schema.validate_value(value):
                    errors.append(f"Column '{col_schema.name}' row {idx}: invalid value '{value}'")
                    failed_rows.append(idx)
        
        # Remove duplicate failed row indices
        failed_rows = list(set(failed_rows))
        
        return ValidationResult(
            passed=len(errors) == 0,
            schema_name=self.schema.name,
            timestamp=datetime.now(),
            errors=errors,
            warnings=warnings,
            row_count=len(df),
            failed_rows=failed_rows
        )
```

**Test Step 1.2:**
```python
# tests/unit/test_validator.py
import pytest
import pandas as pd
from datetime import datetime
from src.validators.validator import DataValidator, ValidationResult
from src.validators.schema import DataSchema, ColumnSchema, ColumnType

@pytest.fixture
def market_data_schema():
    """Sample schema for market data"""
    return DataSchema(
        name="market_data",
        columns=[
            ColumnSchema(name="symbol", dtype=ColumnType.STRING, nullable=False),
            ColumnSchema(name="price", dtype=ColumnType.FLOAT, nullable=False, min_value=0.0),
            ColumnSchema(name="volume", dtype=ColumnType.INTEGER, nullable=False, min_value=0)
        ],
        row_count_min=1
    )

def test_validator_valid_data(market_data_schema):
    """Test validator with valid data"""
    df = pd.DataFrame({
        'symbol': ['AAPL', 'GOOGL', 'MSFT'],
        'price': [150.0, 2800.0, 350.0],
        'volume': [1000000, 500000, 750000]
    })
    
    validator = DataValidator(market_data_schema)
    result = validator.validate(df)
    
    assert result.passed == True
    assert len(result.errors) == 0
    assert result.row_count == 3

def test_validator_missing_column(market_data_schema):
    """Test validator with missing required column"""
    df = pd.DataFrame({
        'symbol': ['AAPL', 'GOOGL'],
        'price': [150.0, 2800.0]
        # Missing 'volume' column
    })
    
    validator = DataValidator(market_data_schema)
    result = validator.validate(df)
    
    assert result.passed == False
    assert any('Missing required columns' in err for err in result.errors)

def test_validator_null_values(market_data_schema):
    """Test validator with null values in non-nullable column"""
    df = pd.DataFrame({
        'symbol': ['AAPL', None, 'MSFT'],
        'price': [150.0, 2800.0, 350.0],
        'volume': [1000000, 500000, 750000]
    })
    
    validator = DataValidator(market_data_schema)
    result = validator.validate(df)
    
    assert result.passed == False
    assert any('null values' in err for err in result.errors)
    assert 1 in result.failed_rows

def test_validator_invalid_range(market_data_schema):
    """Test validator with values outside allowed range"""
    df = pd.DataFrame({
        'symbol': ['AAPL', 'GOOGL', 'MSFT'],
        'price': [150.0, -100.0, 350.0],  # Negative price invalid
        'volume': [1000000, 500000, 750000]
    })
    
    validator = DataValidator(market_data_schema)
    result = validator.validate(df)
    
    assert result.passed == False
    assert any('invalid value' in err for err in result.errors)

def test_validation_result_to_dict():
    """Test ValidationResult serialization"""
    result = ValidationResult(
        passed=False,
        schema_name="test_schema",
        timestamp=datetime.now(),
        errors=["error1", "error2"],
        warnings=["warning1"],
        row_count=100,
        failed_rows=[1, 2, 3]
    )
    
    result_dict = result.to_dict()
    assert result_dict['passed'] == False
    assert len(result_dict['errors']) == 2
    assert result_dict['failed_row_count'] == 3

# Run all tests
pytest.main([__file__, "-v", "--cov=src.validators.validator", "--cov-report=html"])
```

**Validation Checkpoint:**
- [ ] All validator tests pass (100% coverage)
- [ ] Run `pytest tests/unit/test_validator.py -v`
- [ ] Coverage report shows >95% for validator.py
- [ ] Manual test: Create sample CSV, validate it

### Step 1.3: Create Command-Line Interface ⬜
```python
# src/cli.py
import click
import pandas as pd
import yaml
from pathlib import Path
from src.validators.validator import DataValidator
from src.validators.schema import DataSchema, ColumnSchema, ColumnType

@click.group()
def cli():
    """CEI Data Quality Validator"""
    pass

@cli.command()
@click.argument('data_file', type=click.Path(exists=True))
@click.argument('schema_file', type=click.Path(exists=True))
@click.option('--output', '-o', help='Output file for validation report')
def validate(data_file: str, schema_file: str, output: str):
    """Validate a data file against a schema"""
    
    # Load data
    click.echo(f"📂 Loading data from {data_file}...")
    if data_file.endswith('.csv'):
        df = pd.read_csv(data_file)
    elif data_file.endswith('.parquet'):
        df = pd.read_parquet(data_file)
    else:
        click.echo("❌ Unsupported file format. Use CSV or Parquet.", err=True)
        return
    
    click.echo(f"✅ Loaded {len(df)} rows, {len(df.columns)} columns")
    
    # Load schema
    click.echo(f"📋 Loading schema from {schema_file}...")
    with open(schema_file, 'r') as f:
        schema_dict = yaml.safe_load(f)
    
    # Create schema object (simplified - you'd parse the YAML properly)
    schema = _parse_schema(schema_dict)
    
    # Validate
    click.echo(f"🔍 Validating data against schema '{schema.name}'...")
    validator = DataValidator(schema)
    result = validator.validate(df)
    
    # Print results
    click.echo("\n" + "="*60)
    click.echo(str(result))
    click.echo("="*60)
    
    if result.errors:
        click.echo("\n❌ ERRORS:")
        for error in result.errors:
            click.echo(f"  - {error}")
    
    if result.warnings:
        click.echo("\n⚠️  WARNINGS:")
        for warning in result.warnings:
            click.echo(f"  - {warning}")
    
    # Save report if requested
    if output:
        import json
        with open(output, 'w') as f:
            json.dump(result.to_dict(), f, indent=2)
        click.echo(f"\n💾 Report saved to {output}")
    
    # Exit with appropriate code
    exit(0 if result.passed else 1)

def _parse_schema(schema_dict: dict) -> DataSchema:
    """Parse schema from dictionary"""
    columns = []
    for col_def in schema_dict['columns']:
        columns.append(ColumnSchema(
            name=col_def['name'],
            dtype=ColumnType(col_def['type']),
            nullable=col_def.get('nullable', False),
            min_value=col_def.get('min_value'),
            max_value=col_def.get('max_value')
        ))
    
    return DataSchema(
        name=schema_dict['name'],
        columns=columns,
        row_count_min=schema_dict.get('row_count_min'),
        row_count_max=schema_dict.get('row_count_max')
    )

if __name__ == '__main__':
    cli()
```

**Test Step 1.3:**
```bash
# Create test data
cat > test_data.csv << EOF
symbol,price,volume
AAPL,150.5,1000000
GOOGL,2800.0,500000
MSFT,350.0,750000
EOF

# Create test schema
cat > test_schema.yaml << EOF
name: "market_data"
columns:
  - name: "symbol"
    type: "string"
    nullable: false
  - name: "price"
    type: "float"
    nullable: false
    min_value: 0.0
  - name: "volume"
    type: "integer"
    nullable: false
    min_value: 0
row_count_min: 1
EOF

# Test the CLI
python -m src.cli validate test_data.csv test_schema.yaml
python -m src.cli validate test_data.csv test_schema.yaml --output report.json

# Verify output
cat report.json
```

**Validation Checkpoint:**
- [ ] CLI runs without errors
- [ ] Valid data passes validation
- [ ] Invalid data fails validation
- [ ] JSON report generates correctly
- [ ] Exit codes are correct (0 for pass, 1 for fail)

### **Phase 1 Checkpoint** ✋
Before proceeding to Phase 2, verify:
- [x] ✅ Schema system works perfectly
- [x] ✅ Validator catches all test cases
- [x] ✅ 100% test coverage on core logic
- [x] ✅ CLI tool is functional
- [x] ✅ Can validate real CSV/Parquet files
- [x] ✅ Documentation written for validator

**Commit to git**: `git commit -m "Phase 1 complete: Core data validator with 100% test coverage"`

**Estimated Time: 1-2 weeks**

---

## 🔬 PHASE 2: Great Expectations Integration (Week 4)

**Goal**: Replace our custom validator with industry-standard Great Expectations (but keep our validator for learning)

### Step 2.1: Setup Great Expectations ⬜
```bash
# Initialize Great Expectations
great_expectations init

# This creates:
# - great_expectations/
#   - checkpoints/
#   - expectations/
#   - plugins/
#   - uncommitted/
#   - great_expectations.yml
```

**Test Step 2.1:**
```bash
# Verify GE is working
great_expectations --version
great_expectations suite list
```

### Step 2.2: Create First Expectation Suite ⬜
```python
# scripts/create_expectation_suite.py
import great_expectations as gx
import pandas as pd

# Create data context
context = gx.get_context()

# Create a sample dataset
df = pd.DataFrame({
    'symbol': ['AAPL', 'GOOGL', 'MSFT'],
    'price': [150.0, 2800.0, 350.0],
    'volume': [1000000, 500000, 750000]
})

# Create a datasource
datasource = context.sources.add_pandas("market_data_source")
data_asset = datasource.add_dataframe_asset(name="market_data")
batch_request = data_asset.build_batch_request(dataframe=df)

# Create expectation suite
suite = context.add_expectation_suite("market_data_expectations")

# Add expectations
validator = context.get_validator(
    batch_request=batch_request,
    expectation_suite_name="market_data_expectations"
)

# Column expectations
validator.expect_table_columns_to_match_ordered_list(
    column_list=['symbol', 'price', 'volume']
)

validator.expect_column_values_to_not_be_null(column='symbol')
validator.expect_column_values_to_not_be_null(column='price')
validator.expect_column_values_to_not_be_null(column='volume')

validator.expect_column_values_to_be_between(
    column='price',
    min_value=0.0,
    max_value=10000.0
)

validator.expect_column_values_to_be_between(
    column='volume',
    min_value=0,
    max_value=1000000000
)

# Save suite
validator.save_expectation_suite(discard_failed_expectations=False)

print("✅ Expectation suite created successfully!")
```

**Test Step 2.2:**
```bash
python scripts/create_expectation_suite.py
great_expectations suite list  # Should show market_data_expectations
```

### Step 2.3: Create Validation Runner ⬜
```python
# src/validators/ge_validator.py
import great_expectations as gx
from great_expectations.checkpoint import Checkpoint
from typing import Dict, Any
import pandas as pd
from datetime import datetime

class GEValidator:
    """Wrapper for Great Expectations validation"""
    
    def __init__(self, context_root_dir: str = None):
        self.context = gx.get_context(context_root_dir=context_root_dir)
    
    def validate_dataframe(
        self,
        df: pd.DataFrame,
        expectation_suite_name: str,
        datasource_name: str = "pandas_datasource"
    ) -> Dict[str, Any]:
        """
        Validate a DataFrame using Great Expectations
        
        Args:
            df: DataFrame to validate
            expectation_suite_name: Name of the expectation suite to use
            datasource_name: Name of the datasource
            
        Returns:
            Dict with validation results
        """
        # Get or create datasource
        try:
            datasource = self.context.get_datasource(datasource_name)
        except:
            datasource = self.context.sources.add_pandas(datasource_name)
        
        # Add dataframe as asset
        data_asset = datasource.add_dataframe_asset(
            name=f"validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        
        batch_request = data_asset.build_batch_request(dataframe=df)
        
        # Run validation
        validator = self.context.get_validator(
            batch_request=batch_request,
            expectation_suite_name=expectation_suite_name
        )
        
        results = validator.validate()
        
        # Format results
        return {
            'success': results.success,
            'statistics': results.statistics,
            'results': [{
                'expectation_type': r.expectation_config.expectation_type,
                'success': r.success,
                'result': r.result
            } for r in results.results]
        }
    
    def validate_csv(
        self,
        file_path: str,
        expectation_suite_name: str
    ) -> Dict[str, Any]:
        """Validate a CSV file"""
        df = pd.read_csv(file_path)
        return self.validate_dataframe(df, expectation_suite_name)
```

**Test Step 2.3:**
```python
# tests/integration/test_ge_validator.py
import pytest
import pandas as pd
from src.validators.ge_validator import GEValidator

@pytest.fixture
def ge_validator():
    """Fixture for GE validator"""
    return GEValidator()

def test_validate_valid_dataframe(ge_validator):
    """Test validation with valid data"""
    df = pd.DataFrame({
        'symbol': ['AAPL', 'GOOGL', 'MSFT'],
        'price': [150.0, 2800.0, 350.0],
        'volume': [1000000, 500000, 750000]
    })
    
    # First create the expectation suite (if not exists)
    # ... (code from Step 2.2)
    
    results = ge_validator.validate_dataframe(
        df,
        expectation_suite_name='market_data_expectations'
    )
    
    assert results['success'] == True
    assert results['statistics']['successful_expectations'] > 0

def test_validate_invalid_dataframe(ge_validator):
    """Test validation with invalid data"""
    df = pd.DataFrame({
        'symbol': ['AAPL', None, 'MSFT'],  # Null value
        'price': [150.0, -100.0, 350.0],   # Negative price
        'volume': [1000000, 500000, 750000]
    })
    
    results = ge_validator.validate_dataframe(
        df,
        expectation_suite_name='market_data_expectations'
    )
    
    assert results['success'] == False
    assert results['statistics']['unsuccessful_expectations'] > 0

# Run tests
pytest.main([__file__, "-v"])
```

**Validation Checkpoint:**
- [ ] GE validator runs without errors
- [ ] Valid data passes all expectations
- [ ] Invalid data fails appropriate expectations
- [ ] Can generate HTML validation reports
- [ ] Integration tests pass

### **Phase 2 Checkpoint** ✋
Before proceeding to Phase 3, verify:
- [x] ✅ Great Expectations fully integrated
- [x] ✅ Can validate DataFrames and CSV files
- [x] ✅ All tests pass
- [x] ✅ Validation reports are readable and actionable
- [x] ✅ Performance is acceptable (benchmark on 1M rows)

**Commit to git**: `git commit -m "Phase 2 complete: Great Expectations integration with tests"`

**Estimated Time: 1 week**

---

## 📊 PHASE 3: Monitoring & Alerting (Week 5-6)

**Goal**: Add real-time monitoring and Slack alerts when validation fails

### Step 3.1: Setup Logging Infrastructure ⬜
```python
# src/logging_config.py
import logging
import sys
from pathlib import Path
from datetime import datetime

def setup_logging(log_level: str = "INFO", log_file: str = None):
    """Configure application logging"""
    
    # Create logs directory
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))
    
    # Console handler (pretty output)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)
    
    # File handler (detailed logs)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)
    
    return root_logger

# Create logger for validation events
validation_logger = logging.getLogger('cei.validation')
```

**Test Step 3.1:**
```python
# tests/unit/test_logging.py
import logging
from src.logging_config import setup_logging

def test_logging_setup():
    """Test logging configuration"""
    logger = setup_logging(log_level="DEBUG", log_file="logs/test.log")
    
    logger.info("Test info message")
    logger.warning("Test warning message")
    logger.error("Test error message")
    
    # Verify log file exists
    from pathlib import Path
    assert Path("logs/test.log").exists()
```

### Step 3.2: Create Slack Notifier ⬜
```python
# src/monitors/slack_notifier.py
import requests
import json
from typing import Dict, Any, Optional
from datetime import datetime

class SlackNotifier:
    """Send alerts to Slack"""
    
    def __init__(self, webhook_url: str):
        self.webhook_url = webhook_url
    
    def send_validation_alert(
        self,
        validation_result: Dict[str, Any],
        severity: str = "error"
    ) -> bool:
        """
        Send validation failure alert to Slack
        
        Args:
            validation_result: Results from validator
            severity: 'info', 'warning', or 'error'
            
        Returns:
            True if sent successfully
        """
        # Determine color based on severity
        colors = {
            'info': '#36a64f',
            'warning': '#ff9900',
            'error': '#ff0000'
        }
        color = colors.get(severity, colors['error'])
        
        # Determine emoji
        emojis = {
            'info': ':white_check_mark:',
            'warning': ':warning:',
            'error': ':x:'
        }
        emoji = emojis.get(severity, emojis['error'])
        
        # Build message
        success = validation_result.get('success', False)
        status = "PASSED" if success else "FAILED"
        
        message = {
            "attachments": [{
                "color": color,
                "title": f"{emoji} Data Validation {status}",
                "fields": [
                    {
                        "title": "Timestamp",
                        "value": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        "short": True
                    },
                    {
                        "title": "Status",
                        "value": status,
                        "short": True
                    }
                ],
                "footer": "CEI Data Quality Monitor"
            }]
        }
        
        # Add statistics if available
        if 'statistics' in validation_result:
            stats = validation_result['statistics']
            message['attachments'][0]['fields'].extend([
                {
                    "title": "Successful Expectations",
                    "value": str(stats.get('successful_expectations', 0)),
                    "short": True
                },
                {
                    "title": "Failed Expectations",
                    "value": str(stats.get('unsuccessful_expectations', 0)),
                    "short": True
                }
            ])
        
        # Add error details if failed
        if not success and 'results' in validation_result:
            failed_expectations = [
                r for r in validation_result['results']
                if not r.get('success', True)
            ]
            
            if failed_expectations:
                error_text = "\\n".join([
                    f"• {e['expectation_type']}"
                    for e in failed_expectations[:5]  # Limit to first 5
                ])
                message['attachments'][0]['fields'].append({
                    "title": "Failed Expectations",
                    "value": error_text,
                    "short": False
                })
        
        # Send to Slack
        try:
            response = requests.post(
                self.webhook_url,
                data=json.dumps(message),
                headers={'Content-Type': 'application/json'},
                timeout=10
            )
            return response.status_code == 200
        except Exception as e:
            print(f"Failed to send Slack notification: {e}")
            return False
    
    def send_simple_message(self, text: str) -> bool:
        """Send a simple text message"""
        message = {"text": text}
        try:
            response = requests.post(
                self.webhook_url,
                data=json.dumps(message),
                headers={'Content-Type': 'application/json'},
                timeout=10
            )
            return response.status_code == 200
        except Exception as e:
            print(f"Failed to send Slack message: {e}")
            return False
```

**Test Step 3.2:**
```python
# tests/unit/test_slack_notifier.py
import pytest
from unittest.mock import Mock, patch
from src.monitors.slack_notifier import SlackNotifier

@pytest.fixture
def mock_webhook_url():
    return "https://hooks.slack.com/services/TEST/WEBHOOK/URL"

def test_send_validation_alert_success(mock_webhook_url):
    """Test sending successful validation alert"""
    notifier = SlackNotifier(mock_webhook_url)
    
    validation_result = {
        'success': True,
        'statistics': {
            'successful_expectations': 10,
            'unsuccessful_expectations': 0
        }
    }
    
    with patch('requests.post') as mock_post:
        mock_post.return_value.status_code = 200
        result = notifier.send_validation_alert(validation_result, severity='info')
        
        assert result == True
        assert mock_post.called

def test_send_validation_alert_failure(mock_webhook_url):
    """Test sending failed validation alert"""
    notifier = SlackNotifier(mock_webhook_url)
    
    validation_result = {
        'success': False,
        'statistics': {
            'successful_expectations': 8,
            'unsuccessful_expectations': 2
        },
        'results': [
            {'expectation_type': 'expect_column_values_to_not_be_null', 'success': False},
            {'expectation_type': 'expect_column_values_to_be_between', 'success': False}
        ]
    }
    
    with patch('requests.post') as mock_post:
        mock_post.return_value.status_code = 200
        result = notifier.send_validation_alert(validation_result, severity='error')
        
        assert result == True
        # Verify error details included in message
        call_args = mock_post.call_args
        message = json.loads(call_args[1]['data'])
        assert any('Failed Expectations' in field.get('title', '') 
                  for field in message['attachments'][0]['fields'])

# Manual test (requires real Slack webhook)
if __name__ == "__main__":
    import os
    webhook_url = os.getenv('SLACK_WEBHOOK_URL')
    if webhook_url:
        notifier = SlackNotifier(webhook_url)
        notifier.send_simple_message("🧪 CEI Slack integration test - ignore this message")
```

**Validation Checkpoint:**
- [ ] Logging works correctly (console + file)
- [ ] Slack notifier sends test messages
- [ ] Unit tests pass with mocked requests
- [ ] Manual Slack test works with real webhook

### Step 3.3: Integrate Monitoring with Validator ⬜
```python
# src/validators/monitored_validator.py
from typing import Dict, Any, Optional
import logging
from .ge_validator import GEValidator
from ..monitors.slack_notifier import SlackNotifier

logger = logging.getLogger(__name__)

class MonitoredValidator:
    """Validator with integrated monitoring and alerting"""
    
    def __init__(
        self,
        ge_validator: GEValidator,
        slack_notifier: Optional[SlackNotifier] = None,
        alert_on_warning: bool = False
    ):
        self.validator = ge_validator
        self.notifier = slack_notifier
        self.alert_on_warning = alert_on_warning
    
    def validate_with_monitoring(
        self,
        df,
        expectation_suite_name: str,
        pipeline_name: str = "unknown"
    ) -> Dict[str, Any]:
        """
        Validate data with automatic monitoring and alerting
        
        Args:
            df: DataFrame to validate
            expectation_suite_name: Expectation suite name
            pipeline_name: Name of pipeline for logging
            
        Returns:
            Validation results
        """
        logger.info(f"Starting validation for pipeline '{pipeline_name}'")
        
        try:
            # Run validation
            results = self.validator.validate_dataframe(df, expectation_suite_name)
            
            # Log results
            if results['success']:
                logger.info(f"✅ Validation PASSED for '{pipeline_name}'")
                logger.debug(f"Statistics: {results['statistics']}")
                
                # Optional info alert
                if self.alert_on_warning and self.notifier:
                    self.notifier.send_validation_alert(results, severity='info')
            else:
                logger.error(f"❌ Validation FAILED for '{pipeline_name}'")
                logger.error(f"Failed expectations: {results['statistics'].get('unsuccessful_expectations', 0)}")
                
                # Send alert
                if self.notifier:
                    self.notifier.send_validation_alert(results, severity='error')
            
            return results
            
        except Exception as e:
            logger.exception(f"💥 Validation ERROR for '{pipeline_name}': {e}")
            
            # Send critical alert
            if self.notifier:
                self.notifier.send_simple_message(
                    f":rotating_light: CEI Validation Error: {pipeline_name}\\n"
                    f"Error: {str(e)}"
                )
            
            raise
```

**Test Step 3.3:**
```python
# tests/integration/test_monitored_validator.py
import pytest
import pandas as pd
from unittest.mock import Mock
from src.validators.monitored_validator import MonitoredValidator
from src.validators.ge_validator import GEValidator
from src.monitors.slack_notifier import SlackNotifier

def test_monitored_validation_success():
    """Test monitored validation with successful data"""
    ge_validator = GEValidator()
    mock_notifier = Mock(spec=SlackNotifier)
    
    monitored = MonitoredValidator(
        ge_validator=ge_validator,
        slack_notifier=mock_notifier,
        alert_on_warning=True
    )
    
    df = pd.DataFrame({
        'symbol': ['AAPL', 'GOOGL'],
        'price': [150.0, 2800.0],
        'volume': [1000000, 500000]
    })
    
    results = monitored.validate_with_monitoring(
        df,
        expectation_suite_name='market_data_expectations',
        pipeline_name='test_pipeline'
    )
    
    assert results['success'] == True
    # Verify Slack was called for info alert
    assert mock_notifier.send_validation_alert.called

def test_monitored_validation_failure():
    """Test monitored validation with failing data"""
    ge_validator = GEValidator()
    mock_notifier = Mock(spec=SlackNotifier)
    
    monitored = MonitoredValidator(
        ge_validator=ge_validator,
        slack_notifier=mock_notifier
    )
    
    df = pd.DataFrame({
        'symbol': ['AAPL', None],  # Invalid: null value
        'price': [150.0, -100.0],  # Invalid: negative price
        'volume': [1000000, 500000]
    })
    
    results = monitored.validate_with_monitoring(
        df,
        expectation_suite_name='market_data_expectations',
        pipeline_name='test_pipeline'
    )
    
    assert results['success'] == False
    # Verify error alert was sent
    assert mock_notifier.send_validation_alert.called
    call_args = mock_notifier.send_validation_alert.call_args
    assert call_args[1]['severity'] == 'error'
```

### **Phase 3 Checkpoint** ✋
Before proceeding to Phase 4, verify:
- [x] ✅ Logging works across all components
- [x] ✅ Slack alerts send successfully
- [x] ✅ Failed validations trigger alerts automatically
- [x] ✅ All tests pass
- [x] ✅ Can handle validation failures gracefully
- [x] ✅ Alert fatigue is manageable (not too noisy)

**Commit to git**: `git commit -m "Phase 3 complete: Monitoring and Slack alerting integrated"`

**Estimated Time: 1-2 weeks**

---

## 🔗 PHASE 4: Airflow Integration (Week 7-8)

**Goal**: Integrate CEI validator into actual Airflow DAGs

### Step 4.1: Create Airflow Integration Module ⬜
```python
# src/integrations/airflow_integration.py
from airflow.operators.python import PythonOperator
from airflow.exceptions import AirflowException
from typing import Callable, Dict, Any
import pandas as pd

from src.validators.monitored_validator import MonitoredValidator
from src.validators.ge_validator import GEValidator
from src.monitors.slack_notifier import SlackNotifier

def create_validation_task(
    task_id: str,
    data_loader: Callable[[], pd.DataFrame],
    expectation_suite_name: str,
    slack_webhook_url: str = None,
    **kwargs
) -> PythonOperator:
    """
    Create an Airflow task for data validation
    
    Args:
        task_id: Unique task ID
        data_loader: Function that returns DataFrame to validate
        expectation_suite_name: GE expectation suite name
        slack_webhook_url: Optional Slack webhook for alerts
        
    Returns:
        PythonOperator task
    """
    
    def validation_callable(**context):
        """Inner function executed by Airflow"""
        # Load data
        df = data_loader()
        
        # Setup validator
        ge_validator = GEValidator()
        notifier = SlackNotifier(slack_webhook_url) if slack_webhook_url else None
        monitored_validator = MonitoredValidator(
            ge_validator=ge_validator,
            slack_notifier=notifier
        )
        
        # Run validation
        results = monitored_validator.validate_with_monitoring(
            df=df,
            expectation_suite_name=expectation_suite_name,
            pipeline_name=context['dag'].dag_id
        )
        
        # Fail task if validation fails
        if not results['success']:
            raise AirflowException(
                f"Data validation failed: {results['statistics']['unsuccessful_expectations']} "
                f"expectations failed"
            )
        
        # Return results for downstream tasks (via XCom)
        return results
    
    return PythonOperator(
        task_id=task_id,
        python_callable=validation_callable,
        provide_context=True,
        **kwargs
    )
```

### Step 4.2: Create Example Airflow DAG ⬜
```python
# dags/market_data_pipeline.py
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import pandas as pd

from src.integrations.airflow_integration import create_validation_task

# Default args for DAG
default_args = {
    'owner': 'cei',
    'depends_on_past': False,
    'start_date': datetime(2026, 1, 1),
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# Create DAG
dag = DAG(
    'market_data_quality_check',
    default_args=default_args,
    description='Market data pipeline with CEI quality validation',
    schedule_interval=timedelta(hours=1),
    catchup=False,
    tags=['cei', 'data-quality', 'market-data']
)

# Task 1: Extract data (simulated)
def extract_market_data(**context):
    """Simulate extracting market data"""
    # In reality, this would fetch from Bloomberg, etc.
    df = pd.DataFrame({
        'timestamp': pd.date_range('2026-01-15', periods=100, freq='1min'),
        'symbol': ['AAPL'] * 100,
        'price': [150.0 + i * 0.1 for i in range(100)],
        'volume': [1000000 + i * 1000 for i in range(100)]
    })
    
    # Save to XCom for next task
    context['task_instance'].xcom_push(key='market_data', value=df.to_json())
    return "Market data extracted successfully"

extract_task = PythonOperator(
    task_id='extract_market_data',
    python_callable=extract_market_data,
    dag=dag
)

# Task 2: Validate data with CEI
def load_data_for_validation(**context):
    """Load data from previous task"""
    json_data = context['task_instance'].xcom_pull(
        task_ids='extract_market_data',
        key='market_data'
    )
    return pd.read_json(json_data)

validate_task = create_validation_task(
    task_id='validate_market_data',
    data_loader=load_data_for_validation,
    expectation_suite_name='market_data_expectations',
    slack_webhook_url='{{ var.value.slack_webhook_url }}',  # Airflow variable
    dag=dag
)

# Task 3: Transform data (only runs if validation passes)
def transform_market_data(**context):
    """Transform validated data"""
    json_data = context['task_instance'].xcom_pull(
        task_ids='extract_market_data',
        key='market_data'
    )
    df = pd.read_json(json_data)
    
    # Example transformation
    df['price_change'] = df['price'].pct_change()
    df['volume_ma'] = df['volume'].rolling(window=10).mean()
    
    print(f"Transformed {len(df)} rows")
    return "Transformation complete"

transform_task = PythonOperator(
    task_id='transform_market_data',
    python_callable=transform_market_data,
    dag=dag
)

# Task 4: Load data to destination
def load_market_data(**context):
    """Load data to database/data lake"""
    # Simulated load
    print("Loading data to database...")
    return "Data loaded successfully"

load_task = PythonOperator(
    task_id='load_market_data',
    python_callable=load_market_data,
    dag=dag
)

# Define task dependencies
extract_task >> validate_task >> transform_task >> load_task
```

**Test Step 4.2:**
```bash
# Test DAG syntax
python dags/market_data_pipeline.py

# Test DAG in Airflow
airflow dags test market_data_quality_check 2026-01-15

# Trigger DAG manually
airflow dags trigger market_data_quality_check
```

**Validation Checkpoint:**
- [ ] DAG loads in Airflow without errors
- [ ] Validation task runs successfully with valid data
- [ ] Validation task fails appropriately with invalid data
- [ ] Downstream tasks only run if validation passes
- [ ] Slack alerts are sent on validation failure

### **Phase 4 Checkpoint** ✋
Before proceeding to Phase 5, verify:
- [x] ✅ Airflow integration is seamless
- [x] ✅ Can add validation to any DAG easily
- [x] ✅ Pipeline stops on validation failure (fail-closed)
- [x] ✅ All tests pass
- [x] ✅ Monitoring works in Airflow context
- [x] ✅ Documentation for Airflow integration

**Commit to git**: `git commit -m "Phase 4 complete: Airflow integration with automatic validation"`

**Estimated Time: 1-2 weeks**

---

## 📈 Progress Tracking

Update this after each phase:

```yaml
✅ Phase 0: Setup & Environment (100%)
✅ Phase 1: Core Data Validator (100%)
✅ Phase 2: Great Expectations Integration (100%)
✅ Phase 3: Monitoring & Alerting (100%)
✅ Phase 4: Airflow Integration (100%)
⬜ Phase 5: Dashboard & Reporting (0%)
⬜ Phase 6: LLM Integration (0%)
⬜ Phase 7: Production Hardening (0%)
⬜ Phase 8: Scale & Optimize (0%)

Overall Progress: 50% (4/8 phases complete)
```

---

## 🎯 Quality Gates

Every phase must pass these gates before moving forward:

### **Code Quality Gate**
- [ ] All unit tests pass (pytest)
- [ ] Code coverage >80%
- [ ] No linting errors (flake8)
- [ ] Type hints present (mypy clean)
- [ ] Code formatted (black)

### **Functionality Gate**
- [ ] Feature works as designed
- [ ] Integration tests pass
- [ ] Manual testing successful
- [ ] Edge cases handled

### **Documentation Gate**
- [ ] Code has docstrings
- [ ] README updated
- [ ] Examples provided
- [ ] Runbook created (if needed)

### **Production Readiness Gate**
- [ ] Logging in place
- [ ] Error handling robust
- [ ] Performance acceptable
- [ ] Security reviewed

---

## 📝 Daily Progress Log Template

```markdown
## Date: YYYY-MM-DD

### Today's Goal:
[What you planned to accomplish]

### What I Built:
- [X] Task 1
- [X] Task 2
- [ ] Task 3 (in progress)

### What I Tested:
- [Test results and outcomes]

### What I Learned:
- [Key insights or challenges]

### Blockers:
- [Any issues blocking progress]

### Tomorrow's Plan:
- [ ] Complete Task 3
- [ ] Start Task 4
```

---

## 🚀 Next Phases (High-Level)

### Phase 5: Dashboard & Reporting (Week 9-10)
- Build Grafana dashboards for real-time monitoring
- Create automated email reports
- Add metrics tracking (success rate, MTTR, etc.)

### Phase 6: LLM Integration (Week 11-12)
- Add LLM explanations for validation failures
- Implement root cause analysis
- Create natural language reports

### Phase 7: Production Hardening (Week 13-14)
- Load testing and performance optimization
- Security audit
- Disaster recovery planning
- Full documentation

### Phase 8: Scale & Optimize (Week 15-16)
- Support 100+ pipelines
- Advanced features (auto-remediation, predictive analytics)
- User training and onboarding

---

## 💡 Key Principles to Remember

1. **Test Before You Build More**: Never add a new feature until the current one is bulletproof
2. **Commit Often**: Every working increment gets committed
3. **Document as You Go**: Future you will thank present you
4. **Measure Everything**: Track time, lines of code, test coverage, performance
5. **Ask for Help**: If stuck >2 hours, reach out
6. **Celebrate Wins**: Each checkpoint is an achievement!

---

## 🎉 You've Got This!

This plan gives you a clear path forward. Focus on one step at a time, test thoroughly, and build something bulletproof. Start with Phase 0 and don't rush - quality over speed!

**Remember**: A small, working, well-tested system is infinitely more valuable than a large, broken one.

Good luck! 🚀
