"""
Daily Compliance DAG

Orchestrates the daily compliance workflow:
1. Sync data from Snowflake (or client system)
2. Run compliance controls
3. Generate RAG-powered narratives
4. Store results and create workpaper

Schedule: Daily at 6:00 AM ET (after market data is available)
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.empty import EmptyOperator
from airflow.utils.dates import days_ago

# =============================================================================
# DAG CONFIGURATION
# =============================================================================

default_args = {
    'owner': 'compliance',
    'depends_on_past': False,
    'email': ['compliance@fund.com'],
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'daily_compliance_run',
    default_args=default_args,
    description='Daily compliance control execution and narrative generation',
    schedule_interval='0 6 * * *',  # 6 AM daily
    start_date=days_ago(1),
    catchup=False,
    tags=['compliance', 'daily', 'rag'],
)


# =============================================================================
# TASK FUNCTIONS
# =============================================================================

def sync_from_snowflake(**context):
    """
    Task 1: Sync position and control data from Snowflake.
    
    In production, this would:
    - Connect to Snowflake using snowflake-connector-python
    - Query positions table
    - Query pre-calculated control results
    - Store in local PostgreSQL for processing
    """
    import os
    from datetime import date
    
    execution_date = context['ds']
    print(f"Syncing data for {execution_date}")
    
    # Check if Snowflake is configured
    snowflake_account = os.environ.get('SNOWFLAKE_ACCOUNT')
    
    if snowflake_account:
        # Production: Sync from Snowflake
        print(f"Connecting to Snowflake account: {snowflake_account}")
        # import snowflake.connector
        # conn = snowflake.connector.connect(...)
        # ... sync logic
    else:
        # Development: Use mock adapter
        print("Snowflake not configured, using mock data")
        
        import sys
        sys.path.insert(0, '/opt/airflow/src')
        
        from integration.client_adapter import MockAdapter
        
        adapter = MockAdapter()
        snapshot = adapter.get_snapshot(as_of_date=date.fromisoformat(execution_date))
        
        print(f"Mock snapshot: {len(snapshot.positions)} positions, {len(snapshot.control_results)} controls")
        
        # Store snapshot_id in XCom for downstream tasks
        context['ti'].xcom_push(key='snapshot_id', value=snapshot.snapshot_id)
        context['ti'].xcom_push(key='position_count', value=len(snapshot.positions))
        context['ti'].xcom_push(key='control_count', value=len(snapshot.control_results))
    
    return "sync_complete"


def run_compliance_controls(**context):
    """
    Task 2: Execute compliance controls.
    
    Note: For the refactored architecture, we TRUST that the client's
    system has already run the controls. This task just processes
    the results that were synced in Task 1.
    """
    execution_date = context['ds']
    snapshot_id = context['ti'].xcom_pull(key='snapshot_id', task_ids='sync_from_snowflake')
    
    print(f"Processing controls for snapshot: {snapshot_id}")
    
    # In production: read from PostgreSQL where we synced the data
    # For now: use mock data
    
    import sys
    sys.path.insert(0, '/opt/airflow/src')
    
    from datetime import date
    from integration.client_adapter import MockAdapter
    
    adapter = MockAdapter()
    snapshot = adapter.get_snapshot(as_of_date=date.fromisoformat(execution_date))
    
    # Count statuses
    passed = sum(1 for c in snapshot.control_results if c.status == 'pass')
    warnings = sum(1 for c in snapshot.control_results if c.status == 'warning')
    failed = sum(1 for c in snapshot.control_results if c.status == 'fail')
    
    print(f"Control results: {passed} passed, {warnings} warnings, {failed} failed")
    
    context['ti'].xcom_push(key='passed_count', value=passed)
    context['ti'].xcom_push(key='warning_count', value=warnings)
    context['ti'].xcom_push(key='failed_count', value=failed)
    
    return "controls_complete"


def generate_narratives(**context):
    """
    Task 3: Generate RAG-powered narratives for warnings/failures.
    
    Uses:
    - Policy retrieval (pgvector similarity search)
    - Local LLM (Ollama) for narrative generation
    """
    import os
    import sys
    sys.path.insert(0, '/opt/airflow/src')
    
    from datetime import date
    from integration.client_adapter import MockAdapter
    from integration.rag_pipeline import ComplianceRAGPipeline
    from integration.llm_config import LLMConfig, LLMProvider, ComplianceLLM
    
    execution_date = context['ds']
    
    # Configure LLM
    ollama_host = os.environ.get('OLLAMA_HOST', 'ollama')
    
    config = LLMConfig(
        provider=LLMProvider.OLLAMA,
        model_id=os.environ.get('LLM_MODEL', 'llama3.1:8b'),
        api_base=f"http://{ollama_host}:11434",
        anonymize_data=False,  # Local LLM, no need
    )
    
    print(f"Using LLM: {config.provider.value} / {config.model_id}")
    
    # Get data
    adapter = MockAdapter()
    snapshot = adapter.get_snapshot(as_of_date=date.fromisoformat(execution_date))
    
    # Generate report
    pipeline = ComplianceRAGPipeline()
    report = pipeline.generate_report(snapshot)
    
    print(f"Generated {len(report.narratives)} narratives")
    
    # Store narrative count
    context['ti'].xcom_push(key='narrative_count', value=len(report.narratives))
    
    # In production: store narratives to PostgreSQL
    # for narrative in report.narratives:
    #     insert_narrative(narrative)
    
    return "narratives_complete"


def create_workpaper(**context):
    """
    Task 4: Generate the compliance workpaper document.
    """
    import sys
    sys.path.insert(0, '/opt/airflow/src')
    
    from datetime import date
    from pathlib import Path
    from integration.client_adapter import MockAdapter
    from integration.rag_pipeline import ComplianceRAGPipeline
    
    execution_date = context['ds']
    
    # Get data and generate report
    adapter = MockAdapter()
    snapshot = adapter.get_snapshot(as_of_date=date.fromisoformat(execution_date))
    
    pipeline = ComplianceRAGPipeline()
    report = pipeline.generate_report(snapshot)
    
    # Create workpaper
    workpaper_content = f"""# Daily Compliance Workpaper
    
**Date:** {execution_date}
**Generated:** {report.generated_at}

## Executive Summary

{report.get_executive_summary()}

## Control Results

| Control | Status | Value | Threshold |
|---------|--------|-------|-----------|
"""
    
    for control in snapshot.control_results:
        workpaper_content += f"| {control.control_name} | {control.status.upper()} | {control.calculated_value}% | {control.threshold}% |\n"
    
    workpaper_content += "\n## Narratives\n\n"
    
    for narrative in report.narratives:
        workpaper_content += f"### {narrative.control_id}\n\n"
        workpaper_content += narrative.content + "\n\n"
    
    # Save workpaper
    output_dir = Path('/opt/airflow/output')
    output_dir.mkdir(exist_ok=True)
    
    workpaper_path = output_dir / f"workpaper_{execution_date}.md"
    workpaper_path.write_text(workpaper_content)
    
    print(f"Workpaper saved to: {workpaper_path}")
    
    context['ti'].xcom_push(key='workpaper_path', value=str(workpaper_path))
    
    return "workpaper_complete"


def send_notifications(**context):
    """
    Task 5: Send notifications if there are breaches.
    """
    failed_count = context['ti'].xcom_pull(key='failed_count', task_ids='run_compliance_controls')
    warning_count = context['ti'].xcom_pull(key='warning_count', task_ids='run_compliance_controls')
    
    if failed_count > 0:
        print(f"⚠️ ALERT: {failed_count} control breaches detected!")
        # In production: send email, Slack, PagerDuty, etc.
    elif warning_count > 0:
        print(f"⚡ WARNING: {warning_count} controls in warning state")
    else:
        print("✅ All controls passed")
    
    return "notifications_complete"


# =============================================================================
# DAG STRUCTURE
# =============================================================================

start = EmptyOperator(task_id='start', dag=dag)
end = EmptyOperator(task_id='end', dag=dag)

sync_task = PythonOperator(
    task_id='sync_from_snowflake',
    python_callable=sync_from_snowflake,
    dag=dag,
)

controls_task = PythonOperator(
    task_id='run_compliance_controls',
    python_callable=run_compliance_controls,
    dag=dag,
)

narratives_task = PythonOperator(
    task_id='generate_narratives',
    python_callable=generate_narratives,
    dag=dag,
)

workpaper_task = PythonOperator(
    task_id='create_workpaper',
    python_callable=create_workpaper,
    dag=dag,
)

notify_task = PythonOperator(
    task_id='send_notifications',
    python_callable=send_notifications,
    dag=dag,
)

# Define dependencies
start >> sync_task >> controls_task >> narratives_task >> workpaper_task >> notify_task >> end
