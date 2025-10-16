"""
Benchmark result storage using pandas DataFrames and SQLite.

Provides simple, fast storage and retrieval of benchmark results with
automatic schema management and easy querying via pandas.
"""

import sqlite3
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import json


class BenchmarkDB:
    """
    Lightweight storage for benchmark results using pandas + SQLite.
    
    Features:
    - Automatic result saving with timestamps
    - Query results as pandas DataFrames
    - Export to CSV, JSON, or Parquet
    - Simple API for filtering and analysis
    
    Example:
        db = BenchmarkDB()
        db.save_result('cache', result, params={'problem_size': 256})
        df = db.query(benchmark='cache', limit=100)
        df.to_csv('cache_results.csv')
    """
    
    def __init__(self, db_path: str = "benchmark_results.db"):
        """
        Initialize database connection.
        
        Args:
            db_path: Path to SQLite database file (default: benchmark_results.db)
        """
        self.db_path = Path(db_path)
        self._create_schema()
    
    def _create_schema(self):
        """Create benchmark results table if it doesn't exist."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS benchmark_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                gpu_name TEXT NOT NULL,
                gpu_arch TEXT,
                benchmark_type TEXT NOT NULL,
                problem_size INTEGER,
                block_size INTEGER,
                iterations INTEGER,
                parameters TEXT,
                primary_metric REAL NOT NULL,
                metric_name TEXT NOT NULL,
                spread_percent REAL,
                execution_time_ms REAL,
                rocm_version TEXT,
                hostname TEXT
            )
        """)
        
        # Create indices for common queries
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_benchmark_timestamp 
            ON benchmark_results(benchmark_type, timestamp)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_gpu_benchmark 
            ON benchmark_results(gpu_name, benchmark_type)
        """)
        
        conn.commit()
        conn.close()
    
    def save_result(self, 
                    benchmark_name: str,
                    result: Any,  # BenchmarkResult object
                    params: Optional[Dict] = None,
                    gpu_info: Optional[Dict] = None) -> int:
        """
        Save a benchmark result to the database.
        
        Args:
            benchmark_name: Name of the benchmark (e.g., 'cache')
            result: BenchmarkResult object with metrics
            params: Additional parameters used for this run
            gpu_info: Optional GPU information dictionary
        
        Returns:
            Row ID of inserted result
        
        Example:
            db.save_result('cache', result, 
                          params={'problem_size': 256, 'block_size': 256},
                          gpu_info={'name': 'MI325X', 'arch': 'gfx942'})
        """
        import socket
        
        # Extract parameters from params dict or use defaults
        params = params or {}
        
        record = {
            'timestamp': datetime.now().isoformat(),
            'gpu_name': gpu_info.get('name') if gpu_info else 'Unknown',
            'gpu_arch': gpu_info.get('arch') if gpu_info else None,
            'benchmark_type': benchmark_name,
            'problem_size': params.get('problem_size'),
            'block_size': params.get('block_size'),
            'iterations': params.get('iterations'),
            'parameters': json.dumps(params),
            'primary_metric': result.primary_metric,
            'metric_name': result.metric_name,
            'spread_percent': result.spread_percent,
            'execution_time_ms': getattr(result, 'execution_time_ms', None),
            'rocm_version': gpu_info.get('rocm_version') if gpu_info else None,
            'hostname': socket.gethostname()
        }
        
        df = pd.DataFrame([record])
        conn = sqlite3.connect(self.db_path)
        df.to_sql('benchmark_results', conn, if_exists='append', index=False)
        
        # Get the inserted row ID
        cursor = conn.cursor()
        row_id = cursor.lastrowid
        conn.close()
        
        return row_id
    
    def query(self,
              benchmark: Optional[str] = None,
              gpu_name: Optional[str] = None,
              start_date: Optional[str] = None,
              end_date: Optional[str] = None,
              limit: Optional[int] = None) -> pd.DataFrame:
        """
        Query benchmark results and return as pandas DataFrame.
        
        Args:
            benchmark: Filter by benchmark type (e.g., 'cache')
            gpu_name: Filter by GPU name (e.g., 'MI325X')
            start_date: Filter results after this date (ISO format)
            end_date: Filter results before this date (ISO format)
            limit: Maximum number of results to return
        
        Returns:
            pandas DataFrame with filtered results
        
        Example:
            # Get all cache benchmarks
            df = db.query(benchmark='cache')
            
            # Get recent results
            df = db.query(benchmark='cache', limit=100)
            
            # Filter by date range
            df = db.query(start_date='2025-10-01', end_date='2025-10-31')
        """
        conditions = []
        params_list = []
        
        query = "SELECT * FROM benchmark_results"
        
        if benchmark:
            conditions.append("benchmark_type = ?")
            params_list.append(benchmark)
        
        if gpu_name:
            conditions.append("gpu_name LIKE ?")
            params_list.append(f'%{gpu_name}%')
        
        if start_date:
            conditions.append("timestamp >= ?")
            params_list.append(start_date)
        
        if end_date:
            conditions.append("timestamp <= ?")
            params_list.append(end_date)
        
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        
        query += " ORDER BY timestamp DESC"
        
        if limit:
            query += f" LIMIT {limit}"
        
        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql_query(query, conn, params=params_list)
        conn.close()
        
        # Convert timestamp to datetime
        if not df.empty:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        return df
    
    def get_latest(self, benchmark: str, n: int = 1) -> pd.DataFrame:
        """
        Get the n most recent results for a benchmark.
        
        Args:
            benchmark: Benchmark name
            n: Number of results to return
        
        Returns:
            DataFrame with n most recent results
        """
        return self.query(benchmark=benchmark, limit=n)
    
    def get_sweep_data(self, 
                       benchmark: str,
                       sweep_param: str = 'problem_size',
                       gpu_name: Optional[str] = None) -> pd.DataFrame:
        """
        Get data suitable for plotting parameter sweeps.
        
        Args:
            benchmark: Benchmark name
            sweep_param: Parameter that was swept (default: 'problem_size')
            gpu_name: Optional GPU filter
        
        Returns:
            DataFrame grouped by sweep parameter with mean/std of metrics
        
        Example:
            df = db.get_sweep_data('cache', 'problem_size')
            df.plot(x='problem_size', y='bandwidth_mean', yerr='bandwidth_std')
        """
        df = self.query(benchmark=benchmark, gpu_name=gpu_name)
        
        if df.empty:
            return df
        
        # Group by sweep parameter and calculate statistics
        grouped = df.groupby(sweep_param).agg({
            'primary_metric': ['mean', 'std', 'count'],
            'spread_percent': 'mean',
            'execution_time_ms': 'mean'
        }).reset_index()
        
        # Flatten column names
        grouped.columns = [
            sweep_param,
            f'{df["metric_name"].iloc[0]}_mean',
            f'{df["metric_name"].iloc[0]}_std',
            'run_count',
            'spread_percent_mean',
            'execution_time_ms_mean'
        ]
        
        return grouped
    
    def export_csv(self, filepath: str, **query_kwargs):
        """
        Export query results to CSV.
        
        Args:
            filepath: Output CSV file path
            **query_kwargs: Arguments passed to query() method
        """
        df = self.query(**query_kwargs)
        df.to_csv(filepath, index=False)
        print(f"Exported {len(df)} results to {filepath}")
    
    def export_json(self, filepath: str, **query_kwargs):
        """
        Export query results to JSON.
        
        Args:
            filepath: Output JSON file path
            **query_kwargs: Arguments passed to query() method
        """
        df = self.query(**query_kwargs)
        df.to_json(filepath, orient='records', date_format='iso', indent=2)
        print(f"Exported {len(df)} results to {filepath}")
    
    def stats(self) -> Dict[str, Any]:
        """
        Get database statistics.
        
        Returns:
            Dictionary with stats (total_results, benchmarks, gpus, etc.)
        """
        conn = sqlite3.connect(self.db_path)
        
        stats = {
            'total_results': pd.read_sql_query(
                "SELECT COUNT(*) as count FROM benchmark_results", conn
            )['count'][0],
            'benchmarks': pd.read_sql_query(
                "SELECT DISTINCT benchmark_type FROM benchmark_results", conn
            )['benchmark_type'].tolist(),
            'gpus': pd.read_sql_query(
                "SELECT DISTINCT gpu_name FROM benchmark_results", conn
            )['gpu_name'].tolist(),
            'date_range': pd.read_sql_query(
                "SELECT MIN(timestamp) as first, MAX(timestamp) as last FROM benchmark_results", 
                conn
            ).to_dict('records')[0]
        }
        
        conn.close()
        return stats
    
    def clear(self, benchmark: Optional[str] = None):
        """
        Clear results from database.
        
        Args:
            benchmark: If specified, only clear this benchmark type.
                      If None, clear all results.
        
        Warning: This permanently deletes data!
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        if benchmark:
            cursor.execute("DELETE FROM benchmark_results WHERE benchmark_type = ?", (benchmark,))
            print(f"Cleared {cursor.rowcount} results for benchmark '{benchmark}'")
        else:
            cursor.execute("DELETE FROM benchmark_results")
            print(f"Cleared all {cursor.rowcount} results")
        
        conn.commit()
        conn.close()
