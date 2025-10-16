"""
Test suite for GPU STREAM benchmark

Tests all 4 STREAM kernel types (init, read, scale, triad) to verify:
- Kernels compile and execute without crashes
- Bandwidth calculations are reasonable for the hardware
- Different kernel types show expected bandwidth scaling
"""

import pytest
import _rocmGPUBenches


@pytest.fixture
def stream_runner():
    """Create a stream benchmark runner instance"""
    return _rocmGPUBenches.create_stream_benchmark_runner()


@pytest.fixture
def test_problem_size():
    """Standard problem size for tests (100M doubles = 800 MB)"""
    return 100_000_000


class TestStreamBenchmark:
    """Test suite for STREAM benchmark kernels"""
    
    def test_triad_kernel(self, stream_runner, test_problem_size):
        """Test classic STREAM triad: A = B*D + C (4 streams)"""
        result = stream_runner.run(
            "stream",
            {
                "problem_size": test_problem_size,
                "kernel_type": "triad",
                "block_size": 256
            }
        )
        
        assert result.metric_name == "bandwidth_gbs"
        assert result.primary_metric > 0, "Bandwidth should be positive"
        assert result.primary_metric < 10000, "Bandwidth suspiciously high"
        assert result.exec_time_ms > 0, "Execution time should be positive"
        
    def test_scale_kernel(self, stream_runner, test_problem_size):
        """Test STREAM scale: A = α·B (2 streams)"""
        result = stream_runner.run(
            "stream",
            {
                "problem_size": test_problem_size,
                "kernel_type": "scale",
                "block_size": 256
            }
        )
        
        assert result.metric_name == "bandwidth_gbs"
        assert result.primary_metric > 0
        assert result.exec_time_ms > 0
        
    def test_read_kernel(self, stream_runner, test_problem_size):
        """Test STREAM read: temp = B (1 stream, read-only)"""
        result = stream_runner.run(
            "stream",
            {
                "problem_size": test_problem_size,
                "kernel_type": "read",
                "block_size": 256
            }
        )
        
        assert result.metric_name == "bandwidth_gbs"
        assert result.primary_metric > 0
        assert result.exec_time_ms > 0
        
    def test_init_kernel(self, stream_runner, test_problem_size):
        """Test STREAM init: A = scalar (1 stream, write-only)"""
        result = stream_runner.run(
            "stream",
            {
                "problem_size": test_problem_size,
                "kernel_type": "init",
                "block_size": 256
            }
        )
        
        assert result.metric_name == "bandwidth_gbs"
        assert result.primary_metric > 0
        assert result.exec_time_ms > 0
        
    def test_bandwidth_scaling(self, stream_runner, test_problem_size):
        """Verify bandwidth scales with stream count as expected"""
        results = {}
        
        for kernel_type in ["init", "read", "scale", "triad"]:
            result = stream_runner.run(
                "stream",
                {
                    "problem_size": test_problem_size,
                    "kernel_type": kernel_type,
                    "block_size": 256
                }
            )
            results[kernel_type] = result.primary_metric
        
        # Triad (4 streams) should have highest bandwidth
        # Scale (2 streams) should be intermediate
        # Init/read (1 stream each) should be lowest
        # Note: This is a rough heuristic, not always true due to hardware details
        assert results["triad"] > results["scale"], \
            "Triad (4 streams) should exceed scale (2 streams)"
        
    def test_small_problem_size(self, stream_runner):
        """Test with small problem that fits in cache"""
        small_size = 1_000_000  # 1M doubles = 8 MB
        
        result = stream_runner.run(
            "stream",
            {
                "problem_size": small_size,
                "kernel_type": "triad",
                "block_size": 256
            }
        )
        
        assert result.primary_metric > 0
        assert result.exec_time_ms > 0
        
    def test_large_problem_size(self, stream_runner):
        """Test with large problem that stresses memory system"""
        large_size = 500_000_000  # 500M doubles = 4 GB
        
        result = stream_runner.run(
            "stream",
            {
                "problem_size": large_size,
                "kernel_type": "triad",
                "block_size": 256
            }
        )
        
        assert result.primary_metric > 0
        assert result.exec_time_ms > 0
        
    def test_different_block_sizes(self, stream_runner, test_problem_size):
        """Test different block sizes (occupancy variations)"""
        block_sizes = [64, 128, 256, 512, 1024]
        
        for block_size in block_sizes:
            result = stream_runner.run(
                "stream",
                {
                    "problem_size": test_problem_size,
                    "kernel_type": "triad",
                    "block_size": block_size
                }
            )
            
            assert result.primary_metric > 0, \
                f"Block size {block_size} should produce valid bandwidth"


if __name__ == "__main__":
    # Allow running tests directly with: python test_stream_benchmark.py
    pytest.main([__file__, "-v"])
