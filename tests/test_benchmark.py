"""
Tests for the LLM Benchmark tool.
"""

import json
import os
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

# Add the parent directory to the path so we can import benchmark
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import benchmark


class TestBenchmarkConfig(unittest.TestCase):
    """Test the BenchmarkConfig class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Create test config file
        self.config_content = {
            "models": {
                "TestModel": {
                    "id": "test/model",
                    "type": "text-generation",
                    "max_tokens": 10
                }
            },
            "benchmark": {
                "warmup_runs": 1,
                "measurement_runs": 1,
                "max_memory_gb": 4,
                "timeout_seconds": 60
            },
            "output": {
                "format": "json",
                "save_results": False
            }
        }
        
        self.config_file = os.path.join(self.temp_dir, "config.yaml")
        with open(self.config_file, 'w') as f:
            import yaml
            yaml.dump(self.config_content, f)
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_load_config_success(self):
        """Test successful config loading."""
        config = benchmark.BenchmarkConfig(self.config_file)
        self.assertEqual(config.config["models"]["TestModel"]["id"], "test/model")
    
    def test_load_config_missing_file(self):
        """Test config loading with missing file."""
        config = benchmark.BenchmarkConfig("nonexistent.yaml")
        # Should fall back to default config
        self.assertIn("models", config.config)
    
    def test_default_config(self):
        """Test default configuration."""
        config = benchmark.BenchmarkConfig()
        default_config = config._default_config()
        
        self.assertIn("models", default_config)
        self.assertIn("benchmark", default_config)
        self.assertIn("output", default_config)


class TestSystemMonitor(unittest.TestCase):
    """Test the SystemMonitor class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.monitor = benchmark.SystemMonitor()
    
    def test_get_memory_usage(self):
        """Test memory usage measurement."""
        memory = self.monitor.get_memory_usage()
        self.assertIsInstance(memory, float)
        self.assertGreater(memory, 0)
    
    def test_get_system_info(self):
        """Test system information gathering."""
        info = self.monitor.get_system_info()
        
        required_keys = [
            "cpu_count", "total_memory_gb", "available_memory_gb",
            "gpu_available", "gpu_count", "python_version", "torch_version"
        ]
        
        for key in required_keys:
            self.assertIn(key, info)


class TestModelBenchmark(unittest.TestCase):
    """Test the ModelBenchmark class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = benchmark.BenchmarkConfig()
        self.monitor = benchmark.SystemMonitor()
        self.benchmark_obj = benchmark.ModelBenchmark(self.config, self.monitor)
    
    def test_extract_text(self):
        """Test text extraction from model output."""
        # Test with generated_text key
        output1 = [{"generated_text": "Hello world"}]
        result1 = self.benchmark_obj._extract_text(output1)
        self.assertEqual(result1, "Hello world")
        
        # Test with text key
        output2 = [{"text": "Hello world"}]
        result2 = self.benchmark_obj._extract_text(output2)
        self.assertEqual(result2, "Hello world")
        
        # Test with empty output
        output3 = []
        result3 = self.benchmark_obj._extract_text(output3)
        self.assertEqual(result3, "")


class TestBenchmarkRunner(unittest.TestCase):
    """Test the BenchmarkRunner class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a minimal config for testing
        self.temp_dir = tempfile.mkdtemp()
        
        config_content = {
            "models": {
                "TestModel": {
                    "id": "test/model",
                    "type": "text-generation",
                    "max_tokens": 10
                }
            },
            "benchmark": {
                "warmup_runs": 0,
                "measurement_runs": 1,
                "max_memory_gb": 4,
                "timeout_seconds": 60
            },
            "output": {
                "format": "json",
                "save_results": False
            }
        }
        
        self.config_file = os.path.join(self.temp_dir, "config.yaml")
        with open(self.config_file, 'w') as f:
            import yaml
            yaml.dump(config_content, f)
        
        # Create prompts file
        prompts_content = {
            "prompts": {
                "default": "Test prompt"
            }
        }
        
        os.makedirs(os.path.join(self.temp_dir, "prompts"), exist_ok=True)
        prompts_file = os.path.join(self.temp_dir, "prompts", "prompts.yaml")
        with open(prompts_file, 'w') as f:
            import yaml
            yaml.dump(prompts_content, f)
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_get_prompt_default(self):
        """Test default prompt retrieval."""
        # Mock the config paths
        with patch.object(benchmark.BenchmarkConfig, '__init__', lambda x, y=None: None):
            runner = benchmark.BenchmarkRunner()
            runner.config = Mock()
            runner.config.prompts = {"prompts": {"default": "Test prompt"}}
            
            prompt = runner._get_prompt("default")
            self.assertEqual(prompt, "Test prompt")
    
    def test_generate_summary_no_results(self):
        """Test summary generation with no successful results."""
        with patch.object(benchmark.BenchmarkConfig, '__init__', lambda x, y=None: None):
            runner = benchmark.BenchmarkRunner()
            runner.config = Mock()
            runner.monitor = Mock()
            
            results = [{"success": False, "error": "Test error"}]
            summary = runner._generate_summary(results)
            
            self.assertIn("error", summary)
    
    def test_generate_summary_with_results(self):
        """Test summary generation with successful results."""
        with patch.object(benchmark.BenchmarkConfig, '__init__', lambda x, y=None: None):
            runner = benchmark.BenchmarkRunner()
            runner.config = Mock()
            runner.monitor = Mock()
            
            results = [
                {
                    "success": True,
                    "model_name": "TestModel",
                    "metrics": {
                        "load_time_s": 10.0,
                        "avg_inference_time_s": 2.0,
                        "peak_memory_inference_mb": 500.0
                    }
                }
            ]
            
            summary = runner._generate_summary(results)
            
            self.assertEqual(summary["total_models"], 1)
            self.assertEqual(summary["successful_models"], 1)
            self.assertEqual(summary["failed_models"], 0)


class TestUtilityFunctions(unittest.TestCase):
    """Test utility functions."""
    
    def test_setup_logging(self):
        """Test logging setup."""
        # Test that setup_logging doesn't raise an exception
        try:
            benchmark.setup_logging("INFO")
            benchmark.setup_logging("DEBUG")
        except Exception as e:
            self.fail(f"setup_logging raised an exception: {e}")


if __name__ == "__main__":
    unittest.main()