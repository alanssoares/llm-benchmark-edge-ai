#!/usr/bin/env python3
"""
LLM Benchmark for Edge AI
A comprehensive benchmarking tool for evaluating Large Language Models on edge devices.
"""

import argparse
import json
import logging
import os
import sys
import time
import tracemalloc
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import colorlog
import psutil
import torch
import yaml
from tabulate import tabulate
from transformers import pipeline, Pipeline
from tqdm import tqdm


class BenchmarkConfig:
    """Configuration management for the benchmark."""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or "models/config.yaml"
        self.prompts_path = "prompts/prompts.yaml"
        self.config = self._load_config()
        self.prompts = self._load_prompts()
    
    def _load_config(self) -> Dict:
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            logging.warning(f"Config file {self.config_path} not found. Using defaults.")
            return self._default_config()
        except yaml.YAMLError as e:
            logging.error(f"Error parsing config file: {e}")
            return self._default_config()
    
    def _load_prompts(self) -> Dict:
        """Load prompts from YAML file."""
        try:
            with open(self.prompts_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            logging.warning(f"Prompts file {self.prompts_path} not found. Using default.")
            return {"prompts": {"default": "Explique o conceito de aprendizado por reforço."}}
        except yaml.YAMLError as e:
            logging.error(f"Error parsing prompts file: {e}")
            return {"prompts": {"default": "Explique o conceito de aprendizado por reforço."}}
    
    def _default_config(self) -> Dict:
        """Return default configuration."""
        return {
            "models": {
                "TinyLlama": {
                    "id": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                    "type": "text-generation",
                    "max_tokens": 50
                },
                "Phi-1.5": {
                    "id": "microsoft/phi-1_5",
                    "type": "text-generation",
                    "max_tokens": 50
                },
                "Flan-T5": {
                    "id": "google/flan-t5-small",
                    "type": "text2text-generation",
                    "max_tokens": 50
                }
            },
            "benchmark": {
                "warmup_runs": 1,
                "measurement_runs": 3,
                "max_memory_gb": 8,
                "timeout_seconds": 300
            },
            "output": {
                "format": "table",
                "save_results": True,
                "output_file": "benchmark_results.json"
            }
        }


class SystemMonitor:
    """Monitor system resources during benchmarking."""
    
    def __init__(self):
        self.process = psutil.Process()
        self.initial_memory = self.get_memory_usage()
        
    def get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        return self.process.memory_info().rss / 1024**2
    
    def get_cpu_percent(self) -> float:
        """Get current CPU usage percentage."""
        return self.process.cpu_percent()
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get comprehensive system information."""
        return {
            "cpu_count": os.cpu_count(),
            "total_memory_gb": psutil.virtual_memory().total / 1024**3,
            "available_memory_gb": psutil.virtual_memory().available / 1024**3,
            "gpu_available": torch.cuda.is_available(),
            "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "python_version": sys.version,
            "torch_version": torch.__version__
        }


class ModelBenchmark:
    """Benchmark individual models."""
    
    def __init__(self, config: BenchmarkConfig, monitor: SystemMonitor):
        self.config = config
        self.monitor = monitor
        self.logger = logging.getLogger(__name__)
        
    def benchmark_model(self, name: str, model_config: Dict, prompt: str) -> Dict[str, Any]:
        """Benchmark a single model with the given prompt."""
        self.logger.info(f"Benchmarking {name}")
        
        results = {
            "model_name": name,
            "model_id": model_config["id"],
            "prompt": prompt,
            "success": False,
            "error": None,
            "metrics": {}
        }
        
        try:
            # Load model with timeout
            load_start = time.time()
            tracemalloc.start()
            
            pipe = self._load_model(model_config)
            
            load_time = time.time() - load_start
            current, peak_load = tracemalloc.get_traced_memory()
            
            # Warmup runs
            for _ in range(self.config.config["benchmark"]["warmup_runs"]):
                try:
                    _ = pipe(prompt, max_new_tokens=model_config.get("max_tokens", 50))
                except Exception as e:
                    self.logger.warning(f"Warmup failed for {name}: {e}")
            
            # Measurement runs
            inference_times = []
            memory_usages = []
            
            for run in range(self.config.config["benchmark"]["measurement_runs"]):
                mem_before = self.monitor.get_memory_usage()
                
                start_time = time.time()
                output = pipe(prompt, max_new_tokens=model_config.get("max_tokens", 50))
                end_time = time.time()
                
                mem_after = self.monitor.get_memory_usage()
                
                inference_times.append(end_time - start_time)
                memory_usages.append(mem_after - mem_before)
                
                # Store output from first run
                if run == 0:
                    results["output"] = self._extract_text(output)
            
            current, peak_inference = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            
            # Calculate metrics
            results["metrics"] = {
                "load_time_s": round(load_time, 3),
                "avg_inference_time_s": round(sum(inference_times) / len(inference_times), 3),
                "min_inference_time_s": round(min(inference_times), 3),
                "max_inference_time_s": round(max(inference_times), 3),
                "avg_memory_delta_mb": round(sum(memory_usages) / len(memory_usages), 2),
                "peak_memory_load_mb": round(peak_load / 1024**2, 2),
                "peak_memory_inference_mb": round(peak_inference / 1024**2, 2),
                "throughput_tokens_per_s": round(model_config.get("max_tokens", 50) / (sum(inference_times) / len(inference_times)), 2)
            }
            
            results["success"] = True
            self.logger.info(f"Successfully benchmarked {name}")
            
        except Exception as e:
            error_msg = f"Error benchmarking {name}: {str(e)}"
            self.logger.error(error_msg)
            results["error"] = error_msg
            
        finally:
            # Cleanup
            if 'pipe' in locals():
                del pipe
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
        return results
    
    def _load_model(self, model_config: Dict) -> Pipeline:
        """Load a model pipeline with error handling."""
        device = 0 if torch.cuda.is_available() else -1
        
        try:
            pipe = pipeline(
                model_config["type"],
                model=model_config["id"],
                device=device,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
            )
            return pipe
        except Exception as e:
            self.logger.warning(f"Failed to load model with device {device}, trying CPU: {e}")
            # Fallback to CPU
            return pipeline(
                model_config["type"],
                model=model_config["id"],
                device=-1
            )
    
    def _extract_text(self, output: List[Dict]) -> str:
        """Extract generated text from model output."""
        if not output:
            return ""
        
        first_output = output[0]
        if 'generated_text' in first_output:
            return first_output['generated_text']
        elif 'text' in first_output:
            return first_output['text']
        else:
            return str(first_output)


class BenchmarkRunner:
    """Main benchmark runner."""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = BenchmarkConfig(config_path)
        self.monitor = SystemMonitor()
        self.benchmark = ModelBenchmark(self.config, self.monitor)
        self.logger = logging.getLogger(__name__)
        
    def run(self, models: Optional[List[str]] = None, prompt_category: str = "default") -> Dict[str, Any]:
        """Run the complete benchmark suite."""
        self.logger.info("Starting LLM Edge AI Benchmark")
        
        # Get system info
        system_info = self.monitor.get_system_info()
        self.logger.info(f"System: {system_info['cpu_count']} CPUs, "
                        f"{system_info['total_memory_gb']:.1f}GB RAM, "
                        f"GPU: {system_info['gpu_available']}")
        
        # Select models to benchmark
        model_configs = self.config.config["models"]
        if models:
            model_configs = {k: v for k, v in model_configs.items() if k in models}
        
        # Select prompt
        prompt = self._get_prompt(prompt_category)
        self.logger.info(f"Using prompt: {prompt[:50]}...")
        
        # Run benchmarks
        results = []
        total_models = len(model_configs)
        
        with tqdm(total=total_models, desc="Benchmarking models") as pbar:
            for name, config in model_configs.items():
                result = self.benchmark.benchmark_model(name, config, prompt)
                results.append(result)
                pbar.set_description(f"Completed {name}")
                pbar.update(1)
        
        # Compile final results
        benchmark_results = {
            "timestamp": datetime.now().isoformat(),
            "system_info": system_info,
            "prompt": prompt,
            "models": results,
            "summary": self._generate_summary(results)
        }
        
        # Output results
        self._output_results(benchmark_results)
        
        return benchmark_results
    
    def _get_prompt(self, category: str) -> str:
        """Get prompt for the specified category."""
        prompts = self.config.prompts.get("prompts", {})
        
        if category == "default":
            return prompts.get("default", "Explique o conceito de aprendizado por reforço.")
        
        categories = prompts.get("categories", {})
        if category in categories:
            # Return first prompt from category
            return categories[category][0]
        
        self.logger.warning(f"Prompt category '{category}' not found, using default")
        return prompts.get("default", "Explique o conceito de aprendizado por reforço.")
    
    def _generate_summary(self, results: List[Dict]) -> Dict[str, Any]:
        """Generate summary statistics."""
        successful_results = [r for r in results if r["success"]]
        
        if not successful_results:
            return {"error": "No successful benchmarks"}
        
        avg_load_time = sum(r["metrics"]["load_time_s"] for r in successful_results) / len(successful_results)
        avg_inference_time = sum(r["metrics"]["avg_inference_time_s"] for r in successful_results) / len(successful_results)
        
        fastest_model = min(successful_results, key=lambda x: x["metrics"]["avg_inference_time_s"])
        most_efficient = min(successful_results, key=lambda x: x["metrics"]["peak_memory_inference_mb"])
        
        return {
            "total_models": len(results),
            "successful_models": len(successful_results),
            "failed_models": len(results) - len(successful_results),
            "avg_load_time_s": round(avg_load_time, 3),
            "avg_inference_time_s": round(avg_inference_time, 3),
            "fastest_model": {
                "name": fastest_model["model_name"],
                "time_s": fastest_model["metrics"]["avg_inference_time_s"]
            },
            "most_memory_efficient": {
                "name": most_efficient["model_name"],
                "memory_mb": most_efficient["metrics"]["peak_memory_inference_mb"]
            }
        }
    
    def _output_results(self, results: Dict[str, Any]):
        """Output results in the specified format."""
        output_config = self.config.config["output"]
        
        # Console output
        if output_config["format"] == "table":
            self._print_table(results)
        elif output_config["format"] == "json":
            print(json.dumps(results, indent=2, ensure_ascii=False))
        
        # Save to file
        if output_config.get("save_results", False):
            output_file = output_config.get("output_file", "benchmark_results.json")
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            self.logger.info(f"Results saved to {output_file}")
    
    def _print_table(self, results: Dict[str, Any]):
        """Print results in a formatted table."""
        successful_models = [r for r in results["models"] if r["success"]]
        
        if not successful_models:
            print("❌ No successful benchmarks to display")
            return
        
        # Create table data
        headers = [
            "Model", "Load Time (s)", "Avg Inference (s)", 
            "Peak Memory (MB)", "Throughput (tok/s)", "Status"
        ]
        
        table_data = []
        for result in results["models"]:
            if result["success"]:
                metrics = result["metrics"]
                row = [
                    result["model_name"],
                    f"{metrics['load_time_s']:.2f}",
                    f"{metrics['avg_inference_time_s']:.2f}",
                    f"{metrics['peak_memory_inference_mb']:.1f}",
                    f"{metrics['throughput_tokens_per_s']:.1f}",
                    "✅ Success"
                ]
            else:
                row = [
                    result["model_name"],
                    "-", "-", "-", "-",
                    "❌ Failed"
                ]
            table_data.append(row)
        
        print("\n" + "="*80)
        print("🚀 LLM Edge AI Benchmark Results")
        print("="*80)
        print(tabulate(table_data, headers=headers, tablefmt="grid"))
        
        # Print summary
        summary = results["summary"]
        print(f"\n📊 Summary:")
        print(f"   • Successful models: {summary['successful_models']}/{summary['total_models']}")
        print(f"   • Fastest model: {summary['fastest_model']['name']} ({summary['fastest_model']['time_s']:.2f}s)")
        print(f"   • Most efficient: {summary['most_memory_efficient']['name']} ({summary['most_memory_efficient']['memory_mb']:.1f}MB)")


def setup_logging(level: str = "INFO"):
    """Setup colored logging."""
    log_level = getattr(logging, level.upper())
    
    # Create colored formatter
    formatter = colorlog.ColoredFormatter(
        "%(log_color)s%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%H:%M:%S",
        log_colors={
            'DEBUG': 'cyan',
            'INFO': 'green',
            'WARNING': 'yellow',
            'ERROR': 'red',
            'CRITICAL': 'red,bg_white',
        }
    )
    
    # Setup handler
    handler = colorlog.StreamHandler()
    handler.setFormatter(formatter)
    
    # Setup logger
    logger = logging.getLogger()
    logger.setLevel(log_level)
    logger.addHandler(handler)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="LLM Edge AI Benchmark Tool")
    parser.add_argument("--models", nargs="+", help="Specific models to benchmark")
    parser.add_argument("--config", help="Path to config file")
    parser.add_argument("--prompt-category", default="default", 
                       help="Prompt category to use (default, reasoning, creative, factual, portuguese)")
    parser.add_argument("--log-level", default="INFO", 
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    
    try:
        # Create and run benchmark
        runner = BenchmarkRunner(args.config)
        results = runner.run(args.models, args.prompt_category)
        
        # Exit with success
        sys.exit(0)
        
    except KeyboardInterrupt:
        logging.info("Benchmark interrupted by user")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Benchmark failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()