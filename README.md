# 🚀 LLM Benchmark for Edge AI

A comprehensive benchmarking tool for evaluating Large Language Models (LLMs) performance on edge devices. This tool provides detailed metrics on inference time, memory usage, and throughput for various models.

## ✨ Features

- **Multi-model benchmarking**: Support for various LLM architectures
- **Comprehensive metrics**: Inference time, memory usage, throughput, and more
- **Configurable prompts**: Multiple prompt categories for different evaluation scenarios
- **Resource monitoring**: Real-time CPU and memory tracking
- **GPU support**: Automatic GPU detection and utilization
- **Multiple output formats**: Table, JSON, and CSV export options
- **Docker support**: Containerized execution for reproducibility
- **Security-focused**: Non-root container execution

## 🛠️ Installation

### Local Installation

```bash
# Clone the repository
git clone https://github.com/alanssoares/llm-benchmark-edge-ai.git
cd llm-benchmark-edge-ai

# Install dependencies
pip install -r requirements.txt

# Run benchmark
python benchmark.py
```

### Docker Installation

```bash
# Build the image
docker build -t llm-benchmark .

# Run the container
docker run --rm -v $(pwd)/results:/app/results llm-benchmark
```

## 🚀 Usage

### Basic Usage

```bash
# Run all models with default settings
python benchmark.py

# Run specific model with prompt category
python benchmark.py --models TinyLlama --prompt-category reasoning --log-level INFO

# Run specific models
python benchmark.py --models TinyLlama Phi-1.5

# Use different prompt category
python benchmark.py --prompt-category reasoning

# Save results and enable debug logging
python benchmark.py --log-level DEBUG
```

### Command Line Options

```bash
python benchmark.py [OPTIONS]

Options:
  --models MODELS [MODELS ...]     Specific models to benchmark
  --config CONFIG                  Path to config file
  --prompt-category CATEGORY       Prompt category (default, reasoning, creative, factual, portuguese)
  --log-level LEVEL               Log level (DEBUG, INFO, WARNING, ERROR)
  --help                          Show help message
```

### Configuration

The tool uses YAML configuration files for flexibility:

#### Model Configuration (`models/config.yaml`)

```yaml
models:
  TinyLlama:
    id: "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    type: "text-generation"
    max_tokens: 50
  # Add more models...

benchmark:
  warmup_runs: 1
  measurement_runs: 3
  max_memory_gb: 8
  timeout_seconds: 300

output:
  format: "table"  # table, csv, json
  save_results: true
  output_file: "benchmark_results.json"
```

#### Prompt Configuration (`prompts/prompts.yaml`)

```yaml
prompts:
  default: "Explique o conceito de aprendizado por reforço."
  
  categories:
    reasoning:
      - "Se todos os gatos são mamíferos..."
    creative:
      - "Escreva uma história curta..."
    # Add more categories...
```

## 📊 Output

### Table Format (Default)

```
🚀 LLM Edge AI Benchmark Results
================================================================================
┌─────────────┬──────────────┬─────────────────┬─────────────────┬──────────────────┬───────────┐
│ Model       │ Load Time (s)│ Avg Inference  │ Peak Memory     │ Throughput       │ Status    │
│             │              │ (s)             │ (MB)            │ (tok/s)          │           │
├─────────────┼──────────────┼─────────────────┼─────────────────┼──────────────────┼───────────┤
│ TinyLlama   │ 12.45        │ 2.34            │ 1024.5          │ 21.4             │ ✅ Success │
│ Phi-1.5     │ 8.92         │ 1.87            │ 856.2           │ 26.7             │ ✅ Success │
│ Flan-T5     │ 6.23         │ 1.45            │ 512.8           │ 34.5             │ ✅ Success │
└─────────────┴──────────────┴─────────────────┴─────────────────┴──────────────────┴───────────┘

📊 Summary:
   • Successful models: 3/3
   • Fastest model: Flan-T5 (1.45s)
   • Most efficient: Flan-T5 (512.8MB)
```

### JSON Format

Results are also saved as structured JSON for further analysis:

```json
{
  "timestamp": "2025-10-07T10:30:00",
  "system_info": {
    "cpu_count": 8,
    "total_memory_gb": 16.0,
    "gpu_available": true
  },
  "models": [
    {
      "model_name": "TinyLlama",
      "success": true,
      "metrics": {
        "load_time_s": 12.45,
        "avg_inference_time_s": 2.34,
        "peak_memory_inference_mb": 1024.5,
        "throughput_tokens_per_s": 21.4
      }
    }
  ]
}
```

## 🏗️ Architecture

The tool is structured with the following components:

- **BenchmarkConfig**: Configuration management and file loading
- **SystemMonitor**: Resource monitoring and system information
- **ModelBenchmark**: Individual model benchmarking logic
- **BenchmarkRunner**: Main orchestration and result compilation

## 🔧 Supported Models

Currently tested models include:

- **TinyLlama**: Compact chat model optimized for edge deployment
- **Microsoft Phi-1.5**: Efficient small language model
- **Google Flan-T5**: Text-to-text transfer transformer

### Adding New Models

Add models to `models/config.yaml`:

```yaml
models:
  YourModel:
    id: "organization/model-name"
    type: "text-generation"  # or "text2text-generation"
    max_tokens: 50
```

## 🚨 System Requirements

### Minimum Requirements
- Python 3.8+
- 4GB RAM
- 2GB free disk space

### Recommended Requirements
- Python 3.10+
- 8GB+ RAM
- GPU with CUDA support
- 10GB+ free disk space

## 🔒 Security

- **Non-root execution**: Docker containers run as non-root user
- **Resource limits**: Configurable memory and timeout limits
- **Input validation**: Comprehensive input sanitization

## 🐛 Troubleshooting

### Common Issues

1. **Out of Memory Errors**
   ```bash
   # Reduce batch size or model size
   python benchmark.py --models Flan-T5
   ```

2. **Model Download Failures**
   ```bash
   # Check internet connection and disk space
   # Try running with debug logging
   python benchmark.py --log-level DEBUG
   ```

3. **GPU Issues**
   ```bash
   # Verify CUDA installation
   python -c "import torch; print(torch.cuda.is_available())"
   ```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Hugging Face Transformers library
- PyTorch framework
- All the model creators and maintainers

## 📈 Roadmap

- [ ] Support for more model architectures
- [ ] Real-time performance monitoring
- [ ] Model quantization benchmarks
- [ ] Web interface for results visualization
- [ ] Integration with MLflow for experiment tracking
- [ ] Support for custom evaluation metrics