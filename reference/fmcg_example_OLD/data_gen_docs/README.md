# FMCG Data Generation Module

The `data_generation` module is a high-performance, modular synthetic data generator designed to simulate a complete Fast-Moving Consumer Goods (FMCG) supply chain. It generates relationally consistent data across 15 dependency levels, from raw ingredients to consumer sales and returns.

## Key Features

*   **Scale:** Capable of generating 14.7M+ rows efficiently using vectorized NumPy operations.
*   **Realism:** Incorporates industry-standard distributions (Zipf/Pareto for SKUs, Poisson/Gamma for demand) and structural patterns (hub-and-spoke networks).
*   **Consistency:** Level-based architecture ensures referential integrity across 67 tables.
*   **Chaos Engineering:** Injects realistic pathologies ("quirks") and deterministic risk events (strikes, recalls) for stress testing.
*   **Validation:** Online `RealismMonitor` validates data quality and distribution patterns in real-time.

## Structure

*   `generators/`: Stateless generator classes for each of the 15 levels (0-14).
*   `vectorized.py`: Optimized NumPy generators for high-volume tables (POS, Orders).
*   `realism_monitor.py`: Streaming validator for checking data drift and anomalies.
*   `risk_events.py` & `quirks.py`: Engines for injecting supply chain disruptions.
*   `streaming_writer.py`: Memory-efficient PostgreSQL COPY output writer.

## Getting Started

See [Usage](usage.md) for CLI commands and examples.
See [Architecture](architecture.md) for deep dive into the 15-level generation process.
See [Validation](validation.md) for details on quality checks and chaos injection.
