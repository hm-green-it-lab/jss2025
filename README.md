# Evaluating attribution models and tools for software energy consumption at container, process, and transaction levels – Replication Package

This repository contains the data, scripts, and configuration files required to replicate the experiments from our paper. The structure and contents are aligned with the experiments and analyses described in the paper.

## Folder Structure

Key directories and their contents:

- [`EXPERIMENT_AUTOMATION/`](./EXPERIMENT_AUTOMATION/) – Automation scripts and configurations for running the experiments.
  - [`main.py`](./EXPERIMENT_AUTOMATION/main.py) – Main script to control the experiments
  - [`configuration/`](./EXPERIMENT_AUTOMATION/configuration/) – YAML configuration files for different experiment setups.
  - [`docker/`](./EXPERIMENT_AUTOMATION/docker/) – Docker Compose files and tool-specific configurations (Server-side).
  - [`helper/`](./EXPERIMENT_AUTOMATION/helper/) – Python based helper scripts for measurements.
  - [`orchestrator/`](./EXPERIMENT_AUTOMATION/orchestrator/) – Python modules for controlling and evaluating measurements.
  - [`output/`](./EXPERIMENT_AUTOMATION/output/) – (potentially auto-generated) output files with dedicated format.

- [`EXPERIMENT_RESULTS/`](./EXPERIMENT_RESULTS/) – Experiment raw results including Python scripts for analysis and visualization of measurement results, e.g., [`visualizeLoadLevelContainerPowerConsumptionAsBoxplots.py`](./EXPERIMENT_RESULTS/visualizeLoadLevelContainerPowerConsumptionAsBoxplots.py), [`visualizeIdlePowerConsumptionAsBoxPlot.py`](./EXPERIMENT_RESULTS/visualizeIdlePowerConsumptionAsBoxPlot.py), etc.

## Measurement Tools

In addition to the Python analysis scripts, the following measurement tools were used to collect energy and performance data during the experiments:

| Tool | Description                                                                                                                                                                                |
|------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [PowercapReader](https://github.com/hm-green-it-lab/powercap-reader) | Java-based tool that continuously reads RAPL data via powercap on Linux. Used to collect power consumption measurements for the system.                                                    |
| [ProcFSReader](https://github.com/hm-green-it-lab/procfs-reader) | Java-based tool that continuously reads resource demand data (CPU, memory, I/O, network) for processes from the Linux proc file system. Used to collect process-level performance metrics. |
| [RittalReader](https://github.com/hm-green-it-lab/rittal-reader) | Java-based tool for reading power data from Rittal PDU devices via SNMP. Used for external power measurements.                                                                             |
| [JMeter](https://jmeter.apache.org/) | Load testing tool used to generate HTTP requests to the Spring REST application at controlled rates for each experiment scenario.                                                          |
| [Kepler](https://github.com/sustainable-computing-io/kepler) | Kepler is a Prometheus exporter that measures energy consumption metrics at the container and process level.                                                                               |
| [Scaphandre](https://github.com/hubblo-org/scaphandre) | Scaphandre is an agent for exposing server power and energy consumption metrics.                                                                                                           |
| [JoularJX](https://github.com/joular/joularjx) | Java agent for measuring energy consumption of JVM-based applications at the process, thread, and method level.                                                                            |
| [OTJAE](https://github.com/RETIT/opentelemetry-javaagent-extension) | OpenTelemetry Java-Agent Extension for attributing energy consumption to Java processes and transactions.                                                                                  |

These tools were orchestrated and synchronized using the automation scripts described below to ensure reproducible and accurate measurements across all experiment runs.

## Experiment Automation 

For details on running experiments, see the [`EXPERIMENT_AUTOMATION/README.md`](./EXPERIMENT_AUTOMATION/README.md) which describes usage, configuration, and automation scripts in depth.

## Experiment Results

This [folder](./EXPERIMENT_RESULTS/) contains the raw measurement data and analysis scripts for the experiments presented in our paper.

The experiments were conducted using a Spring REST application deployed in Docker, with energy and performance measurements taken under varying load levels. For each experiment run, the system was subjected to one of the following load intensities: **0**, **230**, **350**, **480** and **560** requests per second (RPS) on three distinct REST-endpoints each. All measurement data and results are organized by timestamp and configuration.

The Python scripts in the [`EXPERIMENT_RESULTS/`](./EXPERIMENT_RESULTS/) folder process the raw data and generate the figures and tables used in the paper.

| **Load Level (RPS)** | **Description** |
|----------------------|-----------------|
| 0                    | System idle, no external load applied. Serves as the baseline for energy and performance measurements. Results in CPU utilization of approximately 0%. |
| 230                  | Moderate load: all three REST endpoints are stressed with 230 RPS. Results in CPU utilization of about 25%. |
| 350                  | High load: all three REST endpoints are stressed with 350 RPS. Results in CPU utilization of roughly 50%. |
| 480                  | Very high load: all three REST endpoints are stressed with 480 RPS. Results in CPU utilization of around 75%. |
| 560                  | Maximum load: all three REST endpoints are stressed with 560 RPS. Results in CPU utilization close to 100%. |

For each load level, the experiment was repeated three times to ensure validity. The results for each run are stored in dedicated `.zip` files within [`EXPERIMENT_RESULTS/`](./EXPERIMENT_RESULTS/), organized by load level and timestamps as mentioned. Each `.zip` archive contains all measurement data and logs for a single experiment run.

## Python Scripts for Generating Figures and Tables

This repository contains several Python scripts for processing, analyzing, and visualizing the experimental results. These scripts are located in the [`EXPERIMENT_RESULTS/`](./EXPERIMENT_RESULTS/) folder:

| Script Name | Description |
|-------------|-------------|
| [`create_power_consumption_barchart.py`](./EXPERIMENT_RESULTS/create_power_consumption_barchart.py) | Processes measurement data and generates bar charts of power consumption for different loads and scenarios. |
| [`visualizeIdlePowerConsumptionAsBoxPlot.py`](./EXPERIMENT_RESULTS/visualizeIdlePowerConsumptionAsBoxPlot.py) | Visualizes idle power consumption as boxplots to compare baseline measurements. |
| [`visualizeLoadLevelContainerPowerConsumptionAsBoxplots.py`](./EXPERIMENT_RESULTS/visualizeLoadLevelContainerPowerConsumptionAsBoxplots.py) | Creates boxplots of container-level power consumption across different load levels. |
| [`visualizeLoadLevelProcessPowerConsumptionAsBoxplots.py`](./EXPERIMENT_RESULTS/visualizeLoadLevelProcessPowerConsumptionAsBoxplots.py) | Creates boxplots of process-level power consumption across different load levels. |
| [`visualizeLoadLevelSystemPowerConsumptionAsBoxplots.py`](./EXPERIMENT_RESULTS/visualizeLoadLevelSystemPowerConsumptionAsBoxplots.py) | Creates boxplots of system-level power consumption across different load levels. |
| [`visualizeLoadLevelTransactionPowerConsumptionAsBoxplots.py`](./EXPERIMENT_RESULTS/visualizeLoadLevelTransactionPowerConsumptionAsBoxplots.py) | Visualizes transaction-level power consumption as boxplots for each load scenario. |
| [`visualizePowerCapAsBoxplot.py`](./EXPERIMENT_RESULTS/visualizePowerCapAsBoxplot.py) | Visualizes power cap measurements as boxplots. |
| [`createCpuUtilizationTableForAllLoadLevelsAndScenarios.py`](./EXPERIMENT_RESULTS/createCpuUtilizationTableForAllLoadLevelsAndScenarios.py) | Generates tables summarizing CPU utilization for all load levels and scenarios. |

## Notes

- The configurations and scripts are designed to be flexible for different measurement and analysis scenarios.
- Administrator rights may be required for using Docker and running experiments.
- Detailed descriptions of the experiments, measurements, and analyses can be found in the paper.

## Contact

For questions regarding replication or use of the data, please contact the authors of the paper.