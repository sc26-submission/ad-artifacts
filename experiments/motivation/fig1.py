from dataclasses import dataclass
from pathlib import Path
import math

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter


# ==================================================================
# Scenario configuration
# ==================================================================

@dataclass(frozen=True)
class Scenario:
    """Configuration for a simulated two-job workload."""

    name: str
    fast_job_name: str
    slow_job_name: str

    # Accelerator-side execution time per mini-batch.
    fast_execution_time: float
    slow_execution_time: float

    # Size of one prepared mini-batch stored in the cache.
    cached_batch_size_bytes: float

    # Mini-batches processed by each job over the complete run.
    total_batches_per_job: int

    output_filename: str


# ==================================================================
# Tick formatters
# ==================================================================

def throughput_formatter(value, _):
    """Format throughput-axis values."""
    if abs(value - round(value)) < 1e-6:
        return f"{value:.0f}"
    return f"{value:.1f}"


def cache_tb_formatter(value, _):
    """Format cache-capacity ticks."""
    if value >= 100:
        return f"{value:.0f}"
    if value >= 1:
        return f"{value:.1f}"
    return f"{value:.2f}"


# ==================================================================
# Simulation
# ==================================================================

def simulate_scenario(scenario: Scenario, stall_times: np.ndarray):
    """
    Simulate two jobs consuming the same sequence of prepared mini-batches.

    For each job:

        throughput = 1 / (execution time + data stall)

    Peak cache requirement is measured when the faster job completes
    its specified number of mini-batches.
    """

    fast_batch_times = scenario.fast_execution_time + stall_times
    slow_batch_times = scenario.slow_execution_time + stall_times

    fast_throughput = 1.0 / fast_batch_times
    slow_throughput = 1.0 / slow_batch_times
    aggregate_throughput = fast_throughput + slow_throughput

    total_batches = scenario.total_batches_per_job

    # Time at which the faster job completes the full training run.
    fast_completion_times = total_batches * fast_batch_times

    # Progress of the slower job at that point.
    slow_batches_completed = fast_completion_times * slow_throughput
    slow_batches_completed = np.minimum(
        slow_batches_completed,
        total_batches,
    )

    # Prepared batches still waiting to be reused by the slower job.
    retained_batches = np.maximum(
        total_batches - slow_batches_completed,
        0.0,
    )

    # Decimal terabytes: 1 TB = 10^12 bytes.
    required_cache_tb = (
        retained_batches
        * scenario.cached_batch_size_bytes
        / 1e12
    )

    return {
        "fast_throughput": fast_throughput,
        "slow_throughput": slow_throughput,
        "aggregate_throughput": aggregate_throughput,
        "retained_batches": retained_batches,
        "required_cache_tb": required_cache_tb,
    }


# ==================================================================
# Plot style
# ==================================================================

def configure_plot_style():
    """Configure a compact publication-oriented plot style."""

    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 7.5,
        "axes.labelsize": 7.5,
        "axes.titlesize": 7.5,
        "axes.linewidth": 0.7,
        "legend.fontsize": 6.5,
        "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5,

        # Embed editable TrueType fonts.
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def format_axis(axis, stall_times):
    """Apply formatting shared by all panels."""

    x_min = stall_times.min()
    x_max = stall_times.max()
    x_padding = 0.03 * (x_max - x_min)

    # Small horizontal margin prevents endpoint markers from being clipped.
    axis.set_xlim(
        x_min - x_padding,
        x_max + x_padding,
    )

    axis.set_xticks(
        np.arange(0.0, 1.01, 0.2)
    )

    axis.set_ylim(bottom=0)

    # Horizontal grid lines keep the compact plot readable without clutter.
    axis.grid(
        axis="y",
        linestyle=":",
        linewidth=0.5,
        alpha=0.6,
    )

    axis.set_axisbelow(True)

    axis.tick_params(
        direction="out",
        length=2.5,
        width=0.7,
    )

    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)


# ==================================================================
# Plot generation
# ==================================================================

def create_two_panel_figure(
    scenario: Scenario,
    stall_times: np.ndarray,
    results: dict,
    output_directory: Path,
):
    """
    Generate one two-panel figure.

    Left:  per-job and aggregate training throughput
    Right: peak cache capacity required to preserve reuse
    """

    fig, axes = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(3.45, 1.75),
    )

    throughput_axis, cache_axis = axes

    # --------------------------------------------------------------
    # Left panel: training throughput
    # --------------------------------------------------------------

    throughput_axis.plot(
        stall_times,
        results["fast_throughput"],
        marker="o",
        markersize=2.8,
        linewidth=1.3,
        linestyle="-",
        color="#8c1515",
        label=scenario.fast_job_name,
        zorder=3,
    )

    throughput_axis.plot(
        stall_times,
        results["slow_throughput"],
        marker="s",
        markersize=2.8,
        linewidth=1.3,
        linestyle="-.",
        color="#006d77",
        label=scenario.slow_job_name,
        zorder=3,
    )

    throughput_axis.plot(
        stall_times,
        results["aggregate_throughput"],
        marker="^",
        markersize=2.8,
        linewidth=1.2,
        linestyle="--",
        color="#444444",
        label="Aggregate",
        zorder=3,
    )

    throughput_axis.set_xlabel(
        "Data stall (s/batch)"
    )

    throughput_axis.set_ylabel(
        "Throughput (batches/s)"
    )

    throughput_axis.yaxis.set_major_formatter(
        FuncFormatter(throughput_formatter)
    )

    throughput_axis.legend(
        frameon=True,
        loc="upper right",
        handlelength=1.5,
        handletextpad=0.4,
        labelspacing=0.2,
    )

    format_axis(
        throughput_axis,
        stall_times,
    )

    # --------------------------------------------------------------
    # Right panel: peak cache capacity
    # --------------------------------------------------------------

    cache_axis.plot(
        stall_times,
        results["required_cache_tb"],
        marker="D",
        markersize=2.8,
        linewidth=1.3,
        linestyle="-",
        color="#555555",
        zorder=3,
    )

    cache_axis.set_xlabel(
        "Data stall (s/batch)"
    )

    cache_axis.set_ylabel(
        "Peak cache (TB)"
    )

    cache_axis.yaxis.set_major_formatter(
        FuncFormatter(cache_tb_formatter)
    )

    format_axis(
        cache_axis,
        stall_times,
    )

    # --------------------------------------------------------------
    # Export
    # --------------------------------------------------------------

    fig.tight_layout(
        pad=0.25,
        w_pad=1.0,
    )

    output_path = (
        output_directory
        / scenario.output_filename
    )

    fig.savefig(
        output_path,
        bbox_inches="tight",
        pad_inches=0.02,
    )

    plt.close(fig)

    print(
        f"Saved: {output_path}"
    )


# ==================================================================
# Print important values
# ==================================================================

def print_endpoint_values(
    scenario: Scenario,
    stall_times: np.ndarray,
    results: dict,
):
    """Print results at zero and one second of data stall."""

    print(
        f"\n{scenario.name}"
    )

    print(
        f"  Training horizon: "
        f"{scenario.total_batches_per_job:,} batches per job"
    )

    for index in [0, len(stall_times) - 1]:
        stall_time = stall_times[index]

        fast_rate = results["fast_throughput"][index]
        slow_rate = results["slow_throughput"][index]
        aggregate_rate = results["aggregate_throughput"][index]

        retained_batches = results["retained_batches"][index]
        cache_tb = results["required_cache_tb"][index]

        print(
            f"\n  Data stall: {stall_time:.1f} s"
        )

        print(
            f"    {scenario.fast_job_name}: "
            f"{fast_rate:.2f} batches/s"
        )

        print(
            f"    {scenario.slow_job_name}: "
            f"{slow_rate:.2f} batches/s"
        )

        print(
            f"    Aggregate: "
            f"{aggregate_rate:.2f} batches/s"
        )

        print(
            f"    Retained batches: "
            f"{retained_batches:.0f}"
        )

        print(
            f"    Peak cache: "
            f"{cache_tb:.2f} TB"
        )


# ==================================================================
# Main
# ==================================================================

def main():
    configure_plot_style()

    # --------------------------------------------------------------
    # Shared simulation settings
    # --------------------------------------------------------------

    shared_epochs = 50

    stall_times = np.linspace(
        0.0,
        1.0,
        11,
    )

    output_directory = Path(
        "exp_results/motivation"
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------------
    # ImageNet scenario
    # --------------------------------------------------------------

    imagenet_training_samples = 1_281_167
    imagenet_batch_size = 128

    # Include the final partial mini-batch.
    imagenet_batches_per_epoch = math.ceil(
        imagenet_training_samples
        / imagenet_batch_size
    )

    imagenet_total_batches = (
        shared_epochs
        * imagenet_batches_per_epoch
    )

    imagenet = Scenario(
        name="ImageNet: ResNet-18 and ResNet-50",
        fast_job_name="ResNet-18",
        slow_job_name="ResNet-50",

        # Accelerator-side execution time per mini-batch.
        fast_execution_time=0.10,
        slow_execution_time=0.33,

        # One prepared mini-batch occupies 70 MiB.
        cached_batch_size_bytes=70.0 * 1024**2,

        # 50 epochs × 10,010 batches/epoch = 500,500 batches/job.
        total_batches_per_job=imagenet_total_batches,

        output_filename="fig_imagenet_motivation.pdf",
    )

    # --------------------------------------------------------------
    # CosmoFlow execution times
    # --------------------------------------------------------------

    # Reported computation + communication time per epoch:
    #
    # Coarse: 1.48 + 0.23 s
    # Dense:  9.31 + 0.44 s
    #
    # Training samples:          8,192
    # Global mini-batch size:      256
    # Mini-batches per epoch:       32

    cosmoflow_batches_per_epoch = 32

    coarse_execution_time = (
        (1.48 + 0.23)
        / cosmoflow_batches_per_epoch
    )

    dense_execution_time = (
        (9.31 + 0.44)
        / cosmoflow_batches_per_epoch
    )

    # --------------------------------------------------------------
    # CosmoFlow cache-size assumption
    # --------------------------------------------------------------

    # Shared cache stores the dense float32 representation before
    # resolution-specific processing.
    #
    # One global mini-batch:
    #   256 samples
    #   × 128 × 128 × 128 cells
    #   × 4 channels
    #   × 4 bytes per float32 value

    cosmoflow_batch_bytes = (
        256
        * (128**3)
        * 4
        * 4
    )

    cosmoflow_total_batches = (
        shared_epochs
        * cosmoflow_batches_per_epoch
    )

    cosmoflow = Scenario(
        name="CosmoFlow: coarse and dense",
        fast_job_name="Coarse",
        slow_job_name="Dense",
        fast_execution_time=coarse_execution_time,
        slow_execution_time=dense_execution_time,
        cached_batch_size_bytes=cosmoflow_batch_bytes,

        # 50 epochs × 32 batches/epoch = 1,600 batches/job.
        total_batches_per_job=cosmoflow_total_batches,

        output_filename="fig_cosmoflow_motivation.pdf",
    )

    # --------------------------------------------------------------
    # Run both scenarios
    # --------------------------------------------------------------

    for scenario in [imagenet, cosmoflow]:
        results = simulate_scenario(
            scenario=scenario,
            stall_times=stall_times,
        )

        print_endpoint_values(
            scenario=scenario,
            stall_times=stall_times,
            results=results,
        )

        create_two_panel_figure(
            scenario=scenario,
            stall_times=stall_times,
            results=results,
            output_directory=output_directory,
        )


if __name__ == "__main__":
    main()