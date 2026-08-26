# plot_motivation.py

from dataclasses import dataclass
from pathlib import Path
import math

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter


# ---------------------------------------------------------------------
# Scenario configuration
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class Scenario:
    name: str
    fast_job_name: str
    slow_job_name: str

    # Accelerator execution time per mini-batch.
    fast_execution_time: float
    slow_execution_time: float

    # Size of one prepared mini-batch stored in the cache.
    cached_batch_size_bytes: float

    # Number of mini-batches processed by each job.
    total_batches_per_job: int

    output_filename: str


# ---------------------------------------------------------------------
# Plot style
# ---------------------------------------------------------------------

LINE_STYLE = {
    "fast": {
        "color": "#006d77", #006d77, 4C78A8
        "marker": "o",
        "linestyle": "-",
    },
    "slow": {
        "color": "#8c1515", #8c1515, B55D60
        "marker": "s",
        "linestyle": "--",
    },
    "aggregate": {
        "color": "#3F3F3F", #444444, 3F3F3F
        "marker": "^",
        "linestyle": "-.",
    },
}

CACHE_COLOR = "#3F3F3F" #3F3F3F, 5A5A5A


def configure_plot_style():
    plt.rcParams.update({
        "font.size": 9,
        "font.family": "serif",
        "font.serif": [
            "Times New Roman",
            "Times",
            "Nimbus Roman",
            "DejaVu Serif",
        ],
        "axes.labelsize": 10.5,
        "axes.linewidth": 0.9,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "legend.fontsize": 7.8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


# ---------------------------------------------------------------------
# Tick formatters
# ---------------------------------------------------------------------

def throughput_formatter(value, _):
    if abs(value - round(value)) < 1e-6:
        return f"{value:.0f}"
    return f"{value:.1f}"


def cache_tb_formatter(value, _):
    if value >= 100:
        return f"{value:.0f}"
    if value >= 1:
        return f"{value:.1f}"
    return f"{value:.2f}"


# ---------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------

def simulate_scenario(scenario, stall_times):
    fast_batch_times = scenario.fast_execution_time + stall_times
    slow_batch_times = scenario.slow_execution_time + stall_times

    fast_throughput = 1.0 / fast_batch_times
    slow_throughput = 1.0 / slow_batch_times
    aggregate_throughput = fast_throughput + slow_throughput

    total_batches = scenario.total_batches_per_job

    # Time at which the faster job completes its training run.
    fast_completion_times = total_batches * fast_batch_times

    # Progress of the slower job at that point.
    slow_batches_completed = fast_completion_times * slow_throughput
    slow_batches_completed = np.minimum(slow_batches_completed, total_batches)

    # Prepared batches waiting to be reused by the slower job.
    retained_batches = np.maximum(
        total_batches - slow_batches_completed,
        0.0,
    )

    # Decimal TB: 1 TB = 10^12 bytes.
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


# ---------------------------------------------------------------------
# Axis formatting
# ---------------------------------------------------------------------

def style_axis(ax, stall_times):
    x_min = stall_times.min()
    x_max = stall_times.max()
    padding = 0.03 * (x_max - x_min)

    ax.set_xlim(x_min - padding, x_max + padding)
    ax.set_xticks(np.arange(0.0, 1.01, 0.2))
    ax.set_ylim(bottom=0)

    ax.grid(
        axis="y",
        linestyle="--",
        linewidth=0.45,
        alpha=0.45,
        zorder=0,
    )

    ax.set_axisbelow(True)

    ax.tick_params(
        direction="out",
        length=2.5,
        width=0.8,
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# ---------------------------------------------------------------------
# Plot generation
# ---------------------------------------------------------------------

def create_two_panel_figure(scenario, stall_times, results, output_directory):
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(3.45, 2.00),
    )

    fig.subplots_adjust(
        left=0.14,
        right=0.99,
        bottom=0.23,
        top=0.97,
        wspace=0.42,
    )

    throughput_ax, cache_ax = axes

    # -----------------------------------------------------------------
    # Left: training throughput
    # -----------------------------------------------------------------

    fast_style = LINE_STYLE["fast"]
    slow_style = LINE_STYLE["slow"]
    aggregate_style = LINE_STYLE["aggregate"]

    throughput_ax.plot(
        stall_times,
        results["fast_throughput"],
        color=fast_style["color"],
        marker=fast_style["marker"],
        linestyle=fast_style["linestyle"],
        linewidth=1.35,
        markersize=3.0,
        label=scenario.fast_job_name,
        zorder=3,
    )

    throughput_ax.plot(
        stall_times,
        results["slow_throughput"],
        color=slow_style["color"],
        marker=slow_style["marker"],
        linestyle=slow_style["linestyle"],
        linewidth=1.35,
        markersize=3.0,
        label=scenario.slow_job_name,
        zorder=3,
    )

    throughput_ax.plot(
        stall_times,
        results["aggregate_throughput"],
        color=aggregate_style["color"],
        marker=aggregate_style["marker"],
        linestyle=aggregate_style["linestyle"],
        linewidth=1.5,
        markersize=3.1,
        label="Aggregate",
        zorder=4,
    )

    throughput_ax.set_xlabel("Data stall (s/batch)", labelpad=3)
    throughput_ax.set_ylabel("Throughput (batches/s)", labelpad=3)

    throughput_ax.yaxis.set_major_formatter(
        FuncFormatter(throughput_formatter)
    )

    throughput_ax.legend(
        loc="upper right",
        frameon=True,
        handlelength=1.6,
        handletextpad=0.35,
        labelspacing=0.2,
        borderaxespad=0.25,
    )

    style_axis(throughput_ax, stall_times)

    # -----------------------------------------------------------------
    # Right: peak cache requirement
    # -----------------------------------------------------------------

    cache_ax.plot(
        stall_times,
        results["required_cache_tb"],
        color=CACHE_COLOR,
        marker="D",
        linestyle="-",
        linewidth=1.4,
        markersize=3.0,
        zorder=3,
    )

    cache_ax.set_xlabel("Data stall (s/batch)", labelpad=3)
    cache_ax.set_ylabel("Peak cache (TB)", labelpad=3)

    cache_ax.yaxis.set_major_formatter(
        FuncFormatter(cache_tb_formatter)
    )

    style_axis(cache_ax, stall_times)

    # -----------------------------------------------------------------
    # Save
    # -----------------------------------------------------------------

    pdf_path = output_directory / scenario.output_filename
    png_path = pdf_path.with_suffix(".png")

    save_kwargs = {
        "bbox_inches": "tight",
        "pad_inches": 0.02,
    }

    fig.savefig(pdf_path, **save_kwargs)
    fig.savefig(png_path, dpi=300, **save_kwargs)
    plt.close(fig)

    print(f"Saved: {pdf_path}")
    print(f"Saved: {png_path}")


# ---------------------------------------------------------------------
# Print useful endpoint values
# ---------------------------------------------------------------------

def print_endpoint_values(scenario, stall_times, results):
    print(f"\n{scenario.name}")
    print(
        f"  Training horizon: "
        f"{scenario.total_batches_per_job:,} batches per job"
    )

    for index in [0, len(stall_times) - 1]:
        stall = stall_times[index]

        print(f"\n  Data stall: {stall:.1f} s")
        print(
            f"    {scenario.fast_job_name}: "
            f"{results['fast_throughput'][index]:.2f} batches/s"
        )
        print(
            f"    {scenario.slow_job_name}: "
            f"{results['slow_throughput'][index]:.2f} batches/s"
        )
        print(
            f"    Aggregate: "
            f"{results['aggregate_throughput'][index]:.2f} batches/s"
        )
        print(
            f"    Retained batches: "
            f"{results['retained_batches'][index]:.0f}"
        )
        print(
            f"    Peak cache: "
            f"{results['required_cache_tb'][index]:.2f} TB"
        )


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    configure_plot_style()

    shared_epochs = 50
    stall_times = np.linspace(0.0, 1.0, 11)

    output_directory = Path("exp_results/motivation")
    output_directory.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------
    # ImageNet
    # -----------------------------------------------------------------

    imagenet_training_samples = 1_281_167
    imagenet_batch_size = 128

    imagenet_batches_per_epoch = math.ceil(
        imagenet_training_samples / imagenet_batch_size
    )

    imagenet_total_batches = (
        shared_epochs * imagenet_batches_per_epoch
    )

    imagenet = Scenario(
        name="ImageNet: ResNet-18 and ResNet-50",
        fast_job_name="ResNet-18",
        slow_job_name="ResNet-50",
        fast_execution_time=0.10,
        slow_execution_time=0.33,
        cached_batch_size_bytes=70.0 * 1024**2,
        total_batches_per_job=imagenet_total_batches,
        output_filename="fig1_imagenet.pdf",
    )

    # -----------------------------------------------------------------
    # CosmoFlow
    # -----------------------------------------------------------------

    cosmoflow_batches_per_epoch = 32

    coarse_execution_time = (
        (1.48 + 0.23)
        / cosmoflow_batches_per_epoch
    )

    dense_execution_time = (
        (9.31 + 0.44)
        / cosmoflow_batches_per_epoch
    )

    # One prepared global mini-batch:
    # 256 samples × 128^3 cells × 4 channels × 4 bytes.
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
        total_batches_per_job=cosmoflow_total_batches,
        output_filename="fig1_cosmoflow.pdf",
    )

    # -----------------------------------------------------------------
    # Run
    # -----------------------------------------------------------------

    for scenario in [imagenet, cosmoflow]:
        results = simulate_scenario(scenario, stall_times)

        print_endpoint_values(
            scenario,
            stall_times,
            results,
        )

        create_two_panel_figure(
            scenario,
            stall_times,
            results,
            output_directory,
        )


if __name__ == "__main__":
    main()