#!/usr/bin/env python3
import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path
import sys
import hydra
from omegaconf import DictConfig
import os
from tabulate import tabulate


def load_metrics(filepath):
    """Load metrics from JSON file"""
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: File {filepath} not found")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON in {filepath}")
        sys.exit(1)


def create_pipeline_flow_diagram(metrics_data, output_path="pipeline_flow.png"):
    """Create a flow diagram showing task success/failure and metrics using matplotlib"""
    
    # Extract pipeline data using the aggregated domain
    pipelines = []
    for pipeline_name, pipeline_data in metrics_data.items():
        if 'domains' in pipeline_data and 'aggregated' in pipeline_data['domains']:
            agg = pipeline_data['domains']['aggregated']
            
            pipelines.append({
                'name': pipeline_name.replace('Pipeline_GPTAgent', '').replace('_', ' '),
                'total_tasks': agg.get('total_tasks', 0),
                'planning_success_rate': agg.get('planning_success_rate', 0),
                'grounding_success_rate': agg.get('grounding_success_rate', 0),
                'overall_success_rate': agg.get('overall_success_rate', agg.get('planning_success_rate', 0)),
                'avg_plan_length': agg.get('avg_plan_length_successful', 0),
                'pipeline_type': agg.get('pipeline_type', 'Unknown')
            })
    
    if not pipelines:
        print("No pipeline data found in metrics")
        return
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Pipeline Performance Analysis', fontsize=16, fontweight='bold')
    
    # Color mapping for different pipeline types
    color_map = {
        'SayPlan': '#ff6384',
        'ContextMatters': '#36a2eb', 
        'Delta': '#ffce56',
        'DeltaPlus': '#ff9f40',
        'LLMAsPlanner': '#9966ff',
        'Unknown': '#808080'
    }
    
    # 1. Success/Failure Bar Chart (top left)
    ax1 = axes[0, 0]
    pipeline_names = [p['name'] for p in pipelines]
    success_rates = [p['overall_success_rate'] for p in pipelines]
    failure_rates = [100 - p['overall_success_rate'] for p in pipelines]
    colors = [color_map.get(p['pipeline_type'], color_map['Unknown']) for p in pipelines]
    
    x = np.arange(len(pipeline_names))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, success_rates, width, label='Success Rate', color='lightgreen', alpha=0.8)
    bars2 = ax1.bar(x + width/2, failure_rates, width, label='Failure Rate', color='lightcoral', alpha=0.8)
    
    ax1.set_ylabel('Percentage (%)')
    ax1.set_title('Success vs Failure Rates by Pipeline')
    ax1.set_xticks(x)
    ax1.set_xticklabels(pipeline_names, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax1.annotate(f'{height:.1f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)
    
    # 2. Task Distribution Pie Chart (top right)
    ax2 = axes[0, 1]
    total_tasks = [p['total_tasks'] for p in pipelines]
    colors_pie = [color_map.get(p['pipeline_type'], color_map['Unknown']) for p in pipelines]
    
    wedges, texts, autotexts = ax2.pie(total_tasks, labels=pipeline_names, colors=colors_pie, 
                                      autopct='%1.1f%%', startangle=90)
    ax2.set_title('Task Distribution by Pipeline')
    
    # 3. Plan Length for Successful Tasks (bottom left)
    ax3 = axes[1, 0]
    plan_lengths = [p['avg_plan_length'] for p in pipelines]
    successful_tasks = [p['total_tasks'] * p['overall_success_rate'] / 100 for p in pipelines]
    
    bars3 = ax3.bar(pipeline_names, plan_lengths, color=colors, alpha=0.7)
    ax3.set_ylabel('Average Plan Length')
    ax3.set_title('Average Plan Length for Successful Tasks')
    ax3.set_xticklabels(pipeline_names, rotation=45, ha='right')
    ax3.grid(True, alpha=0.3)
    
    # Add value labels
    for i, (bar, tasks) in enumerate(zip(bars3, successful_tasks)):
        height = bar.get_height()
        ax3.annotate(f'{height:.1f}\n({tasks:.0f} tasks)',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)
    
    # 4. Flow Diagram (bottom right)
    ax4 = axes[1, 1]
    
    # Create a simplified flow visualization
    y_positions = np.linspace(0.1, 0.9, len(pipelines))
    
    for i, pipeline in enumerate(pipelines):
        y = y_positions[i]
        total = pipeline['total_tasks']
        success = int(total * pipeline['overall_success_rate'] / 100)
        failure = total - success
        
        # Pipeline name
        ax4.text(0.05, y, pipeline['name'], fontsize=10, va='center', weight='bold')
        
        # Total tasks box
        rect_total = mpatches.Rectangle((0.2, y-0.02), 0.15, 0.04, 
                                      facecolor=color_map.get(pipeline['pipeline_type'], color_map['Unknown']), 
                                      alpha=0.7)
        ax4.add_patch(rect_total)
        ax4.text(0.275, y, f'{total}', ha='center', va='center', fontsize=9, weight='bold')
        
        # Success box
        if success > 0:
            success_width = 0.1 * (success / max([p['total_tasks'] for p in pipelines]))
            rect_success = mpatches.Rectangle((0.4, y-0.015), success_width, 0.03, 
                                            facecolor='lightgreen', alpha=0.8)
            ax4.add_patch(rect_success)
            ax4.text(0.4 + success_width/2, y, f'{success}', ha='center', va='center', fontsize=8)
        
        # Failure box
        if failure > 0:
            failure_width = 0.1 * (failure / max([p['total_tasks'] for p in pipelines]))
            rect_failure = mpatches.Rectangle((0.6, y-0.015), failure_width, 0.03, 
                                            facecolor='lightcoral', alpha=0.8)
            ax4.add_patch(rect_failure)
            ax4.text(0.6 + failure_width/2, y, f'{failure}', ha='center', va='center', fontsize=8)
        
        # Arrows
        ax4.annotate('', xy=(0.39, y), xytext=(0.36, y),
                    arrowprops=dict(arrowstyle='->', lw=1.5, color='black'))
        if success > 0:
            ax4.annotate('', xy=(0.6 + 0.1 * (failure / max([p['total_tasks'] for p in pipelines])) + 0.02, y), 
                        xytext=(0.4 + 0.1 * (success / max([p['total_tasks'] for p in pipelines])) + 0.02, y),
                        arrowprops=dict(arrowstyle='->', lw=1.5, color='red'))
    
    # Add column headers
    ax4.text(0.275, 0.95, 'Total Tasks', ha='center', fontsize=12, weight='bold')
    ax4.text(0.45, 0.95, 'Success', ha='center', fontsize=12, weight='bold', color='green')
    ax4.text(0.65, 0.95, 'Failure', ha='center', fontsize=12, weight='bold', color='red')
    
    ax4.set_xlim(0, 0.8)
    ax4.set_ylim(0, 1)
    ax4.set_title('Task Flow: Total → Success/Failure')
    ax4.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Flow diagram saved to: {output_path}")
    
    return fig



def create_detailed_metrics_chart(metrics_data, output_path="detailed_metrics.png"):
    """Create detailed metrics visualization"""
    
    # Extract pipeline data using the aggregated domain
    pipelines = []
    for pipeline_name, pipeline_data in metrics_data.items():
        if 'domains' in pipeline_data and 'aggregated' in pipeline_data['domains']:
            agg = pipeline_data['domains']['aggregated']
            pipelines.append({
                'name': pipeline_name.replace('Pipeline_GPTAgent', '').replace('_', ' '),
                'total_tasks': agg.get('total_tasks', 0),
                'planning_success_rate': agg.get('planning_success_rate', 0),
                'grounding_success_rate': agg.get('grounding_success_rate', 0),
                'overall_success_rate': agg.get('overall_success_rate', agg.get('planning_success_rate', 0)),
                'avg_plan_length': agg.get('avg_plan_length_successful', 0),
                'pipeline_type': agg.get('pipeline_type', 'Unknown')
            })
    
    if not pipelines:
        print("No pipeline data found in metrics")
        return
    
    # Create comprehensive metrics chart
    fig, ax = plt.subplots(figsize=(14, 8))
    
    pipeline_names = [p['name'] for p in pipelines]
    x = np.arange(len(pipeline_names))
    width = 0.2
    
    # Different metrics to plot
    total_tasks = [p['total_tasks'] for p in pipelines]
    success_rates = [p['overall_success_rate'] for p in pipelines]
    plan_lengths = [p['avg_plan_length'] for p in pipelines]
    
    # Normalize plan lengths to 0-100 scale for comparison
    max_plan_length = max(plan_lengths) if plan_lengths else 1
    normalized_plan_lengths = [(p/max_plan_length)*100 for p in plan_lengths]
    
    # Create bars
    bars1 = ax.bar(x - width, total_tasks, width, label='Total Tasks', alpha=0.8, color='skyblue')
    bars2 = ax.bar(x, success_rates, width, label='Success Rate (%)', alpha=0.8, color='lightgreen')
    bars3 = ax.bar(x + width, normalized_plan_lengths, width, 
                   label=f'Plan Length (normalized, max={max_plan_length:.1f})', alpha=0.8, color='orange')
    
    ax.set_xlabel('Pipelines')
    ax.set_ylabel('Values')
    ax.set_title('Comprehensive Pipeline Metrics Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(pipeline_names, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bars, values in [(bars1, total_tasks), (bars2, success_rates), (bars3, plan_lengths)]:
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.annotate(f'{value:.1f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3), textcoords="offset points",
                       ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Detailed metrics chart saved to: {output_path}")
    
    return fig


def create_success_rates_chart(metrics_data, output_path="success_rates.png"):
    """Create grouped bar chart for planning and grounding success rates by pipeline"""
    pipelines = []
    for pipeline_name, pipeline_data in metrics_data.items():
        if 'domains' in pipeline_data and 'aggregated' in pipeline_data['domains']:
            agg = pipeline_data['domains']['aggregated']
            pipelines.append({
                'name': pipeline_name.replace('Pipeline_GPTAgent', '').replace('_', ' '),
                'planning_success_rate': agg.get('planning_success_rate', 0),
                'grounding_success_rate': agg.get('grounding_success_rate', 0),
                'total_tasks': agg.get('total_tasks', 0),
            })
    if not pipelines:
        print("No pipeline data found in metrics")
        return

    fig, ax = plt.subplots(figsize=(12, 6))
    names = [p['name'] for p in pipelines]
    x = np.arange(len(names))
    # Use a consistent group width and offsets to prevent overlap
    group_width = 0.75
    bar_width = group_width / 3.0  # three bars per group

    planning = [p['planning_success_rate'] for p in pipelines]
    grounding = [p['grounding_success_rate'] if p['grounding_success_rate'] is not None else 0 for p in pipelines]
    totals = [p['total_tasks'] for p in pipelines]

    # Left, center, right positions for the three bars
    bars1 = ax.bar(x - bar_width, planning, bar_width*0.9, label='Planning Success (%)', color='#4caf50', alpha=0.85)
    bars2 = ax.bar(x + 0.0, grounding, bar_width*0.9, label='Grounding Success (%)', color='#2196f3', alpha=0.85)

    # Secondary axis for total tasks counts
    ax_counts = ax.twinx()
    bars3 = ax_counts.bar(x + bar_width, totals, bar_width*0.9, label='Total Tasks (count)', color='#ff9800', alpha=0.7)

    ax.set_ylabel('Percentage (%)')
    ax.set_title('Planning vs Grounding Success by Pipeline')
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha='right')
    ax.set_ylim(0, 100)
    ax_counts.set_ylabel('Tasks (count)')
    # Build a combined legend from both axes
    handles1, labels1 = ax.get_legend_handles_labels()
    handles2, labels2 = ax_counts.get_legend_handles_labels()
    ax.legend(handles1 + handles2, labels1 + labels2, loc='upper left')
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for bars, values in ((bars1, planning), (bars2, grounding)):
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.annotate(f'{value:.1f}%',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=8)
    # Add count labels on the tasks bars (right axis)
    for bar, value in zip(bars3, totals):
        height = bar.get_height()
        ax_counts.annotate(f'{int(value)}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3), textcoords="offset points",
                           ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Success rates chart saved to: {output_path}")
    return fig


def print_summary_stats(metrics_data):
    """Print summary statistics from the metrics"""
    print("\n" + "="*60)
    print("PIPELINE PERFORMANCE SUMMARY")
    print("="*60)
    
    for pipeline_name, pipeline_data in metrics_data.items():
        if 'domains' in pipeline_data and 'aggregated' in pipeline_data['domains']:
            agg = pipeline_data['domains']['aggregated']
            print(f"\n{pipeline_name}:")
            print(f"  Total Tasks: {agg.get('total_tasks', 0)}")
            print(f"  Planning Success Rate: {agg.get('planning_success_rate', 0):.1f}%")
            if 'grounding_success_rate' in agg and agg['grounding_success_rate'] is not None:
                print(f"  Grounding Success Rate: {agg.get('grounding_success_rate', 0):.1f}%")
            if 'overall_success_rate' in agg and agg['overall_success_rate'] is not None:
                print(f"  Overall Success Rate: {agg.get('overall_success_rate', 0):.1f}%")
            print(f"  Pipeline Type: {agg.get('pipeline_type', 'Unknown')}")

def split_metrics(metrics):
    """Split metrics into gen-domain and no-gen-domain groups."""
    gen, no_gen = [], []
    for key, val in metrics.items():
        if "gendomain" in key.lower():
            gen.append(val)
        else:
            no_gen.append(val)
    return gen, no_gen


def extract_metrics(approaches):
    """Extract pipeline metrics based on pipeline type."""
    results = []
    for p in approaches:
        pname_lower = p["pipeline_name"].lower()
        pipeline_name = p["pipeline_name"].split("Pipeline")[0]
        domains = p["domains"]["aggregated"]

        if "contextmatters" in pname_lower:
            avg_time = domains["avg_inference_time"]
        elif any(x in pname_lower for x in ["delta", "sayplan", "llmasplanner"]):
            avg_time = domains["avg_plan_time"]
        else:
            continue

        results.append([
            pipeline_name,
            "ground" in pname_lower,
            domains["grounding_success_rate"],
            domains["planning_success_rate"],
            avg_time,
            domains["avg_plan_length_successful"]
        ])
    return results


def format_and_sort(metrics, domain_label):
    """Format metrics and return merged table rows."""
    formatted = [
        [
            method,
            "Gr" if gr_flag else "w/o Gr",
            f"{sr_gp:.2f}" if gr_flag else "-",
            f"{sr_p:.2f}",
            f"{avg_time:.2f}",
            f"{avg_len:.2f}"
        ]
        for method, gr_flag, sr_gp, sr_p, avg_time, avg_len in metrics
    ]

    order_map = {"Gr": 0, "w/o Gr": 1}
    sorted_results = sorted(
        formatted,
        key=lambda row: (row[0], order_map[row[1]])
    )

    merged = []
    last_method, first_block = None, True
    for row in sorted_results:
        method = row[0]
        if method == last_method:
            merged.append(["", "", *row[1:]])
        else:
            if first_block:
                merged.append([domain_label, *row])
                first_block = False
            else:
                merged.append(["", *row])
        last_method = method
    return merged


def generate_comparison_performance_tables(metrics_data):
    comparison_tabulate_headers = ["", "", "","SR (%)\nGrounding + Planning", "SR (%)\nPlanning Only", "Avg.\nPlanning Time (s)", "Avg.\nPlan Length"]
    performance_tabulate_headers = ["", "SR (%)", "Plan \nLength", "Planning \nTime", "Expanded \nNodes", "Inference \nTime (s)"]
    pipeline_name = str("ContextMattersPipeline_GPTAgent_gendomain_ground")
    pipeline_data = metrics_data[pipeline_name]["domains"]

    gendomain_approaches, no_gendomain_approaches = split_metrics(metrics_data)

    gen_domain_metrics = extract_metrics(gendomain_approaches)
    no_gen_domain_metrics = extract_metrics(no_gendomain_approaches)

    gen_domain_results = format_and_sort(gen_domain_metrics, "Domain Gen")
    no_gen_domain_results = format_and_sort(no_gen_domain_metrics, "W/O Domain Gen")

    final_results = gen_domain_results + no_gen_domain_results

    performance_metrics = []

    for key, value in pipeline_data.items():
        if key == "aggregated": continue
        performance_metrics.append([key,
                                    value["overall_success_rate"],
                                    value["avg_plan_length_successful"],
                                    value["avg_inference_time"],
                                    value["avg_no_nodes"],
                                    value["avg_total_llm_time"],
                                    ])
    print(" ")
    print(" ")    
    print("Comparison of baselines and our approach, w/ and w/o domain generation and gr(ounding) in the environment: ")
    print(tabulate(final_results, headers = comparison_tabulate_headers, tablefmt = "fancy_grid"))

    print(" ")
    print(" ")
    print("Average performance on the General, House Cleaning, PC Assembly, Office Setup, Laundry, and Dining Setup tasks: ")
    print(tabulate(performance_metrics, headers = performance_tabulate_headers, tablefmt = "fancy_grid"))



@hydra.main(config_path = "config", config_name = "config", version_base = None)
def main(cfg: DictConfig):
    detailed_metrics_path = Path(os.path.join(cfg.metrics_res_path, "detailed_metrics.json"))
    
    # Load metrics data
    metrics_data = load_metrics(detailed_metrics_path)

    print_summary_stats(metrics_data)
    
    # Create output directory if it doesn't exist
    output_dir = Path(cfg.metrics_res_path)
    
    try:
        # Pipeline flow diagram
        flow_output = output_dir / "pipeline_flow_diagram.png"
        create_pipeline_flow_diagram(metrics_data, str(flow_output))
        
        
        # Detailed metrics chart
        metrics_output = output_dir / "detailed_metrics_chart.png"
        create_detailed_metrics_chart(metrics_data, str(metrics_output))

        # Planning vs Grounding success rates chart
        success_rates_output = output_dir / "success_rates_chart.png"
        create_success_rates_chart(metrics_data, str(success_rates_output))
        
        print(f"\nAll diagrams saved to: {output_dir}")
        print("Generated files:")
        print(f"  - {flow_output}")
        print(f"  - {metrics_output}")
        print(f"  - {success_rates_output}")

        generate_comparison_performance_tables(metrics_data)
        
    except Exception as e:
        print(f"Error generating diagrams: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()