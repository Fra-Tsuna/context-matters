#!/usr/bin/env python3
"""
Comprehensive metrics calculator for different pipeline results.
This script processes CSV files from multiple pipeline types and calculates
key performance metrics for comparison and plotting.
"""

import os
import pandas as pd
import json
from typing import Dict, List, Any
from pathlib import Path
import numpy as np
from collections import defaultdict
from omegaconf import DictConfig
import hydra

class PipelineMetricsCalculator:
    """Calculator for metrics across different pipeline types."""
    
    def __init__(self, results_dir: str):
        self.results_dir = Path(results_dir)
        self.pipeline_results = {}
        self.summary_metrics = {}
        
    def detect_pipelines(self) -> List[str]:
        """Detect all pipeline directories in the results folder."""
        pipelines = []
        if not self.results_dir.exists():
            print(f"Results directory {self.results_dir} does not exist!")
            return pipelines
            
        for item in self.results_dir.iterdir():
            if item.is_dir():
                pipelines.append(item.name)
        
        print(f"Found {len(pipelines)} pipelines: {pipelines}")
        return pipelines
    
    def find_csv_files(self, pipeline_dir: Path) -> List[Path]:
        """Find all CSV files in a pipeline directory, handling different structures."""
        csv_files = []
        
        # Check for direct CSV files
        for csv_file in pipeline_dir.glob("*.csv"):
            csv_files.append(csv_file)
        
        # Check for subdirectories (like goodresults in SayPlan)
        for subdir in pipeline_dir.iterdir():
            if subdir.is_dir():
                for csv_file in subdir.glob("*.csv"):
                    csv_files.append(csv_file)
        
        return csv_files
    
    def determine_pipeline_type(self, pipeline_name: str, csv_columns: List[str]) -> str:
        """Determine the pipeline type based on name and CSV columns."""
        pipeline_name_lower = pipeline_name.lower()
        print(pipeline_name_lower)
        if 'contextmatters' in pipeline_name_lower:
            return 'ContextMatters'
        elif 'deltaplus' in pipeline_name_lower:
            return 'DeltaPlus'
        elif 'delta' in pipeline_name_lower:
            return 'Delta'
        elif 'sayplan' in pipeline_name_lower:
            return 'SayPlan'
        elif 'llmasplanner' in pipeline_name_lower:
            return 'LLMAsPlanner'
        else:
            raise ValueError(f"Unknown pipeline type for {pipeline_name}")

    
    def safe_numeric_mean(self, series, default=0):
        """Safely calculate mean of a series, handling non-numeric values."""
        try:
            # Convert to numeric, errors='coerce' will turn invalid values to NaN
            numeric_series = pd.to_numeric(series, errors='coerce')
            return numeric_series.mean() if not numeric_series.isna().all() else default
        except Exception:
            return default
        
    def get_total_refinements(self, ref_string):
        try:
            return sum(int(p.strip()) for p in str(ref_string).split(';'))
        except (ValueError, TypeError):
            return 0
        
    def parse_refinement_string_to_list(self, ref_str):
        if pd.isna(ref_str) or ref_str == '': return []
        try: return [int(p.strip()) for p in str(ref_str).split(';')]
        except (ValueError, TypeError): return []

    def calculate_context_matters_metrics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate metrics for Context Matters pipeline."""
        metrics = {}
        
        print(df)

        # Convert boolean columns
        df['Planning Successful'] = df['Planning Successful'].astype(str).str.strip().str.lower() == 'true'
        df['Grounding Successful'] = df['Grounding Successful'].astype(str).str.strip().str.lower() == 'true'
        df['total_refinements'] = df['Refinements per iteration'].apply(self.get_total_refinements)
        df['Relaxations'] = pd.to_numeric(df['Relaxations'], errors = 'coerce').fillna(0)
        
        total_tasks = len(df)
        # Success rates
        planning_success_rate = (df['Planning Successful'].sum() / total_tasks) * 100
        grounding_success_rate = (df['Grounding Successful'].sum() / total_tasks) * 100
        
        # Calculate overall success rate with division by zero protection
        planning_successful_count = df['Planning Successful'].sum()
        if planning_successful_count > 0:
            overall_success_rate = (df[df['Planning Successful']]['Grounding Successful']).sum() / planning_successful_count * 100
        else:
            overall_success_rate = 0.0

        avg_plan_length_successful = self.safe_numeric_mean(df[df['Planning Successful']]['Plan Length'])
        avg_no_relaxations = self.safe_numeric_mean(df['Relaxations'])
        avg_no_refinements = self.safe_numeric_mean(df['Refinements per iteration'])
        avg_no_nodes = self.safe_numeric_mean(df[df['Planning Successful']]['num_node_expansions'])
        avg_inference_time = self.safe_numeric_mean(df['total_inference_time'])
        avg_total_llm_time = self.safe_numeric_mean(df['Total LLM Time'])

        total_refinements_sum = df['total_refinements'].sum()
        total_relaxations_sum = df['Relaxations'].sum()

        if total_relaxations_sum > 0:
            avg_refinements_per_relaxation = total_refinements_sum / total_relaxations_sum
        else:
            avg_refinements_per_relaxation = 0.0

        df['refinement_list'] = df['Refinements per iteration'].apply(self.parse_refinement_string_to_list)

        
        metrics = {
            'total_tasks': total_tasks,
            'planning_success_rate': round(planning_success_rate, 2),
            'grounding_success_rate': round(grounding_success_rate, 2),
            'overall_success_rate': round(overall_success_rate, 2),
            'avg_plan_length_successful': round(avg_plan_length_successful, 2),
            'avg_no_relaxations': round(avg_no_relaxations, 2),
            'avg_no_refinements': round(avg_no_refinements, 2),
            'avg_refinements_per_relaxation': round(avg_refinements_per_relaxation, 2),
            'avg_no_nodes': round(avg_no_nodes, 2),
            'avg_inference_time': round(avg_inference_time, 2),
            "avg_total_llm_time": round(avg_total_llm_time, 2),
        }
        
        return metrics
    
    def calculate_delta_metrics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate metrics for Delta pipeline."""
        metrics = {}
    
        # Convert boolean columns
        df['Planning Successful'] = df['Planning Successful'].astype(str).str.strip().str.lower() == 'true'
        df['Grounding Successful'] = df['Grounding Successful'].astype(str).str.strip().str.lower() == 'true'
        
        total_tasks = len(df)
        # Success rates
        planning_success_rate = (df['Planning Successful'].sum() / total_tasks) * 100
        grounding_success_rate = (df['Grounding Successful'].sum() / total_tasks) * 100
        
        # Calculate overall success rate with division by zero protection
        planning_successful_count = df['Planning Successful'].sum()
        if planning_successful_count > 0:
            overall_success_rate = (df[df['Planning Successful']]['Grounding Successful']).sum() / planning_successful_count * 100
        else:
            overall_success_rate = 0.0

        avg_plan_length_successful = self.safe_numeric_mean(df[df['Planning Successful']]['Plan Length'])
        avg_no_subgoals = self.safe_numeric_mean(df[df['Planning Successful']]['Number of subgoals'])
        avg_planning_costs = self.safe_numeric_mean(df[df['Planning Successful']]['Total Subgoal Planning Costs'])
        avg_plan_time = self.safe_numeric_mean(df[df['Planning Successful']]['Total Subgoal Planning Time'])
        avg_total_time = self.safe_numeric_mean(df['Total LLM Time'])
        
        
        metrics = {
            'total_tasks': total_tasks,
            'planning_success_rate': round(planning_success_rate, 2),
            'grounding_success_rate': round(grounding_success_rate, 2),
            'overall_success_rate': round(overall_success_rate, 2),
            'avg_plan_length_successful': round(avg_plan_length_successful, 2),
            'avg_plan_time': round(avg_plan_time, 2),
            'avg_no_subgoals': round(avg_no_subgoals, 2),
            'avg_planning_costs': round(avg_planning_costs, 2),
            'avg_total_time': round(avg_total_time, 2)
            
        }
        
        return metrics
    
    def calculate_llm_as_planner_metrics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate metrics for LLMasplanner pipeline."""
        metrics = {}
        
        # Convert boolean columns
        df['Planning Successful'] = df['Planning Successful'].astype(str).str.strip().str.lower() == 'true'
        total_tasks = len(df)
        
        # Success rates
        planning_success_rate = (df['Planning Successful'].sum() / total_tasks) * 100
        
        # Plan lengths(computed if plan was successful, if the plan failed, the plan length will be not considered)
        avg_plan_length_successful = self.safe_numeric_mean(df[df['Planning Successful']]['Plan Length'])
        avg_time = self.safe_numeric_mean(df['Total LLM Time'])
        
        metrics = {
            'total_tasks': total_tasks,
            'planning_success_rate': round(planning_success_rate, 2),
            'grounding_success_rate': 0,  # SayPlan doesn't have grounding
            'avg_plan_length_successful': round(avg_plan_length_successful, 2),
            'avg_plan_time': round(avg_time, 2)
        }
        
        return metrics
    
    def calculate_sayplan_metrics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate metrics for SayPlan pipeline."""
        metrics = {}
        
        # Convert boolean columns
        df['Planning Successful'] = df['Planning Successful'].astype(str).str.strip().str.lower() == 'true'
        total_tasks = len(df)
        
        # Success rates
        planning_success_rate = (df['Planning Successful'].sum() / total_tasks) * 100
        
        # Plan lengths(computed if plan was successful, if the plan failed, the plan length will be not considered)
        avg_plan_length_successful = self.safe_numeric_mean(df[df['Planning Successful']]['Plan Length'])
        avg_replan_count = self.safe_numeric_mean(df[df['Planning Successful']]['Replan Count'])
        avg_search_time = self.safe_numeric_mean(df[df['Planning Successful']]['Search Time'])
        avg_plan_time = self.safe_numeric_mean(df[df['Planning Successful']]['Plan Time'])
        
        metrics = {
            'total_tasks': total_tasks,
            'planning_success_rate': round(planning_success_rate, 2),
            'grounding_success_rate': 0,  # SayPlan doesn't have grounding
            'avg_plan_length_successful': round(avg_plan_length_successful, 2),
            'avg_replan_count': round(avg_replan_count, 2),
            'avg_search_time': round(avg_search_time, 2),
            'avg_plan_time': round(avg_plan_time, 2)
        }
        
        return metrics
    
    def find_value_anywhere(self, data, target_key, default_value=None):
        def _search(obj):
            if isinstance(obj, dict):
                if target_key in obj:
                    return obj[target_key]
                
                for value in obj.values():
                    result = _search(value)
                    if result is not None:
                        return result
            
            elif isinstance(obj, list):
                for item in obj:
                    result = _search(item)
                    if result is not None:
                        return result
            
            return None
        
        result = _search(data)
        return result if result is not None else default_value

    
    def process_pipeline(self, pipeline_name: str) -> Dict[str, Any]:
        """Process a single pipeline and calculate its metrics."""
        pipeline_dir = self.results_dir / pipeline_name
        csv_files = self.find_csv_files(pipeline_dir)
        
        if not csv_files:
            print(f"No CSV files found in {pipeline_name}")
            return {}
        
        pipeline_metrics = {
            'pipeline_name': pipeline_name,
            'domains': {},
        }
        
        all_data = []
        
        for csv_file in csv_files:
            folder_path = (str(csv_file).replace(".csv", ""))
            
            df = pd.read_csv(csv_file, sep='|')
            if df.empty:
                print(f"Empty CSV file: {csv_file}")
                continue
            
            domain_name = csv_file.stem
            pipeline_type = self.determine_pipeline_type(pipeline_name, df.columns.tolist())
            
            ##IF CM ADD THE NUM_NODE_EXPANSION AND TIME METRICS TO DF 
            if pipeline_type == 'ContextMatters':
                for row in df.iterrows():
                    json_statistics_path = os.path.join(folder_path, f"{row[1]['Scene']}/{row[1]['Problem']}/statistics.json")
                    #OPEN JSON FILE
                    if os.path.exists(json_statistics_path):
                        print(f"Opening {json_statistics_path}")
                        with open(json_statistics_path, 'r') as f:
                            data = json.load(f)
                            # extract nested statistics under 'statistics' -> '0'
                            stats_root = data.get('statistics', {}).get('0', {})

                            total_node_expansion = self.find_value_anywhere(data, 'num_node_expansions', 0)
                            print(total_node_expansion)
                            total_inference_time = stats_root.get("TOTAL_INFERENCE", {}).get("total_inference_time", 0)
                            print("node", total_node_expansion)
                            print("time", total_inference_time)
                            #ADD TO DF
                            df.loc[row[0], 'num_node_expansions'] = total_node_expansion
                            df.loc[row[0], 'total_inference_time'] = total_inference_time
                
                            
            # Calculate domain-specific metrics
            if pipeline_type == 'ContextMatters':
                domain_metrics = self.calculate_context_matters_metrics(df)
            elif pipeline_type == 'Delta':
                domain_metrics = self.calculate_delta_metrics(df)
            elif pipeline_type == 'DeltaPlus':
                domain_metrics = self.calculate_delta_metrics(df)
            elif pipeline_type == 'SayPlan':
                domain_metrics = self.calculate_sayplan_metrics(df)
            elif pipeline_type == 'LLMAsPlanner':
                domain_metrics = self.calculate_llm_as_planner_metrics(df)
            else:
                print(f"Unknown pipeline type for {pipeline_name}")
                continue
            
            domain_metrics['pipeline_type'] = pipeline_type
            pipeline_metrics['domains'][domain_name] = domain_metrics
            
            # Add to aggregated data
            all_data.append(df)
            
        
        return pipeline_metrics
    
    def calculate_all_metrics(self):
        """Calculate metrics for all detected pipelines."""
        pipelines = self.detect_pipelines()
        
        for pipeline in pipelines:
            print(f"\n{'='*50}")
            print(f"Processing pipeline: {pipeline}")
            print(f"{'='*50}")
            
            metrics = self.process_pipeline(pipeline)
            if metrics:
                self.pipeline_results[pipeline] = metrics
    
    def compute_aggregated_metrics(self):
        """Compute aggregated metrics for each pipeline across all its domains."""
        # For each pipeline, calculate aggregated metrics across all its domains
        for pipeline_name, pipeline_data in self.pipeline_results.items():
            domains = pipeline_data.get('domains', {})
            if not domains:
                continue
            
            # Collect all metrics from all domains in this pipeline
            pipeline_metrics = defaultdict(list)
            pipeline_type = None
            
            for domain_name, domain_metrics in domains.items():
                # Get pipeline type (should be the same for all domains in a pipeline)
                if pipeline_type is None:
                    pipeline_type = domain_metrics.get('pipeline_type')
                
                for metric_name, metric_value in domain_metrics.items():
                    if metric_name != 'pipeline_type' and isinstance(metric_value, (int, float)):
                        pipeline_metrics[metric_name].append(metric_value)
            
            # Calculate aggregated values
            aggregated_domain = {}
            for metric_name, values in pipeline_metrics.items():
                if values:  # Only if we have values
                    if metric_name == 'total_tasks':
                        # For total_tasks, sum instead of average
                        aggregated_domain[metric_name] = sum(values)
                    else:
                        # For other metrics, calculate average
                        aggregated_domain[metric_name] = round(np.mean(values), 2)
            
            # Add pipeline type
            if pipeline_type:
                aggregated_domain['pipeline_type'] = pipeline_type
            
            # Add the aggregated domain to the pipeline
            self.pipeline_results[pipeline_name]['domains']['aggregated'] = aggregated_domain
    
    def save_results(self, output_dir: str = None):
        """Save all results to files."""
        if output_dir is None:
            output_dir = self.results_dir / "metrics_output"
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Save detailed results as JSON
        json_path = output_path / "detailed_metrics.json"
        with open(json_path, 'w') as f:
            json.dump(self.pipeline_results, f, indent=2)

  
@hydra.main(config_path = "config", config_name = "config", version_base = None)
def main(cfg: DictConfig):
    results_dir = cfg.res_path
    metrics_results_dir = cfg.metrics_res_path

    print("Results path:", results_dir)
    
    # Initialize calculator
    calculator = PipelineMetricsCalculator(results_dir)
    
    # Calculate metrics for all pipelines
    calculator.calculate_all_metrics()
    
    # Compute aggregated metrics
    calculator.compute_aggregated_metrics()
    
    calculator.save_results(metrics_results_dir)
    

if __name__ == "__main__":
    main()