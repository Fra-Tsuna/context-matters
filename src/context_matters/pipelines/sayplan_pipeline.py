# Implementation of the SayPlan Pipeline due to absence of released code.
# Original paper: K. Rana et al., CoRL 2023, "SayPlan: Grounding Large Language Models using 3D Scene Graphs for Scalable Robot Task Planning"
# Paper link: https://proceedings.mlr.press/v229/rana23a/rana23a.pdf
# Based on the code re-implementation by Y. Liu et al. https://github.com/boschresearch/DELTA/blob/main/baselines/sayplan.py 


import csv
import os
import time
from pathlib import Path
from .base_pipeline import BasePipeline

from src.context_matters.prompt.sayplan_prompts import (
    search_prompt,
    plan_prompt,
    export_search_cmd,
    get_situational_actions,
)
from src.context_matters.utils.graph import (
    read_graph_from_path,
    get_verbose_scene_graph,
    collapse_scene_graph,
    update_scene_graph,
)
from src.context_matters.utils.log import (
    dump_sayplan_plan
)

from src.context_matters.logger_cfg import logger


class SayPlanPipeline(BasePipeline):
    def __init__(self, 
                max_debug_attempts = 4,
                max_search_attempts = 50,
                episodes = 5,
                temperature = 0.0,
                top_p = 1.0,
                 **base_init_kwargs
                ):
        super().__init__(**base_init_kwargs)
        self.max_debug_attempts = max_debug_attempts
        self.max_search_attempts = max_search_attempts
        self.episodes = episodes
        self.temperature = temperature
        self.top_p = top_p
        self.experiment_name: str = super()._construct_experiment_name()

    def _initialize_csv(self, csv_filepath):
        # Initialize CSV with headers
        header = ["Task", "Scene", "Problem",
                "Episode", "Exit Code", "Planning Successful",
                "Replan Count", "Search Time", "Plan Time",
                "Plan Length", "GT Cost"]
        
        with open(csv_filepath, mode="w", newline='') as f:
            writer = csv.writer(f, delimiter='|')
            writer.writerow(header)
    
    
    def _run_and_log_pipeline(self, task_name, scene_name, problem_id, results_problem_dir,
                            domain_file_path, domain_description, scene_graph_file_path,
                            csv_filepath):
        '''
        Runs the SayPlan pipeline for a specific task and logs the results.
            - task_name: Name of the task (e.g., "dining_setup").
            - scene_name: Name of the scene (e.g., "Allensville").
            - problem_id: Identifier for the problem instance. (e.g. "problem_1")
            - results_problem_dir: Directory where results will be saved.
            - domain_file_path: File path to the domain file (PDDL).
            - domain_description: contains "actions", "objects"
            - scene_graph_file_path: Path to the scene graph file (.npz).
            - csv_filepath: Path to the CSV file where results will be logged.
        '''
        
        scene_graph = read_graph_from_path(Path(scene_graph_file_path))
        extracted_scene_graph = get_verbose_scene_graph(scene_graph, as_string=False)
        extracted_scene_graph_str = get_verbose_scene_graph(scene_graph, as_string=True)
        current_goal = open(os.path.join(results_problem_dir, "task.txt"), "r").read()
        initial_robot_location = open(os.path.join(results_problem_dir, "init_loc.txt"), "r").read()
        query_additional_actions = get_situational_actions(task_name)
        query_sg = extracted_scene_graph
        query_goal = current_goal

        num_successes = 0
        for e in range(self.episodes):
            self.agent.reset()
            stats = {
                "exit_code": 0,
                "search_time": 0.0,
                "plan_time": 0.0,
                "plan_cost": 0,
                "replan_count": -1,
                "search_count": 0,
                "search_complete": False,
                "max_search_reached": False,
                "success": False,
            }
            
            query_sg, stats = self.scene_graph_search(query_sg, query_goal, initial_robot_location, stats)
            stats = self.iterative_replanning(query_sg, query_goal, query_additional_actions, results_problem_dir, stats)
            
            num_successes += int(stats["success"])
            
            logger.info(f"-------------------- Episode {e + 1}/{self.episodes}, \
                    Total Success: {num_successes},\
                    Replan: {stats['replan_count']} --------------"
                )
            
            with open(csv_filepath, mode="a", newline='') as f:
                writer = csv.writer(f, delimiter='|')
                writer.writerow([
                    task_name, scene_name, problem_id, 
                    e, stats["exit_code"], stats["success"],
                    stats["replan_count"], stats["search_time"], stats["plan_time"],
                    stats["plan_cost"], None
                ])

        return


    def scene_graph_search(self, query_sg, query_goal, initial_robot_location, stats):

        memory = set([])
        
        query_collapsed_sg = collapse_scene_graph(query_sg)
        content_s, prompt_s = search_prompt(
            query_goal, query_collapsed_sg, initial_robot_location
        )
        self.agent.init_prompt_chain(content_s, prompt_s)

        while not stats["search_complete"]:
            start = time.time()
            output = self.agent.query_msg_chain(
                temperature=self.temperature, top_p=self.top_p
            )
            stats["search_time"] += time.time() - start
            logger.info(f"--------- SayPlan. Search attempt #{stats['search_count'] + 1}. Elapsed time: {stats['search_time']:.2f}s -----------")
            self.agent.update_prompt_chain_with_response(output)
            
            mode, cot, reasoning, cmd = export_search_cmd(output)
            
            if "search complete" in cot or ("switch" in reasoning and "planning mode" in reasoning):
                stats["search_complete"] = True
                logger.info("Scene graph search complete! Switching to planning mode.")
                break
            
            if len(memory) == len(query_sg):
                logger.info("All rooms have been explored! Search stop!")
                break
            
            if cmd[0] == "expand" and cmd[1] in memory:
                prompt_s = f"Room {cmd[1]} has already been explored. Remaining unexplored rooms are: {set(query_sg.keys()) - memory}.\n"
            else:
                query_collapsed_sg = update_scene_graph(
                    query_collapsed_sg, query_sg, cmd[0], cmd[1]
                )
                memory.add(cmd[1])
                prompt_s = f"3D Scene Graph: {query_collapsed_sg}\nMemory: {memory}"
            self.agent.update_prompt_chain_with_response(prompt_s, role="user")
            
            stats["search_count"] += 1
            if stats["search_count"] == self.max_search_attempts:
                stats["max_search_reached"] = True
                stats["exit_code"] = 6
                logger.warning("Maximum search attempts reached! Stopping search.")
                break

        return query_collapsed_sg, stats

    def iterative_replanning(self, query_sg, query_goal, query_additional_actions, results_problem_dir, stats):

        if not stats["max_search_reached"]:
            content_p, prompt_p = plan_prompt(
                query_goal, query_sg, query_additional_actions
            )
            self.agent.update_prompt_chain(content_p, prompt_p)
            
            for t in range(self.max_debug_attempts+1):
                stats["replan_count"] += 1
                start = time.time()
                output = self.agent.query_msg_chain(
                    temperature=self.temperature, top_p=self.top_p
                )
                stats["plan_time"] += time.time() - start
                
                logger.info(f"--------- SayPlan. Replanning attempt #{stats['replan_count'] + 1}. Elapsed time: {stats['plan_time']:.2f}s -----------")
                self.agent.update_prompt_chain_with_response(output)
                
                plan_file = os.path.join(results_problem_dir, f"plan_{stats['replan_count']}.txt")
                try:
                    plan_list, plan_cost = dump_sayplan_plan(output, plan_file)
                    stats["plan_cost"] = plan_cost
                except Exception as err:
                    err_msg = f"Error while dumping plan: {err}"
                    self.agent.update_prompt_chain_with_response(err_msg, role="user")
                    logger.error(err_msg)
                    stats["exit_code"] = 7
                    continue

                stats["success"] = True
                
        return stats