import csv
import os
import time
from pathlib import Path
from typing import Optional
import json
import sys

from src.context_matters.pipelines.base_pipeline import BasePipeline
from src.context_matters.logger_cfg import logger

from src.context_matters.prompt.llm_as_planner_prompts import sg_2_plan
from src.context_matters.prompt.delta_example import load_example_data
from src.context_matters.utils.graph import read_graph_from_path, get_verbose_scene_graph


class LLMAsPlannerPipeline(BasePipeline):
    """Pipeline wrapper for the LLM-as-planner baseline (ported from DELTA/baselines/llm_as_planner.py).

    This class intentionally mirrors the original baseline structure and leaves TODOs and
    commented code similar to how `DeltaPipeline` was implemented in the repo.
    """

    def __init__(self,
                 model_name: str = "gpt-4o",
                 temperature: float = 0.0,
                 top_p: float = 1.0,
                 episodes: int = 5,
                 **base_init_kwargs):
        super().__init__(**base_init_kwargs, generate_domain = False, ground_in_sg = False)
        self.model_name = model_name
        self.temperature = temperature
        self.top_p = top_p
        self.episodes = episodes
        self.experiment_name = super()._construct_experiment_name()


    def _initialize_csv(self, csv_filepath: str):
        """Initializes the results CSV file with a header relevant to the delta_pipeline.csv."""
        
        # This header is tailored for a single-shot LLM planner.
        # It removes all the subgoal and decomposition metrics from the original delta_pipeline.py
        header = [
            # Identifiers
            "Task",
            "Scene",
            "Problem",
            "Total LLM Time",
            "Planning Successful", 
            "Grounding Successful"
            "Plan Length",
            "Failure stage", 
            "Failure Reason"
        ]
        
        with open(csv_filepath, mode = "w", newline = '') as f:
            writer = csv.writer(f, delimiter = '|')
            writer.writerow(header)

    def _serialize_domain_knowledge(self, knowledge_list: list | dict, actions: bool = False) -> list[str]:
        """
        Serialize domain objects or actions into a single string, wrapped inside a one-element list.
        This function is robust and handles two possible formats for knowledge_list:
        1. A list of dictionaries (standard format).
        2. A dictionary of dictionaries (will be converted to the standard format).
        """
        processed_list = []

        if isinstance(knowledge_list, dict):
            for name, details in knowledge_list.items():
                if isinstance(details, dict):
                    item = details.copy()
                    item['name'] = name
                    processed_list.append(item)
        elif isinstance(knowledge_list, list):
            processed_list = knowledge_list

        # --- FIX IS HERE ---
        prompt_lines = [] # Initialize as an empty list
        # ------------------

        for i, item in enumerate(processed_list, start=1):
            if not isinstance(item, dict):
                continue
            if actions:
                name = item.get('name', 'N/A')
                arguments = item.get('arguments', [])
                description = item.get('description', 'No description available.')
                prompt_lines.append(f"{i}. {name}({', '.join(arguments)}): {description}")
            else:
                item_type = item.get('type', 'N/A')
                description = item.get('description', 'No description available.')
                prompt_lines.append(f"{i}. {item_type}: {description}")

        return ["\n".join(prompt_lines)]



    def _parse_and_save_plan(self, llm_response_str: str, results_problem_dir: str):
        """
        Parses the LLM's string response to extract, clean, and save the plan.
        This logic is adapted from the original DELTA baseline's llm_utils.
        """
        plan_filepath = os.path.join(results_problem_dir, "plan.txt")

        try:
            start_idx = llm_response_str.find("{")
            end_idx = llm_response_str.rfind("}") + 1 
            if start_idx == -1 or end_idx == 0:
                raise ValueError("Could not find a JSON object in the LLM response.")

            json_str = llm_response_str[start_idx:end_idx]
            response_data = json.loads(json_str)
            
            plan = response_data["plan"]

        except (json.JSONDecodeError, KeyError, ValueError) as e:
            raise ValueError(f"Failed to parse plan from LLM response. Details: {e}")


        if "```" in plan:
            plan = plan.split("```")[1]
        if "\\n" in plan:
            plan = plan.replace("\\n", "\n")
        
        response_lines = plan.strip().split("\n")
        
        plan_list = [line.strip() for line in response_lines if line.strip()]
        
        plan_length = len(plan_list)

        with open(plan_filepath, "w") as f:
            f.write("\n".join(plan_list))

        #print(f"Saved plan with {plan_length} steps to {plan_filepath}")
            
        return plan_list, plan_length


    def _run_and_log_pipeline(self, task_name, scene_name, problem_id, results_problem_dir,
                              domain_file_path, domain_description, scene_graph_file_path,
                              csv_filepath):
        """Run the LLM-as-planner baseline for a single problem and log outputs.

        This method contains a mostly literal port of the original baseline's main loop and
        intentionally keeps TODOs and commented sections where integration with the rest of
        the framework is pending (LLM loader, planner validator, etc.).
        """

        #Setting up paths
        domain_file_path = os.path.join(self.data_dir, f"{task_name}.json")
        agent_starting_position_path = os.path.join(self.data_dir, str(task_name), str(scene_name), str(problem_id), "init_loc.txt")
        goal_qry_path = os.path.join(self.data_dir, str(task_name), str(scene_name), str(problem_id), "task.txt")
        scene_graph_file_path = Path(scene_graph_file_path)

        with open(domain_file_path, "r") as f: 
            domain_file = json.load(f) 

        with open(agent_starting_position_path, "r") as f:
            agent_starting_position = f.read().strip()

        with open(goal_qry_path, "r") as f:
            agent_starting_position = f.read().strip()  

        """
        scene_graph_file_path_jason_path = scene_graph_file_path.with_suffix('.json')
        with open(scene_graph_file_path_jason_path, "r") as f: 
            scene_qry = json.load(f)
        """

        _, scene_exp, _, domain_data_exp = load_example_data()

        scene_qry = read_graph_from_path(scene_graph_file_path)
        scene_qry = get_verbose_scene_graph(scene_qry, as_string = False, include_descriptions = False)
        scene_exp = get_verbose_scene_graph(scene_exp, as_string = False, include_descriptions = False)


        goal_exp = domain_data_exp["goal"]
        add_act_exp = domain_data_exp.get("add_act")
        add_obj_exp = domain_data_exp.get("add_obj") 

        #goal_qry = domain_file["problems"][str(problem_id)]["goal"]
        with open(goal_qry_path, "r") as f:
            goal_qry = f.read().strip()

        add_obj_qry = self._serialize_domain_knowledge(domain_file.get("domain", {}).get("objects", []), actions = False)
        add_act_qry = self._serialize_domain_knowledge(domain_file.get("domain", {}).get("actions", []), actions = True)


        logger.info(f"Agent starting position: {agent_starting_position}")
        logger.info(f"Goal query: {goal_qry}")
        
        content, prompt = sg_2_plan(sg_exp = scene_exp, 
                                    sg_qry = scene_qry, 
                                    add_obj_exp = add_obj_exp,
                                    add_act_exp = add_act_exp,
                                    add_act_qry = add_act_qry,
                                    add_obj_qry = add_obj_qry,
                                    goal_exp = goal_exp, 
                                    goal_qry = goal_qry,
                                    agent_starting_position = agent_starting_position)  
        
        start = time.time()

        try:
            llm_response_str = self.agent.llm_call(content = content, 
                                                   prompt = prompt)
            
            logger.info("Successfully received response from LLM.")
            
        except Exception as e:
            logger.error("LLM API call failed: %s", e)
            llm_response_str = f"Error: LLM query failed. Details: {e}"

        llm_time = time.time() - start


        try:
            plan_list, plan_length = self._parse_and_save_plan(llm_response_str, results_problem_dir)
            planning_successful = True
            logger.info(f"Successfully parsed plan with {plan_length} steps.")
        
        except ValueError as e:
            planning_successful = False
            failure_stage = "Plan Parsing"
            failure_reason = str(e)
            logger.error(f"Failed to parse LLM response: {failure_reason}")
        
        except Exception as e:
            planning_successful = False
            failure_stage = "Unknown"
            failure_reason = f"An unexpected error occurred during parsing: {e}"
            logger.error(failure_reason)

        print("**********************************************************************************************************************************************")
        print()

        #write to csv:
        with open(csv_filepath, mode = "a", newline = '') as f:
            writer = csv.writer(f, delimiter = '|')
            if planning_successful:
                writer.writerow([
                    task_name,
                    scene_name,
                    problem_id,
                    f"{llm_time:.2f}",
                    planning_successful,
                    "N/A", # Grounding successful (TODO)
                    plan_length,
                    "N/A", # Failure stage
                    "N/A"  # Failure reason
                ])
            else:
                writer.writerow([
                    task_name,
                    scene_name,
                    problem_id,
                    f"{llm_time:.2f}",
                    planning_successful,
                    "N/A", # Grounding successful (TODO)
                    0, # Plan length
                    failure_stage,
                    failure_reason
                ])



        # TODO Validation and grounding as in delta_pipeline.py

        # Append per-episode logging to csv or internal structures as needed
        #logger.info("LLM-As-Planner episode %s ran (placeholder). LLM time: %.2fs", e + 1, llm_time)

        # End for episodes
