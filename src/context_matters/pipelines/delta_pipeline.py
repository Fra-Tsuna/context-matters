# Adaptation of the DELTA Pipeline from the official code repository.
# Original paper: Y. Liu et al. TODO, TODO
# Paper link: TODO
# Based on the official code implementation by Y. Liu et al. TODO(link)

import csv
import os
import time
import json
from pathlib import Path
from datetime import datetime
import traceback

from src.context_matters.planner import plan_with_output
from src.context_matters.prompt import delta_example
from src.context_matters.pddl_generation import _save_prompt_response


from .base_pipeline import BasePipeline

from src.context_matters.prompt.delta_prompts import (
    nl_2_pddl_domain,
    nl_prune_item,
    sg_2_pddl_problem,
    decompose_problem,
    decompose_problem_chain
)
from src.context_matters.utils.graph import (
    filter_graph,
    get_verbose_scene_graph,
    read_graph_from_path,
    extract_accessible_items_from_sg,
    prune_sg_with_item,
    prune_sg_with_item_OURS,
    save_graph
)
from src.context_matters.utils.log import (
    save_file, save_statistics, save_log_file
)

from src.context_matters.logger_cfg import logger

from src.context_matters.pddl_verification import (
    VAL_validate,
    VAL_ground,
    verify_groundability_in_scene_graph,
    convert_JSON_to_locations_dictionary,
    translate_plan,
)

from src.context_matters.pipelines import delta_planner

from src.context_matters.utils.graph import (
    export_obj_list,
)

from src.context_matters.utils.llm import (
    export_result,
    export_subgoal_list
)

from pddlgym.core import PDDLEnv

#TODO: save planner/grounder failure reasons in separate files

class DeltaPipeline(BasePipeline):
    def __init__(self, 
                 max_time=60,
                 temperature=0.0,
                 top_p=1.0,
                 experiments=["all"],
                 sg_pruning=True,
                 goal_decomposition=True,
                 generate_domain=True,
                 ground_in_sg=False,
                 intermediate_validation=True,
                 use_extracted_sg=True,
                 whole_problem_planning=True,
                 delta_plus=False,
                 **base_init_kwargs):
        super().__init__(**base_init_kwargs, generate_domain=generate_domain, ground_in_sg=ground_in_sg)
        self.max_time = max_time
        self.temperature = temperature
        self.top_p = top_p
        self.experiments = experiments
        self.generate_domain = generate_domain
        self.sg_pruning = sg_pruning
        self.goal_decomposition = goal_decomposition
        self.ground_in_sg = ground_in_sg
        self.use_extracted_sg = use_extracted_sg
        self.intermediate_validation = intermediate_validation
        self.whole_problem_planning = whole_problem_planning
        self.experiment_name = super()._construct_experiment_name()
        self.current_phase = None  # Initialize current_phase attribute
        self.delta_plus = delta_plus


    def _initialize_csv(self, csv_filepath):
        # Initialize CSV with headers
        header = ["Task", "Scene", "Problem", "Planning Successful", "Grounding Successful", 
                "Plan Length", "Number of subgoals", "Number of successful subplans", "Domain Generation Time", "Scene Pruning Time", 
                "Problem Generation Time", "Problem Decomposition Time", "Total LLM Time", 
                "Subgoal Planning Times", "Subgoal Planning Nodes", "Subgoal Planning Costs", "Exit Code Decomp",
                "Total Subgoal Planning Time", "Total Subgoal Planning Nodes", "Total Subgoal Planning Costs",
                "Failure stage", "Failure Reason", "VAL Validation Log Path", "VAL Grounding Log Path", "Planning Log Path", "Grounding Log Path"]
        
        with open(csv_filepath, mode="w", newline='') as f:
            writer = csv.writer(f, delimiter='|')
            writer.writerow(header)


    def _generate_domain(self, domain_example, domain_name, domain_pddl_path, add_obj_example, add_obj_qry, add_act_example, add_act_qry, logs_dir):
        """
        Stage 1: Generating domain file
        """

        content_d, prompt_d = nl_2_pddl_domain(
            domain_example, domain_name, add_obj_example, add_obj_qry, add_act_example, add_act_qry)

        print(content_d)
        print(prompt_d)

        #Part of TODO #5: this should be part of the statistics refactor
        #print("Prompt tokens for domain generation: {}".format(
        #    model.count_tokens(prompt_d)))

        self.agent.init_prompt_chain(content_d, prompt_d) #OK

        d_start = time.time()
        
        domain_pddl = self.agent.query_msg_chain()

        print(domain_pddl)

        d_time = time.time() - d_start
        export_result(domain_pddl, domain_pddl_path)
        print(
            "Response time for generating domain file: {:.2f}s".format(d_time))

        # Save prompts and response
        _save_prompt_response(
            prompt=f"{content_d}\n\n{prompt_d}",
            response=domain_pddl,
            prefix="DELTA_pddl_domain_generation",
            suffix="",
            output_dir=logs_dir
        )

        self.agent.update_prompt_chain_with_response(domain_pddl)
        
        return domain_pddl, d_time


    def _prune_scene_graph(self, scene_example, scene_qry, domain_pddl,
                          goal_example, goal_qry, item_keep_exp, domain_example, logs_dir, chain_from_domain=False):
        """
        Stage 2: Pruning scene graph items
        """
        items_example = scene_example
        items_qry = scene_qry

        print(items_example)
        print(items_qry)
        
        content_pr, prompt_pr = nl_prune_item(items_example, items_qry, goal_example, goal_qry, item_keep_exp, domain_example, domain_pddl)

        print(content_pr)
        print(prompt_pr)
        
        # Part of TODO #5: this should be part of the statistics refactor
        #print("Prompt tokens for pruning scene graph: {}".format(
        #    model.count_tokens(prompt_pr)))

        if chain_from_domain:
            self.agent.update_prompt_chain(content_pr, prompt_pr)
        else:
            self.agent.init_prompt_chain(content_pr, prompt_pr)
        
        pr_start = time.time()
        
        pruned_sg = self.agent.query_msg_chain() #OK
        
        pr_time = time.time() - pr_start

        item_keep = export_obj_list(pruned_sg)

        logger.info("Items to keep: {}".format(item_keep))
        logger.info("Response time for pruning scene graph: {:.2f}s".format(pr_time))

        # Save prompts and response
        _save_prompt_response(
            prompt=f"{content_pr}\n\n{prompt_pr}",
            response=pruned_sg,
            prefix="DELTA_pruned_scene_graph",
            suffix="",
            output_dir=logs_dir
        )

        self.agent.update_prompt_chain_with_response(pruned_sg) #OK

        scene_example = prune_sg_with_item_OURS(scene_example, item_keep_exp, is_extracted_sg = self.use_extracted_sg)
        pruned_sg = prune_sg_with_item_OURS(scene_qry, item_keep, is_extracted_sg=self.use_extracted_sg)


        print("PRUNED:\n"+str(pruned_sg))

        return scene_example, pruned_sg, item_keep, pr_time


    def _generate_problem(self, domain_example_name, domain_example, problem_example, scene_example, problem_pddl_path, scene_qry,
                         goal_example, goal_qry, domain_pddl, domain_name, initial_robot_location, logs_dir, chain_from_previous=False):
        """
        Stage 3: Generating problem file
        """

        content_p, prompt_p = sg_2_pddl_problem(domain_example_name, domain_example, problem_example,
                                                scene_example, scene_qry, goal_example,
                                                goal_qry, domain_pddl, domain_name, initial_robot_location, delta_plus=self.delta_plus)
        
        print(content_p)
        print(prompt_p)

        # Part of TODO #5: statistics and logging refactor
        #print("Prompt tokens for problem generation: {}".format(
        #    model.count_tokens(prompt_p)))

        # OK
        if chain_from_previous:
            self.agent.update_prompt_chain(content_p, prompt_p)
        else:
            self.agent.init_prompt_chain(content_p, prompt_p)

        p_start = time.time()
        
        problem_pddl = self.agent.query_msg_chain() #OK

        print(problem_pddl)
        
        p_time = time.time() - p_start
        
        export_result(problem_pddl, problem_pddl_path)
        
        print("Response time for generating problem file: {:.2f}s".format(p_time))

        # Save prompts and response
        _save_prompt_response(
            prompt=f"{content_p}\n\n{prompt_p}",
            response=problem_pddl,
            prefix="DELTA_generate_pddl_problem",
            suffix="",
            output_dir=logs_dir
        )

        self.agent.update_prompt_chain_with_response(problem_pddl) #OK

        return problem_pddl, p_time


    def _decompose_problem(self, goal_example, subgoal_example, subgoal_pddl_example, item_keep_example,
                      goal_qry, problem_example, item_keep, problem_pddl, domain_pddl,
                      logs_dir, chain_from_previous=False, accumulate_subgoal=False):
        """
        Stage 4: Decomposing problem file
        """

        if chain_from_previous:
            content_dp, prompt_dp = decompose_problem_chain(
                goal_example, subgoal_example, subgoal_pddl_example, item_keep_example,
                goal_qry, problem_example, item_keep, problem_pddl, domain_pddl,
                accumulate_subgoal)
            
            self.agent.update_prompt_chain(content_dp, prompt_dp) #OK
        else:
            content_dp, prompt_dp = decompose_problem(
                goal_example, subgoal_example, subgoal_pddl_example, item_keep_example,
                goal_qry, problem_example, item_keep, problem_pddl, domain_pddl,
                accumulate_subgoal)
            
            self.agent.init_prompt_chain(content_dp, prompt_dp) #OK

        # Part of TODO #5: statistics and logging refactor
        #print("Prompt tokens for problem decomposition: {}".format(
        #    model.count_tokens(prompt_dp)))

        dp_start = time.time()
        
        decomp_goal = self.agent.query_msg_chain()
        
        dp_time = time.time() - dp_start

        subgoal_pddl_list = export_subgoal_list(decomp_goal)
        
        print("Response time for decomposing problem file: {:.2f}s".format(dp_time))

        # Save prompts and response
        _save_prompt_response(
            prompt=f"{content_dp}\n\n{prompt_dp}",
            response=decomp_goal,
            prefix="DELTA_decompose_pddl_goal",
            suffix="",
            output_dir=logs_dir
        )

        self.agent.update_prompt_chain_with_response(decomp_goal) #OK

        return subgoal_pddl_list, dp_time


    def _perform_planning(self, domain_name, domain_pddl_file_path, problem_pddl_file_path, 
                          plan_file_path, plan_decomp_file_path, subgoal_pddl_list,
                          subproblems_dir, logs_dir, results_dir, pruned_sg=None, initial_robot_location=None):
        """
        Planning stage: Execute planner with whole-problem planning first, then decomposed subgoals
        """
        
        # Whole-problem planning first
        self.current_phase = "PLANNING"
        logger.info("Attempting whole-problem planning")
        
        if self.whole_problem_planning:
            try:
                whole_plan, whole_pddlenv_error_log, whole_planner_error_log, whole_statistic = plan_with_output(
                    domain_pddl_file_path, os.path.dirname(problem_pddl_file_path), plan_file_path, env=None, timeout=int(self.max_time))
                
                whole_plan_successful = (whole_plan is not None and whole_plan != "")
                
                # Save whole-problem planning statistics
                save_statistics(
                    dir=results_dir,
                    workflow_iteration=0,
                    phase=self.current_phase,
                    plan_successful=whole_plan_successful,
                    pddlenv_error_log=whole_pddlenv_error_log,
                    planner_error_log=whole_planner_error_log,
                    planner_statistics=whole_statistic
                )
                
                if whole_plan_successful:
                    logger.info("Whole-problem planning succeeded")
                    # Validate whole plan using VAL
                    try:
                        whole_translated = os.path.join(results_dir, "plan_whole_translated.txt")
                        translate_plan(plan_file_path, whole_translated)
                    except Exception:
                        # fallback to original plan if translation fails
                        whole_translated = plan_file_path

                    is_valid_whole, val_info_whole = VAL_validate(domain_pddl_file_path, problem_pddl_file_path, whole_translated)
                    if is_valid_whole:
                        logger.info("Whole-problem plan is valid - continuing with subgoal decomposition for comparison")
                    else:
                        logger.info("Whole-problem plan is invalid - continuing with subgoal decomposition")
                else:
                    logger.info("Whole-problem planning failed - continuing with subgoal decomposition")
                    
            except Exception as e:
                logger.info(f"Whole-problem planning failed with exception: {e} - continuing with subgoal decomposition")
        
        # Hierarchical planning for sub-problems using PDDLEnv
        # Final success is determined by subgoal decomposition results (validation/grounding moved to Stage 5)
        if len(subgoal_pddl_list) > 0:

            plans, times, nodes, costs, exit_code_decomp, completed_sp, pddlenv_error_logs, planner_error_logs = delta_planner.query_pddlenv(
                domain_name, domain_pddl_file_path, problem_pddl_file_path, subgoal_pddl_list,
                results_dir, save_path=subproblems_dir, max_time=self.max_time)

            # Save planning logs to files
            if pddlenv_error_logs:
                planning_log_path = save_log_file(str(pddlenv_error_logs), "planning", logs_dir)
            elif planner_error_logs:
                planning_log_path = save_log_file(str(planner_error_logs), "planning", logs_dir)
            else:
                planning_log_path = save_log_file("No planning errors", "planning", logs_dir)

            # Save planning statistics
            save_statistics(
                dir=results_dir,
                workflow_iteration=0,
                phase="SUBGOAL_PLANNING",
                plan_successful=(exit_code_decomp == 1),
                planner_statistics={"times": times, "nodes": nodes, "costs": costs, "exit_code": exit_code_decomp, "completed_subproblems": completed_sp}
            )

            return plans, times, nodes, costs, exit_code_decomp, completed_sp
        else:
            print("No decomposed subgoals!")
            plans, times, nodes, costs, completed_sp = [], [], [], [], None
            exit_code_decomp = 7
            return plans, times, nodes, costs, exit_code_decomp, completed_sp
    

    def _delta_planning(self, **kwargs):

        domain_name = kwargs["domain_name"]
        goal_file_path = kwargs["goal_file_path"]
        initial_location_file_path = kwargs["initial_location_file_path"]
        scene_graph_file_path = kwargs["scene_graph_file_path"]
        domain_pddl_path = kwargs["domain_pddl_path"]
        results_dir = kwargs["results_dir"]
        domain_description = kwargs["domain_description"]
        
        logs_dir = os.path.join(results_dir, "logs")
        os.makedirs(logs_dir, exist_ok=True)
        
        initial_robot_location = open(initial_location_file_path, "r").read()

        self.agent.reset()
        
        # Initialize log path variables
        val_validation_log_path = ""
        val_grounding_log_path = ""
        planning_log_path = ""
        grounding_log_path = ""

        # NOTICE: change this method if you want to load examples dinamically
        domain_example, scene_example, problem_example, domain_example_data = delta_example.load_example_data(delta_example.HARDCODED_DOMAIN_EXAMPLE_NAME)
        # TODO #4: Consider whether to replace the hard-coded examples based on cm_pipeline.py with dynamic examples loading as in DELTA
        #exp = example.get_example(domain_example)
        #qry = example.get_example(domain_name)
        #if scene_example not in exp["scene"]:
        #    raise Exception("Scene example {} is not supported for domain example{}!".format(
        #        scene_example, domain_example))
        #if scene_name not in qry["scene"]:
        #    raise Exception("Scene {} is not supported for domain {}!".format(
        #        scene_name, domain_name))        
        #if scene_name == scene_example:
        #    raise ValueError(
        #        "Scene graph example cannot be identical to scene graph query!")
        #print("Using model {}".format(model_name))
               

        d_time, pr_time, p_time, dp_time = 0., 0., 0., 0.
        start = time.time()


        ###################### Stage 1: Generating domain file ######################
        if self.generate_domain:

            self.current_phase = "DOMAIN GENERATION"
            assert domain_description is not None, "Provide the domain or the domain description"
            assert "actions" in domain_description, "Domain description must contain 'actions'"
            assert "objects" in domain_description, "Domain description must contain 'objects'"
            add_obj_qry = domain_description["objects"]
            add_act_qry = domain_description["actions"]

            logger.info("Generating PDDL domain")
            
            domain_pddl_path = os.path.join(results_dir, "domain", "domain.pddl")
            
            domain_pddl, d_time = self._generate_domain(
                domain_example, domain_name, domain_pddl_path, 
                domain_example_data["add_obj"], add_obj_qry, domain_example_data["add_act"], add_act_qry, 
                logs_dir)
            
            # Save domain generation statistics
            save_statistics(
                dir=results_dir,
                workflow_iteration=0,
                phase=self.current_phase,
                domain_generation_time=d_time
            )
                
            logger.info("Domain generation successful")
        else:
            logger.info("Using provided PDDL domain file: {}".format(domain_pddl_path))

            assert domain_pddl_path is not None and os.path.exists(domain_pddl_path) and os.path.isfile(domain_pddl_path), "Provide a valid domain file path"
            # Load domain file from the provided path
            with open(domain_pddl_path, "r") as f:
                domain_pddl = f.read()
            d_time = 0.0  # No time spent on generation since using existing file

        
        ####################### Stage 1.1: Domain VALidation ######################

        # Domain validation (optional, can be enabled if needed)
        self.current_phase = "DOMAIN VALIDATION"
        logger.info("Validating the domain")
        val_parse_success, val_parse_log = VAL_validate(domain_pddl_path)
        
        # Save VAL validation log to file
        val_validation_log_path = save_log_file(val_parse_log, "VAL_validation", logs_dir)
        
        save_statistics(
            dir=results_dir,
            workflow_iteration=0,
            phase=self.current_phase,
            VAL_validation_log_path=val_validation_log_path
        )
        if self.intermediate_validation:
            if not val_parse_success:
                logger.error("Domain validation failed on {}: {}".format(domain_pddl_path, val_parse_log))
                # return placeholders for the full expected tuple (26 elements with log paths)
                return domain_pddl_path, None, None, None, False, False, None, "DOMAIN_GENERATION", val_parse_log, None, d_time, 0.0, 0.0, 0.0, d_time, [], [], [], None, 0.0, 0, 0, val_validation_log_path, "", "", ""
            else:
                logger.info("Domain validation successful")
        else:
            logger.info("Skipping domain validation")

        ###################### Step 2: Pruning scene graph items ######################
        scene_qry = read_graph_from_path(Path(scene_graph_file_path))


        scene_qry = get_verbose_scene_graph(scene_qry, as_string=False, include_descriptions=False)
        scene_example = get_verbose_scene_graph(scene_example, as_string=False, include_descriptions=False)

            
        goal_qry = Path(goal_file_path).read_text()

        if self.sg_pruning:
            logger.info("Pruning scene graph")
            self.current_phase = "PRUNING_SCENE_GRAPH"

            chain_from_domain = self.generate_domain

            scene_exp, scene_qry, item_keep, pr_time = self._prune_scene_graph(
                scene_example, scene_qry, domain_pddl,
                domain_example_data["goal"], goal_qry, domain_example_data["item_keep"], domain_example, logs_dir, chain_from_domain)

            # Save pruned scene graph
            pruned_sg_path_readable = os.path.join(results_dir, "pruned_sg.txt")
            with open(pruned_sg_path_readable, "w") as f:
                f.write(json.dumps(scene_qry, indent=4))

            pruned_sg_path_npz = os.path.join(results_dir, "pruned_sg.npz")
            save_graph(scene_qry, pruned_sg_path_npz)

            # Save items_keep to a txt file
            items_keep_path = os.path.join(results_dir, "items_keep.txt")
            with open(items_keep_path, "w") as f:
                f.write("\n".join(item_keep))

            # Save scene graph pruning statistics
            save_statistics(
                dir=results_dir,
                workflow_iteration=0,
                phase=self.current_phase,
                scene_pruning_time=pr_time
            )

        else:
            logger.info("Skipping scene graph pruning")
            item_keep = None
            pr_time = 0.0
            
            # Save original scene graph as "pruned" for consistency
            pruned_sg_path_readable = os.path.join(results_dir, "pruned_sg.txt")
            with open(pruned_sg_path_readable, "w") as f:
                f.write(json.dumps(scene_qry, indent=4))

            # Only save as npz if scene_qry is a dictionary (graph object)
            if isinstance(scene_qry, dict):
                pruned_sg_path_npz = os.path.join(results_dir, "pruned_sg.npz")
                save_graph(scene_qry, pruned_sg_path_npz)


        ###################### Stage 3: Generating problem file ######################
        
        logger.info("Generating PDDL problem")
        self.current_phase = "PROBLEM GENERATION"

        chain_from_previous = domain_pddl_path is None or self.sg_pruning
        logger.info("Chain to previous prompts: {}".format(chain_from_previous))

        problem_pddl_path = os.path.join(results_dir, "problem.pddl")

        problem_pddl, p_time = self._generate_problem(
            delta_example.HARDCODED_DOMAIN_EXAMPLE_NAME, domain_example, problem_example, scene_example, problem_pddl_path, scene_qry,
            domain_example_data["goal"], goal_qry, domain_pddl, domain_name, initial_robot_location, logs_dir, chain_from_previous)
        
        # Save problem generation statistics
        save_statistics(
            dir=results_dir,
            workflow_iteration=0,
            phase=self.current_phase,
            problem_generation_time=p_time
        )
            

        ####################### Stage 3.1: Problem VALidation ######################

        logger.info("Validating the problem")
        val_parse_success, val_parse_log = VAL_validate(domain_pddl_path, problem_pddl_path)
        
        # Save VAL validation log to file
        val_validation_log_path = save_log_file(val_parse_log, "VAL_validation", logs_dir)
        
        save_statistics(
            dir=results_dir,
            workflow_iteration=0,
            phase=self.current_phase,
            VAL_validation_log_path=val_validation_log_path
        )
        
        if self.intermediate_validation:
            if not val_parse_success:
                logger.info("Problem validation failed on {}: {}".format(problem_pddl_path, val_parse_log))
                # Return placeholders for the full expected tuple (26 elements with log paths)
                return domain_pddl_path, scene_qry, problem_pddl_path, None, False, False, None, "PROBLEM_GENERATION", val_parse_log, None, d_time, pr_time, p_time, 0.0, d_time + pr_time + p_time, [], [], [], None, 0.0, 0, 0, val_validation_log_path, "", "", ""
            else:
                logger.info("Problem validation successful")
        else:
            logger.info("Skipping problem validation")


        ###################### Stage 4: Decomposing problem file ######################

        if self.goal_decomposition:

            logger.info("Generating PDDL sub-goals")
            self.current_phase = "SUBGOAL_GENERATION"

            accumulate_subgoal = True if domain_name == "office" else False
            chain_from_previous = self.generate_domain or self.sg_pruning

            assert item_keep is not None and len(item_keep) > 0, "Item keep list is empty"

            subgoal_pddl_list, dp_time = self._decompose_problem(
                domain_example_data["goal"], domain_example_data["subgoal"], domain_example_data["subgoal_pddl"], domain_example_data["item_keep"],
                goal_qry, problem_example, item_keep, problem_pddl, domain_pddl,
                logs_dir, chain_from_previous, accumulate_subgoal)

            print(subgoal_pddl_list)
            
            # Save problem decomposition statistics
            save_statistics(
                dir=results_dir,
                workflow_iteration=0,
                phase=self.current_phase,
                problem_decomposition_time=dp_time
            )
                
            total_llm_time = d_time + pr_time + p_time + dp_time
            print("Total response time: {:.2f}s".format(total_llm_time))
        else:
            subgoal_pddl_list = []
            dp_time = 0.0
            total_llm_time = d_time + pr_time + p_time  # No decomposition time

        # Calculate total inference time (ensure we have it for all paths)
        total_llm_time = d_time + pr_time + p_time + dp_time

        # Save total inference time statistics
        save_statistics(
            dir=results_dir,
            workflow_iteration=0,
            phase="TOTAL_INFERENCE",
            total_inference_time=total_llm_time,
            domain_generation_time=d_time,
            scene_pruning_time=pr_time,
            problem_generation_time=p_time,
            problem_decomposition_time=dp_time
        )


        ##################### Generating task plan(s) ######################

        logger.info("Autoregressive planning and grounding")
        self.current_phase="PLANNING"
        
        plan_file_path = os.path.join(results_dir, f"plan.txt")
        plan_decomp_file_path = os.path.join(results_dir, f"plan_decomp.txt")
        subproblems_dir = os.path.join(results_dir, "subproblems")

        subplans, times, nodes, costs, exit_code_decomp, completed_sp = self._perform_planning(
            domain_name, domain_pddl_path, problem_pddl_path,
            plan_file_path, plan_decomp_file_path, subgoal_pddl_list,
            subproblems_dir, logs_dir, results_dir, pruned_sg=scene_qry, initial_robot_location=initial_robot_location
        )
        

        ###################### Stage 5: Validation and Grounding ######################

        # Final success is determined ONLY by subgoal decomposition results
        planning_successful = (exit_code_decomp == 1)
        grounding_successful = False  # Initial assumption
        
        # Separate grounding phase after planning
        if planning_successful:
            logger.info("Planning successful, starting grounding validation")
            self.current_phase = "SUBGOAL_GROUNDING"
            
            # Validation and grounding for each successful subgoal
            for i, (sub_goal, plan) in enumerate(zip(subgoal_pddl_list, subplans)):
                logger.info(f"Processing sub-goal {i+1}/{len(subgoal_pddl_list)}")
                if plan:  # Check if plan exists
                    logger.info("Plan exists for this sub-goal")
                    
                    sub_goal_dir = os.path.join(subproblems_dir, f"sub_goal_{i+1}")
                    os.makedirs(sub_goal_dir, exist_ok=True)
                    sub_goal_file = os.path.join(sub_goal_dir, f"sub_goal_{i+1}.pddl")
                    output_plan_file_path = os.path.join(sub_goal_dir, f"plan_{i+1}.txt")
                    translated_plan_path = os.path.join(sub_goal_dir, f"translated_plan_{i+1}.txt")
                    
                    # Write subgoal to file
                    with open(sub_goal_file, "w") as sgf:
                        sgf.write(sub_goal)
                                
                    # Write plan to file
                    with open(output_plan_file_path, "w") as f:
                        f.write("\n".join([str(action) for action in plan]))

                    # Translate the plan into a format parsable by VAL
                    translate_plan(output_plan_file_path, translated_plan_path)

                    logger.info("Validating the plan")
                    self.current_phase = f"SUBGOAL_{i+1}:VALIDATION"
                    
                    # Use VAL to validate the plan
                    val_successful, val_log = VAL_validate(domain_pddl_path, sub_goal_file, translated_plan_path)
                    logger.info(f"Result: {val_successful}")
                    logger.debug(val_log)

                    
                    self.current_phase = f"SUBGOAL_{i+1}:VAL"
                    save_statistics(
                        dir=results_dir,
                        workflow_iteration=0,
                        phase=self.current_phase,
                        plan_successful=True,
                        VAL_validation_log=val_log,
                        planner_statistics={"times": times, "nodes": nodes, "costs": costs}
                    )
                        
                else:
                    logger.info("Empty plan for this sub-goal. Skipping")

                    save_statistics(
                        dir=results_dir,
                        workflow_iteration=0,
                        phase=f"SUBGOAL_{i+1}:PLANNING",
                        plan_successful=False
                    )
                    
                    continue
                    
                logger.info(f"Sub-goal {i+1} completed successfully")
            

            final_plan_file_path = os.path.join(results_dir, "plan_final.txt")

            # Concatenate all subplans and write the resulting plan
            # Flatten subplans into a single list of action strings, stripping empty lines,
            # then write them consecutively without inserting extra blank lines between subplans.
            final_plan = []
            final_plan_text = ""
            for subplan in subplans:
                for action in subplan:
                    final_plan.append(action)
                    final_plan_text += str(action).strip() + "\n"
            with open(final_plan_file_path, "w") as f:
                # Write actions with a comma and newline after each action (keeps original formatting),
                # but ensure no empty lines between subplans by using the flattened, filtered list.
                f.write(final_plan_text)
            
            
            # Perform grounding only on the full aggregated plan (not on individual subplans)
            # Translate final plan for VAL
            translated_final_plan_path = os.path.join(results_dir, "translated_plan_final.txt")
            translate_plan(final_plan_file_path, translated_final_plan_path)

            # Use VAL to validate the final plan and attempt grounding
            self.current_phase = "FINAL:VALIDATION"
            val_successful_full, val_log_full = VAL_validate(domain_pddl_path, problem_pddl_path, translated_final_plan_path)
            logger.info(f"Final VAL validation result: {val_successful_full}")
            logger.debug(val_log_full)

            self.current_phase = "FINAL:VAL:GROUNDING"
            val_ground_successful_full, val_ground_log_full = VAL_ground(domain_pddl_path, problem_pddl_path)
            logger.info(f"Final VAL grounding result: {val_ground_successful_full}")
            logger.debug(val_ground_log_full)
            
            # Save VAL grounding log to file
            val_grounding_log_path = save_log_file(val_ground_log_full, "VAL_grounding", logs_dir)

            # Default grounding_successful reflects full VAL grounding
            grounding_successful = val_successful_full and val_ground_successful_full

            save_statistics(
                dir=results_dir,
                workflow_iteration=0,
                phase=self.current_phase,
                VAL_grounding_log_path=val_grounding_log_path
            )

            # If VAL grounding succeeded and scene-graph grounding is enabled, try grounding in SG
            scene_graph_grounding_log = None
            grounding_success_percentage = None
            if grounding_successful and self.ground_in_sg:
                self.current_phase = "FINAL:SCENE_GRAPH:GROUNDING"
                logger.info("Performing scene-graph grounding on the aggregated plan")
                extracted_locations_dictionary = convert_JSON_to_locations_dictionary(scene_qry)
                extracted_locations_dictionary_file_path = os.path.join(results_dir, "extracted_locations_dictionary.json")
                with open(extracted_locations_dictionary_file_path, "w") as f:
                    json.dump(extracted_locations_dictionary, f, indent=4)

                grounding_success_percentage, grounding_error_log = verify_groundability_in_scene_graph(
                    final_plan,
                    graph=scene_qry,
                    domain_file_path=domain_pddl_path,
                    problem_dir=results_dir,
                    move_action_str="move_to",
                    location_relation_str="at",
                    location_type_str="room",
                    initial_robot_location=initial_robot_location,
                    pddlgym_environment=None,
                    locations_dictionary=extracted_locations_dictionary
                )
                logger.info(f"Final scene-graph grounding percentage: {grounding_success_percentage}")
                logger.debug(grounding_error_log)
                
                # Save grounding log to file
                grounding_log_path = save_log_file(grounding_error_log, "grounding", logs_dir)

                if grounding_success_percentage is not None:
                    grounding_successful = grounding_success_percentage == 1

                save_statistics(
                    dir=results_dir,
                    workflow_iteration=0,
                    phase=self.current_phase,
                    grounding_success_percentage=grounding_success_percentage,
                    scene_graph_grounding_log=grounding_error_log
                )

        else:
            logger.info("Planning failed, skipping grounding")
            # Planning failed completely
            save_statistics(
                dir=results_dir,
                workflow_iteration=0,
                phase=self.current_phase,
                plan_successful=False
            )
            grounding_successful = False

        # compute totals for subproblem planning
        total_sp_planning_time = sum(times) if times else 0.0
        total_sp_planning_nodes = sum(nodes) if nodes else 0
        total_sp_planning_costs = sum(costs) if costs else 0

        # number of successful subplans
        num_successful_subplans = sum(1 for p in (subplans or []) if p and p != "")

        return domain_pddl_path, scene_qry, problem_pddl_path, subgoal_pddl_list, \
            planning_successful, grounding_successful, subplans, "", "", num_successful_subplans, d_time, pr_time, p_time, dp_time, total_llm_time, times, nodes, costs, exit_code_decomp, total_sp_planning_time, total_sp_planning_nodes, total_sp_planning_costs, val_validation_log_path, val_grounding_log_path, planning_log_path, grounding_log_path


    def _run_and_log_pipeline(self, task_name, scene_name, problem_id, results_problem_dir,
                            domain_file_path, domain_description, scene_graph_file_path,
                            csv_filepath):
        '''
        Runs the DELTA pipeline for a specific task and logs the results.
            - task_name: Name of the task (e.g., "dining_setup").
            - scene_name: Name of the scene (e.g., "Allensville").
            - problem_id: Identifier for the problem instance. (e.g. "problem_1")
            - results_problem_dir: Directory where results will be saved.
            - domain_pddl_path: File path to the domain file (PDDL).
            - domain_description: contains "actions", "objects"
            - scene_graph_file_path: Path to the scene graph file (.npz).
            - csv_filepath: Path to the CSV file where results will be logged.
        '''
    
        try:
            results = self._delta_planning(
                domain_name=task_name,
                goal_file_path=os.path.join(results_problem_dir, "task.txt"),
                initial_location_file_path=os.path.join(results_problem_dir, "init_loc.txt"),
                scene_graph_file_path=scene_graph_file_path,
                domain_pddl_path=domain_file_path,
                results_dir=results_problem_dir,
                domain_description=domain_description,
                )
            (final_domain_pddl_path, final_pruned_scene_graph, final_problem_file_path, 
            final_subgoals_file_paths, planning_successful, grounding_successful, 
            subplans, failure_stage, failure_reason, completed_sp, d_time, pr_time, p_time, dp_time, total_llm_time, times, nodes, costs, exit_code_decomp, total_sp_planning_time, total_sp_planning_nodes, total_sp_planning_costs, val_validation_log_path, val_grounding_log_path, planning_log_path, grounding_log_path) = results
            
            plan_length = 0
            if planning_successful and ((self.ground_in_sg and grounding_successful) or not self.ground_in_sg):
                final_plan_file_path = os.path.join(results_problem_dir, "plan_final.txt")
                
                # Concatenate all subplans and write the resulting plan
                with open(final_plan_file_path, "w") as f:
                    if subplans:  # Check if subplans is not None and not empty
                        for subplan in subplans:
                            if subplan:  # Check if subplan is not None and not empty
                                for action in subplan:
                                    f.write(str(action) + ",\n")
                                    plan_length += 1
                            
                # Save final generated domain
                with open(final_domain_pddl_path, "r") as f:
                    final_generated_domain = f.read()
                save_file(final_generated_domain, os.path.join(results_problem_dir, "domain_final.pddl"))
                
                # Save results to CSV
                with open(csv_filepath, mode="a", newline='') as f:
                    writer = csv.writer(f, delimiter='|')
                    # compute totals for subproblem planning
                    total_sp_planning_time = sum(times) if times else 0.0
                    total_sp_planning_nodes = sum(nodes) if nodes else 0
                    total_sp_planning_costs = sum(costs) if costs else 0

                    writer.writerow([
                        task_name, scene_name, problem_id, planning_successful, grounding_successful,
                        plan_length, len(subplans) if subplans else 0, completed_sp if completed_sp is not None else 0, d_time, pr_time, 
                        p_time, dp_time, total_llm_time, 
                        str(times) if times else "[]", str(nodes) if nodes else "[]", str(costs) if costs else "[]", exit_code_decomp if exit_code_decomp is not None else "",
                        total_sp_planning_time, total_sp_planning_nodes, total_sp_planning_costs,
                        "", "",  # failure_stage, failure_reason
                        val_validation_log_path, val_grounding_log_path, planning_log_path, grounding_log_path
                    ])
                
            else:
                with open(csv_filepath, mode="a", newline='') as f:
                    writer = csv.writer(f, delimiter='|')
                    total_sp_planning_time = sum(times) if times else 0.0
                    total_sp_planning_nodes = sum(nodes) if nodes else 0
                    total_sp_planning_costs = sum(costs) if costs else 0

                    writer.writerow([
                        task_name, scene_name, problem_id, planning_successful, grounding_successful,
                        "", "", "", d_time, pr_time, p_time, dp_time, total_llm_time,
                        str(times) if times else "[]", str(nodes) if nodes else "[]", str(costs) if costs else "[]", exit_code_decomp if exit_code_decomp is not None else "",
                        total_sp_planning_time, total_sp_planning_nodes, total_sp_planning_costs,
                        str(failure_stage).replace('\n', ' ').replace('\r', ''),
                        str(failure_reason).replace('\n', ' ').replace('\r', ''),
                        val_validation_log_path if 'val_validation_log_path' in locals() else "",
                        val_grounding_log_path if 'val_grounding_log_path' in locals() else "",
                        planning_log_path if 'planning_log_path' in locals() else "",
                        grounding_log_path if 'grounding_log_path' in locals() else ""
                    ])
                    
            if planning_successful and grounding_successful:
                logger.info("DELTA pipelines successful")
                logger.debug(planning_successful)
                logger.debug(grounding_successful)
            else:
                logger.info("DELTA pipelines completed with issues")
                logger.debug(planning_successful)
                logger.debug(grounding_successful)
                
        except Exception as e:
            traceback.print_exc()
            exception_str = str(e).strip().replace('\n', ' ').replace('\r', '').replace('\t', ' ')
            
            logger.warning(f"Encountered problem: {exception_str}")

            with open(csv_filepath, mode="a", newline='') as f:
                writer = csv.writer(f, delimiter='|')
                writer.writerow([task_name, scene_name, problem_id, False, False, "", "", "", "", "", "", "", "", "[]", "[]", "[]", "", "Exception", exception_str, "", "", "", ""])
            
            # Save the exception to statistics.json
            save_statistics(
                dir=results_problem_dir,
                workflow_iteration=0,
                phase=self.current_phase,
                exception=e
            )