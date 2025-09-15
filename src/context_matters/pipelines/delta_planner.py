import os
from pathlib import Path
import pddlgym
from pddlgym_planners.fd import FD
from src.context_matters.logger_cfg import logger
import shutil
import subprocess
import traceback

import src.context_matters.utils.pddl as pddl_utils
import re
import time
from src.context_matters.planner import initialize_pddl_environment, plan_with_output

PLANNER = "./downward/fast-downward.py "
ALIAS = "--alias seq-opt-lmcut "
MAX_ERR_MSG_LEN = 100
PDDLGYM_PATH = os.path.dirname(pddlgym.__file__)


def replace_pddl_goal_string_based(pddl_file_path, new_goal):
    """Replace PDDL goal section using string search, not comment markers"""
    with open(pddl_file_path, "r") as f:
        pddl_content = f.read()
    
    goal_start = pddl_content.index("(:goal")
    # Find the end of the goal section by counting parentheses
    paren_count = 0
    goal_end = goal_start
    for i, char in enumerate(pddl_content[goal_start:], goal_start):
        if char == '(':
            paren_count += 1
        elif char == ')':
            paren_count -= 1
            if paren_count == 0:
                goal_end = i + 1
                break
    
    # Replace the entire goal section
    updated_pddl = pddl_content[:goal_start] + new_goal + pddl_content[goal_end:]
    
    with open(pddl_file_path, "w") as f:
        f.write(updated_pddl)


def SEARCH_CONFIG(
    mt): return "--search 'astar(lmcut(), max_time={})' ".format(mt)


def query(domain_path: str, problem_path: str, plan_file: str, print_plan: False, max_time: float = 120):
    plan_file = os.path.join(os.getcwd(), plan_file)
    command = PLANNER + "--plan-file {} ".format(plan_file) +\
        domain_path + " " + problem_path + " " + SEARCH_CONFIG(max_time)

    # Execute planner
    print("Planning...")
    p = subprocess.Popen(command, stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE, shell=True)
    (output, err) = p.communicate()
    p.wait()

    exit_code = 0
    cost = 0
    plan_time = 0.
    plan = None
    err_msg = err.decode()
    if "Solution found" in str(output):
        for line in str(output).split("\\n"):
            if "Plan cost: " in line:
                cost = int(line.strip().split(" ")[-1])
            if "Planner time: " in line:
                plan_time = float(line.strip().split(" ")[-1].replace("s", ""))
        print("Found solution in {} with cost {}".format(plan_time, cost))
        # Read plan file
        if os.path.isfile(plan_file):
            with open(plan_file, "r") as pf:
                plan = pf.read()
        if print_plan:
            print("Task plan: \n{}".format(plan))
        exit_code = 1
    else:
        # TODO: error handling with exit code
        print(err_msg)
        if "Time limit reached" in err_msg or "Time limit reached" in str(output):
            exit_code = 3
        elif "Argument" in err_msg and "not in params" in err_msg:
            exit_code = 4
        elif "Undeclared predicate" in err_msg or ("Predicate" in err_msg and "not defined" in err_msg):
            exit_code = 5
        else:
            exit_code = 2
        print("Could not find solution")

    return plan, cost, plan_time, exit_code, err_msg


def export_domain_to_pddlgym(domain: str, src_domain_file: str):
    dst_domain_file = os.path.join(
        PDDLGYM_PATH, "pddl/{}.pddl".format(domain))
    if os.path.isfile(dst_domain_file):
        os.remove(dst_domain_file)
    shutil.copyfile(src_domain_file, dst_domain_file)


def export_problem_to_pddlgym(domain: str, src_problem_file: str, p_idx: str, clear_dir=False):
    dst_problem_path = os.path.join(
        PDDLGYM_PATH, "pddl/{}/".format(domain))
    Path(dst_problem_path).mkdir(parents=True, exist_ok=True)
    if clear_dir:
        for f in os.listdir(dst_problem_path):
            os.remove(os.path.join(dst_problem_path, f))
    dst_problem_file = os.path.join(
        dst_problem_path, "problem{}.pddl".format(p_idx))
    shutil.copyfile(src_problem_file, dst_problem_file)


def register_new_pddlgym_env(new_domain: str):
    new_env = (new_domain, {'operators_as_actions': True,
               'dynamic_action_space': True})
    new_line = "\t\t" + str(new_env) + ",\n"

    with open(os.path.join(PDDLGYM_PATH, "__init__.py"), "r+") as file:
        lines = file.readlines()
        if new_line not in lines:
            print("Registering new PDDLGym environment '{}'...".format(new_domain))
            for i, line in enumerate(lines):
                if "for env_name, kwargs in [" in line:
                    lines.insert(i + 1, new_line)
                    file.seek(0)
                    file.writelines(lines)
            print("Registered new PDDLGym environment '{}' at {}!".format(new_domain, os.path.join(PDDLGYM_PATH, "__init__.py")))
        else:
            print("PDDLGym environment '{}' already registered at {}!".format(new_domain, os.path.join(PDDLGYM_PATH, "__init__.py")))


def query_pddlgym(domain: str, p_idx: int = 0, max_time: float = 120):
    plan = None
    cost = 0
    node = 0
    time = 0.
    exit_code = 0
    print("Planning with undecomposed problem...")

    try:
        fd_planner = FD()
        env = pddlgym.make("PDDLEnv{}-v0".format(domain.capitalize()))
        env.fix_problem_index(p_idx)
        state, _ = env.reset()
        plan = fd_planner(env.domain, state, timeout=max_time)
        print("Plan: "+str(plan))
        statistic = fd_planner.get_statistics()
        print("Statistic: "+str(statistic))
        time = statistic["total_time"]
        cost = statistic["plan_cost"]
        node = statistic["num_node_expansions"]
        for act in plan:
            state, reward, done, truncated, info = env.step(act)
        print("Found solution in {}s with cost {}".format(time, cost))
        exit_code = 1
    except Exception as err:
        # TODO: error handling with exit code
        traceback.print_exc()
        print("ERROR: "+str(err))
        err_msg = str(err)
        print(err_msg)
        if "Planning timed out" in err_msg:
            exit_code = 3
        elif "Argument" in err_msg and "not in params" in err_msg:
            exit_code = 4
        elif "Undeclared predicate" in err_msg or ("Predicate" in err_msg and "not defined" in err_msg):
            exit_code = 5
        else:
            exit_code = 2
        print("Could not find solution!", err_msg if len(
            err_msg) <= MAX_ERR_MSG_LEN else "")

    return [p.pddl_str() for p in plan] if exit_code == 1 else None, time, node, cost, exit_code


def query_pddlgym_decompose(domain: str, subgoal_pddl_list: list, save_path: str = None, max_time: float = 120):
    digits = 2 if len(str(len(subgoal_pddl_list))) > 1 else 1
    d_file = os.path.join(PDDLGYM_PATH, "pddl/{}.pddl".format(domain))
    p_0_file = os.path.join(
        PDDLGYM_PATH, "pddl/{}/problem{}.pddl".format(domain, str(0).zfill(digits)))
    assert os.path.isfile(d_file), "Missing domain file in PDDLGym directory!"
    assert os.path.isfile(
        p_0_file), "Missing initial problem file in PDDLGym directory!"
    assert len([name for name in os.listdir(os.path.join(PDDLGYM_PATH, "pddl/{}/".format(domain)))
                if os.path.isfile(os.path.join(PDDLGYM_PATH, "pddl/{}/{}".format(domain, name)))]) == 1, \
        "PDDLGym problem directory contains more than 1 file!"
    print("Planning with decomposed problems...")

    plans = []
    times = []
    nodes = []
    costs = []
    final_state_list = []
    exit_code = 0
    fd_planner = FD()
    completed_sp = 0

    for idx, sgp in enumerate(subgoal_pddl_list, start=1):
        # Create new sub-problem file with new states and sub-goal
        sp_file = os.path.join(
            PDDLGYM_PATH, "pddl/{}/problem{}.pddl".format(domain, str(idx).zfill(digits)))

        shutil.copyfile(p_0_file, sp_file)
        try:
            replace_pddl_goal_string_based(sp_file, sgp)
            if idx > 1:
                pddl_utils.set_pddl_problem_init(sp_file, final_state_list)
        except Exception as err:
            exit_code = 6
            print("Error when writing PDDL states!", str(err))
            traceback.print_exc()

        # Log sub-problem file
        if save_path is not None:
            save_file = os.path.join(
                save_path, "p{}.pddl".format(str(idx).zfill(digits)))
            if not os.path.exists(save_file):
                os.makedirs(os.path.dirname(save_file), exist_ok=True)
            shutil.copyfile(sp_file, save_file)

        # Planning with sub-problem
        plan, statistic = None, None
        try:
            env = pddlgym.make("PDDLEnv{}-v0".format(domain.capitalize()))
            env.fix_problem_index(idx)
            state, _ = env.reset()
            plan = fd_planner(env.domain, state, timeout=max_time)
            statistic = fd_planner.get_statistics()
            print("Subgoal: {}/{}, Plan time: {:.2f}s, Plan cost: {}".format(
                idx, len(subgoal_pddl_list), statistic["total_time"], statistic["plan_cost"]))
            for act in plan:
                state, reward, done, truncated, info = env.step(act)
            final_state_list = sorted(
                [lit.pddl_str() for lit in state.literals if not lit.is_negative])
            completed_sp += 1
            exit_code = 1
        except Exception as err:
            # TODO: error handling with exit code
            err_msg = str(err)
            print(err_msg)
            if "Planning timed out" in err_msg:
                exit_code = 3
            elif "Argument" in err_msg and "not in params" in err_msg:
                exit_code = 4
            elif "Undeclared predicate" in err_msg or ("Predicate" in err_msg and "not defined" in err_msg):
                exit_code = 5
            else:
                exit_code = 2
            print("Could not find solution!", err_msg if len(
                err_msg) <= MAX_ERR_MSG_LEN else "")
            break
        plans.append([p.pddl_str() for p in plan] if plan is not None else "")
        times.append(statistic["total_time"] if statistic is not None else 0.)
        nodes.append(statistic["num_node_expansions"]
                     if statistic is not None else 0)
        costs.append(statistic["plan_cost"] if statistic is not None else 0)

    print("Total plan time of all sub-problems: {:.2f}s".format(sum(times)))
    print("Total cost: {}".format(sum(costs)))
    return plans, times, nodes, costs, exit_code, completed_sp


def query_pddlgym_decompose_new(domain: str, domain_file_path: str, base_problem_file: str, 
                               subgoal_pddl_list: list, save_path: str = None, max_time: float = 120,
                               simulate_plan: bool = True):
    """
    PDDLGym-based decomposition planning with optional simulation.
    """
    plans = []
    times = []
    nodes = []
    costs = []
    exit_code = 0
    completed_sp = 0
    final_state_list = []  # For simulate_plan mode

    digits = 2 if len(str(len(subgoal_pddl_list))) > 1 else 1
    
    # Ensure the domain and base problem are placed in the pddlgym path
    d_file = os.path.join(PDDLGYM_PATH, "pddl/{}.pddl".format(domain))
    dst_problem_dir = os.path.join(PDDLGYM_PATH, "pddl/{}/".format(domain))
    Path(dst_problem_dir).mkdir(parents=True, exist_ok=True)
    p_0_file = os.path.join(dst_problem_dir, "problem{}.pddl".format(str(0).zfill(digits)))
    shutil.copyfile(base_problem_file, p_0_file)

    # Initialize environment for non-simulate mode
    env = None
    if not simulate_plan:
        env = pddlgym.make("PDDLEnv{}-v0".format(domain.capitalize()))
        env.fix_problem_index(0)
        initial_state, _ = env.reset()

    for idx, sgp in enumerate(subgoal_pddl_list, start=1):
        try:
            sp_file = os.path.join(PDDLGYM_PATH, "pddl/{}/problem{}.pddl".format(domain, str(idx).zfill(digits)))
            shutil.copyfile(p_0_file, sp_file)
            # Set goal using string-based replacement
            replace_pddl_goal_string_based(sp_file, sgp)
            
            # Handle state propagation for both modes
            if idx > 1:
                if simulate_plan:
                    # delta_pipeline style: use final_state_list from previous execution
                    pddl_utils.set_pddl_problem_init(sp_file, final_state_list)
                else:
                    # delta_plus style: extract current state from environment
                    current_state = env.get_state()
                    state_literals = sorted([lit.pddl_str() for lit in current_state.literals if not lit.is_negative])
                    pddl_utils.set_pddl_problem_init(sp_file, state_literals)

            # Optionally save subproblem
            if save_path is not None:
                save_file = os.path.join(save_path, "p{}.pddl".format(str(idx).zfill(digits)))
                if not os.path.exists(save_file):
                    os.makedirs(os.path.dirname(save_file), exist_ok=True)
                shutil.copyfile(sp_file, save_file)

            # Planning with pddlgym FD planner
            fd_planner = FD()
            current_env = pddlgym.make("PDDLEnv{}-v0".format(domain.capitalize()))
            current_env.fix_problem_index(idx)
            state, _ = current_env.reset()
            plan = fd_planner(current_env.domain, state, timeout=max_time)
            statistic = fd_planner.get_statistics()

            if simulate_plan:
                # Execute plan to get final state for next iteration (delta_pipeline style)
                for act in plan:
                    state, reward, done, truncated, info = current_env.step(act)
                final_state_list = sorted([lit.pddl_str() for lit in state.literals if not lit.is_negative])
            else:
                # Don't execute, just update env state to the current goal (delta_plus style)
                # Update the env reference for next iteration
                env = current_env

            completed_sp += 1
            exit_code = 1

        except Exception as err:
            # Error handling similar to existing functions
            err_msg = str(err)
            print(err_msg)
            if "Planning timed out" in err_msg:
                exit_code = 3
            elif "Argument" in err_msg and "not in params" in err_msg:
                exit_code = 4
            elif "Undeclared predicate" in err_msg or ("Predicate" in err_msg and "not defined" in err_msg):
                exit_code = 5
            else:
                exit_code = 2
            print("Could not find solution!", err_msg if len(err_msg) <= MAX_ERR_MSG_LEN else "")
            break

        plans.append([p.pddl_str() for p in plan] if plan is not None else "")
        times.append(statistic["total_time"] if statistic is not None and "total_time" in statistic else 0.)
        nodes.append(statistic["num_node_expansions"] if statistic is not None and "num_node_expansions" in statistic else 0)
        costs.append(statistic["plan_cost"] if statistic is not None and "plan_cost" in statistic else 0)

    print("Total plan time of all sub-problems: {:.2f}s".format(sum(times)))
    print("Total cost: {}".format(sum(costs)))
    return plans, times, nodes, costs, exit_code, completed_sp


def query_pddlenv_decompose_new(domain: str, domain_file_path: str, base_problem_file: str,
                               subgoal_pddl_list: list, results_dir: str, save_path: str = None, 
                               max_time: float = 120, simulate_plan: bool = True):
    """
    PDDLEnv-based decomposition planning with optional simulation.
    Uses delta_plus_pipeline directory structure for non-simulate mode.
    """
    plans = []
    times = []
    nodes = []
    costs = []
    exit_code = 0
    completed_sp = 0
    final_state_list = []  # For simulate_plan mode

    digits = 2 if len(str(len(subgoal_pddl_list))) > 1 else 1

    # Initialize environment for non-simulate mode
    env = None
    if not simulate_plan:
        # Create a proper temp directory for PDDLEnv initialization  
        temp_init_dir = os.path.join(os.getcwd(), f"_pddl_init_{domain}")
        os.makedirs(temp_init_dir, exist_ok=True)
        temp_init_problem = os.path.join(temp_init_dir, "problem.pddl")
        shutil.copyfile(base_problem_file, temp_init_problem)
        env, initial_state = initialize_pddl_environment(domain_file_path, temp_init_dir)

    print("A")

    for idx, sgp in enumerate(subgoal_pddl_list, start=1):
        try:
            # Helper function for extracting PDDL predicates from environment state
            # Note: Objects are static in PDDL and should not change during planning
            def extract_pddl_predicates(obs):
                # Extract PDDL predicates only (objects remain unchanged)
                pddl_predicates_str = "(:init\n"
                for literal in obs.literals:
                    predicate_name = literal.predicate.name
                    predicate_variables = [var.split(":")[0] for var in literal.variables]
                    pddl_predicates_str += f"    ({predicate_name} {' '.join(predicate_variables)})\n"
                pddl_predicates_str += ")\n\n"
                
                return pddl_predicates_str

            # Helper function for goal replacement using string-based approach
            def replace_pddl_goal(pddl_content, new_goal):
                """Replace PDDL goal section using string search, not comment markers"""
                goal_start = pddl_content.index("(:goal")
                # Find the end of the goal section by counting parentheses
                paren_count = 0
                goal_end = goal_start
                for i, char in enumerate(pddl_content[goal_start:], goal_start):
                    if char == '(':
                        paren_count += 1
                    elif char == ')':
                        paren_count -= 1
                        if paren_count == 0:
                            goal_end = i + 1
                            break
                
                # Replace the entire goal section
                return pddl_content[:goal_start] + new_goal + pddl_content[goal_end:]

            if simulate_plan:
                # Use the same directory structure as non-simulate mode for consistency
                sub_goal_dir = os.path.join(results_dir, f"sub_goal_{idx}")
                os.makedirs(sub_goal_dir, exist_ok=True)
                sp_file = os.path.join(sub_goal_dir, f"sub_goal_{idx}.pddl")
                
                if idx == 1:
                    # First iteration: copy base problem
                    shutil.copyfile(base_problem_file, sp_file)
                else:
                    # Use previous environment state for consistent extraction
                    # Copy base problem first, then update only the init section
                    shutil.copyfile(base_problem_file, sp_file)
                    
                    if env is not None:
                        current_obs = env.get_state()
                        pddl_predicates_str = extract_pddl_predicates(current_obs)
                        
                        # Read the copied file and replace only the init section
                        with open(sp_file, "r") as f:
                            base_pddl = f.read()
                        
                        # Replace only init section (objects remain unchanged)
                        init_start = base_pddl.index("(:init")
                        init_end = base_pddl.index("(:goal")
                        
                        updated_pddl = (base_pddl[:init_start] + 
                                       pddl_predicates_str + 
                                       base_pddl[init_end:])
                        
                        with open(sp_file, "w") as f:
                            f.write(updated_pddl)
                
                print("B")

                # Set the goal for this subproblem using string-based replacement
                with open(sp_file, "r") as f:
                    current_pddl = f.read()
                updated_pddl = replace_pddl_goal(current_pddl, sgp)
                with open(sp_file, "w") as f:
                    f.write(updated_pddl)
                    
                # Initialize env and plan
                current_env, initial_obs = initialize_pddl_environment(domain_file_path, sub_goal_dir)
                fd_planner = FD()
                plan = fd_planner(current_env.domain, initial_obs, timeout=max_time)
                statistic = fd_planner.get_statistics()
                
                # Execute plan to get final state and update env reference
                for act in plan:
                    initial_obs, reward, done, truncated, info = current_env.step(act)
                env = current_env  # Update env reference for next iteration
                
            else:
                # delta_plus style: follow delta_plus_pipeline directory structure
                sub_goal_dir = os.path.join(results_dir, f"sub_goal_{idx}")
                os.makedirs(sub_goal_dir, exist_ok=True)
                sp_file = os.path.join(sub_goal_dir, f"sub_goal_{idx}.pddl")
                
                if idx == 1:
                    # First iteration: copy base problem
                    shutil.copyfile(base_problem_file, sp_file)
                else:
                    # For subsequent iterations, start with base problem and then update with current state
                    shutil.copyfile(base_problem_file, sp_file)
                    
                    # Use existing env state and update problem file (delta_plus style)
                    if env is not None:
                        current_obs = env.get_state()
                        pddl_predicates_str = extract_pddl_predicates(current_obs)

                        # Read the just-copied file and modify only the init section
                        with open(sp_file, "r") as f:
                            sub_goal_pddl = f.read()
                        
                        # Replace only init section (objects remain unchanged)
                        init_start = sub_goal_pddl.index("(:init")
                        init_end = sub_goal_pddl.index("(:goal")
                        
                        sub_goal_pddl = (sub_goal_pddl[:init_start] + 
                                       pddl_predicates_str + 
                                       sub_goal_pddl[init_end:])
                        
                        with open(sp_file, "w") as f:
                            f.write(sub_goal_pddl)
                
                # Set the goal for this subproblem using string-based replacement
                with open(sp_file, "r") as f:
                    current_pddl = f.read()
                updated_pddl = replace_pddl_goal(current_pddl, sgp)
                with open(sp_file, "w") as f:
                    f.write(updated_pddl)
                
                # Use plan_with_output like delta_plus_pipeline
                output_plan_file_path = os.path.join(sub_goal_dir, f"plan_{idx}.txt")
                plan, pddlenv_error_log, planner_error_log, statistic = plan_with_output(
                    domain_file_path, sub_goal_dir, output_plan_file_path, env=None, timeout=int(max_time))
                
                # Store error logs
                pddlenv_error_logs.append(pddlenv_error_log)
                planner_error_logs.append(planner_error_log)
                
                # Initialize new env for next iteration
                env, initial_obs = initialize_pddl_environment(domain_file_path, sub_goal_dir)

            # Save if requested
            if save_path is not None:
                save_file = os.path.join(save_path, "p{}.pddl".format(str(idx).zfill(digits)))
                if not os.path.exists(save_file):
                    os.makedirs(os.path.dirname(save_file), exist_ok=True)
                shutil.copyfile(sp_file, save_file)

            completed_sp += 1
            exit_code = 1

        except Exception as err:
            # Error handling similar to existing functions
            err_msg = str(err)
            print(err_msg)
            if "Planning timed out" in err_msg:
                exit_code = 3
            elif "Argument" in err_msg and "not in params" in err_msg:
                exit_code = 4
            elif "Undeclared predicate" in err_msg or ("Predicate" in err_msg and "not defined" in err_msg):
                exit_code = 5
            else:
                exit_code = 2
            print("Could not find solution!", err_msg if len(err_msg) <= MAX_ERR_MSG_LEN else "")
            break

        # Handle plan format conversion for different return types
        if isinstance(plan, list):
            plans.append([str(p) for p in plan] if plan is not None else "")
        else:
            plans.append(plan if plan is not None else "")
            
        times.append(statistic["total_time"] if statistic is not None and "total_time" in statistic else 0.)
        nodes.append(statistic["num_node_expansions"] if statistic is not None and "num_node_expansions" in statistic else 0)
        costs.append(statistic["plan_cost"] if statistic is not None and "plan_cost" in statistic else 0)

    print("Total plan time of all sub-problems: {:.2f}s".format(sum(times)))
    print("Total cost: {}".format(sum(costs)))
    return plans, times, nodes, costs, exit_code, completed_sp


def query_pddlenv(domain: str, domain_file_path: str, base_problem_file: str,
                  subgoal_pddl_list: list, results_dir: str, save_path: str = "", 
                  max_time: float = 120):
    """
    PDDLEnv-based decomposition planning with simulation.
    This is the simplified version that always uses PDDLEnv with simulate_plan=True behavior.
    """
    plans = []
    times = []
    nodes = []
    costs = []
    exit_code = 0
    completed_sp = 0
    pddlenv_error_logs = []
    planner_error_logs = []
    
    digits = 2 if len(str(len(subgoal_pddl_list))) > 1 else 1

    # Initialize environment for the first subgoal
    env = None
    
    for idx, sgp in enumerate(subgoal_pddl_list, start=1):
        logger.info(f"Processing subgoal {idx}/{len(subgoal_pddl_list)}")
        try:
            # Helper function for extracting PDDL predicates from environment state
            def extract_pddl_predicates(obs):
                pddl_predicates_str = "(:init\n"
                for literal in obs.literals:
                    predicate_name = literal.predicate.name
                    predicate_variables = [var.split(":")[0] for var in literal.variables]
                    pddl_predicates_str += f"    ({predicate_name} {' '.join(predicate_variables)})\n"
                pddl_predicates_str += ")\n\n"
                return pddl_predicates_str

            # Helper function for goal replacement using string-based approach
            def replace_pddl_goal(pddl_content, new_goal):
                goal_start = pddl_content.index("(:goal")
                paren_count = 0
                goal_end = goal_start
                for i, char in enumerate(pddl_content[goal_start:], goal_start):
                    if char == '(':
                        paren_count += 1
                    elif char == ')':
                        paren_count -= 1
                        if paren_count == 0:
                            goal_end = i + 1
                            break
                return pddl_content[:goal_start] + new_goal + pddl_content[goal_end:]

            # Create subgoal directory structure
            sub_goal_dir = os.path.join(results_dir, f"sub_goal_{idx}")
            os.makedirs(sub_goal_dir, exist_ok=True)
            sp_file = os.path.join(sub_goal_dir, f"sub_goal_{idx}.pddl")
            
            if idx == 1:
                # First iteration: copy base problem
                shutil.copyfile(base_problem_file, sp_file)
            else:
                # Copy base problem first, then update only the init section with current env state
                shutil.copyfile(base_problem_file, sp_file)
                
                if env is not None:
                    current_obs = env.get_state()
                    pddl_predicates_str = extract_pddl_predicates(current_obs)
                    
                    # Read the copied file and replace only the init section
                    with open(sp_file, "r") as f:
                        base_pddl = f.read()
                    
                    # Replace only init section (objects remain unchanged)
                    init_start = base_pddl.index("(:init")
                    init_end = base_pddl.index("(:goal")
                    
                    updated_pddl = (base_pddl[:init_start] + 
                                   pddl_predicates_str + 
                                   base_pddl[init_end:])
                    
                    with open(sp_file, "w") as f:
                        f.write(updated_pddl)
            
            # Set the goal for this subproblem using string-based replacement
            with open(sp_file, "r") as f:
                current_pddl = f.read()
            updated_pddl = replace_pddl_goal(current_pddl, sgp)
            with open(sp_file, "w") as f:
                f.write(updated_pddl)
                
            # Initialize env and plan
            current_env, initial_obs = initialize_pddl_environment(domain_file_path, sub_goal_dir)
            fd_planner = FD()
            plan = fd_planner(current_env.domain, initial_obs, timeout=max_time)
            statistic = fd_planner.get_statistics()
            
            # For FD direct calls, no error logs are available, add empty placeholders
            pddlenv_error_logs.append("")
            planner_error_logs.append("")
            
            # Execute plan to get final state and update env reference for next iteration
            for act in plan:
                initial_obs, reward, done, truncated, info = current_env.step(act)
            env = current_env  # Update env reference for next iteration
            
            # Save the plan to the subproblem directory so each subproblem has its plan file
            try:
                plan_file_path = os.path.join(sub_goal_dir, f"plan_{idx}.txt")
                with open(plan_file_path, "w") as pf:
                    if isinstance(plan, list):
                        for p in plan:
                            # prefer p.pddl_str() when available for consistency with other code paths
                            action_str = p.pddl_str() if hasattr(p, "pddl_str") else str(p)
                            pf.write(action_str + "\n")
                    else:
                        pf.write(str(plan))
                logger.info(f"Saved plan for subgoal {idx} to {plan_file_path}")
            except Exception as e:
                logger.warning(f"Could not write plan file for subgoal {idx} at {sub_goal_dir}: {e}")

            # Save if requested
            if save_path and save_path != "":
                save_file = os.path.join(save_path, "p{}.pddl".format(str(idx).zfill(digits)))
                if not os.path.exists(save_file):
                    os.makedirs(os.path.dirname(save_file), exist_ok=True)
                shutil.copyfile(sp_file, save_file)

            completed_sp += 1
            exit_code = 1

        except Exception as err:
            # Error handling similar to existing functions
            err_msg = str(err)
            print(err_msg)
            if "Planning timed out" in err_msg:
                exit_code = 3
            elif "Argument" in err_msg and "not in params" in err_msg:
                exit_code = 4
            elif "Undeclared predicate" in err_msg or ("Predicate" in err_msg and "not defined" in err_msg):
                exit_code = 5
            else:
                exit_code = 2
            print("Could not find solution!", err_msg if len(err_msg) <= MAX_ERR_MSG_LEN else "")
            break

        plans.append(plan)

        times.append(statistic["total_time"] if statistic is not None and "total_time" in statistic else 0.)
        nodes.append(statistic["num_node_expansions"] if statistic is not None and "num_node_expansions" in statistic else 0)
        costs.append(statistic["plan_cost"] if statistic is not None and "plan_cost" in statistic else 0)

    print("Total plan time of all sub-problems: {:.2f}s".format(sum(times)))
    print("Total cost: {}".format(sum(costs)))
    return plans, times, nodes, costs, exit_code, completed_sp, pddlenv_error_logs, planner_error_logs


def _compute_plan(domain: str,
                  domain_file_path: str,
                  base_problem_file: str,
                  subgoal_pddl_list: list,
                  results_dir: str = "",
                  save_path: str = "",
                  max_time: float = 120,
                  use_pddlgym: bool = False,
                  simulate_plan: bool = True):
    """
    Compute plans for a list of subgoals. Supports two backends and two modes:
      - use_pddlgym: if True use the pddlgym directory/registration based flow; otherwise use PDDLEnv directly.
      - simulate_plan: if True, after planning each subproblem execute the plan to obtain the final state (as done in the original delta pipeline);
                       if False, do not execute plans and instead propagate the previous subgoal as the initial state for the next subproblem (as in delta_plus behaviour).

    Returns the same tuple as query_pddlgym_decompose: (plans, times, nodes, costs, exit_code, completed_sp)
    """

    # Sanity
    if subgoal_pddl_list is None or len(subgoal_pddl_list) == 0:
        return [], [], [], [], 0, 0

    # If use_pddlgym and simulate_plan is default True we can reuse existing function
    if use_pddlgym and simulate_plan:
        return query_pddlgym_decompose(domain, subgoal_pddl_list, save_path=save_path, max_time=max_time)

    # Delegate to the appropriate specialized method
    if use_pddlgym:
        return query_pddlgym_decompose_new(domain, domain_file_path, base_problem_file, 
                                          subgoal_pddl_list, save_path, max_time, simulate_plan)
    else:
        return query_pddlenv_decompose_new(domain, domain_file_path, base_problem_file,
                                          subgoal_pddl_list, results_dir, save_path, max_time, simulate_plan)


def validate(domain_file: str, problem_file: str, plan_file: str):
    command = "Validate -v " + domain_file + \
        " " + problem_file + " " + plan_file
    p = subprocess.Popen(command, stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE, shell=True)
    (output, err) = p.communicate()
    p.wait()

    if "Plan valid" in str(output):
        print("VAL: Plan valid!")
        return True, "Plan succeeded."
    else:
        print("VAL: Plan invalid!")
        repair_phrase = "Plan Repair Advice:"
        if repair_phrase in str(output):
            out_str = str(output)
            msg = out_str[out_str.index(repair_phrase) + len(repair_phrase):]
            msg, _ = msg.split("Failed plans:")
            msg = "NOTE: " + msg.strip()
        else:
            msg = "NOTE: The plan did not achieve the goal."
        return False, msg

if __name__ == "__main__":
    # Hardcoded configuration - adjust paths and values as needed
    domain = "my_domain"
    domain_file = "/DATA/context-matters/results/DeltaPipeline_GPTAgent_gendomain/20250903-142247/laundry/Kemblesville/problem_1/domain_final.pddl"       # <-- set to your domain file path
    problem_file = "/DATA/context-matters/results/DeltaPipeline_GPTAgent_gendomain/20250903-142247/laundry/Kemblesville/problem_1/problem.pddl"     # <-- set to your base problem file path

    # Example subgoals list (PDDL goal strings). Replace with your actual subgoal goals.
    subgoal_list = [
        "(:goal (and (is-clean shirt_21)))",
        "(:goal (and (is-clean sock_22)))",
        "(:goal (and (is-clean sock_23)))"
    ]

    results_dir = "./test_results3"   # results directory
    save_path = "./subproblems"      # optional: where to save generated subproblem files
    max_time = 60.0

    # Ensure directories exist
    os.makedirs(results_dir, exist_ok=True)
    if save_path:
        os.makedirs(save_path, exist_ok=True)

    print(f"\nTesting query_pddlenv with:")
    print(f"  Domain: {domain}")
    print(f"  Domain file: {domain_file}")
    print(f"  Problem file: {problem_file}")
    print(f"  Subgoals: {len(subgoal_list)}")
    print(f"  Max time: {max_time}s")
    print(f"  Results dir: {results_dir}")
    print(f"  Save path: {save_path}")

    start_time = time.time()
    plans, times, nodes, costs, exit_code, completed_sp, pddlenv_error_logs, planner_error_logs = query_pddlenv(
        domain=domain,
        domain_file_path=domain_file,
        base_problem_file=problem_file,
        subgoal_pddl_list=subgoal_list,
        results_dir=results_dir,
        save_path=save_path,
        max_time=max_time
    )
    total_time = time.time() - start_time

    print(f"\n=== RESULTS ===")
    print(f"Exit code: {exit_code}")
    print(f"Completed subproblems: {completed_sp}/{len(subgoal_list)}")
    print(f"Total execution time: {total_time:.2f}s")
    print(f"Total plan time: {sum(times):.2f}s")
    print(f"Total cost: {sum(costs)}")
    print(f"Total nodes expanded: {sum(nodes)}")

    print(f"\nPer-subgoal breakdown:")
    for i, (plan, plan_time, plan_nodes, plan_cost) in enumerate(zip(plans, times, nodes, costs), 1):
        plan_length = len(plan) if isinstance(plan, list) else (1 if plan else 0)
        print(f"  Subgoal {i}: {plan_length} actions, {plan_time:.2f}s, cost={plan_cost}, nodes={plan_nodes}")

    if exit_code == 1:
        print(f"\n✅ All subproblems solved successfully!")
    else:
        print(f"\n❌ Planning failed at subproblem {completed_sp + 1}")

    print(f"\nTest completed. Results saved to: {results_dir}")
    if save_path:
        print(f"Subproblem files saved to: {save_path}")
