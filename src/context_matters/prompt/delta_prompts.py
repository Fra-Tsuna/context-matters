# Adaptation of the DELTA Pipeline from the official code repository.
# Original paper: Liu, Yuchen, et al. "Delta: Decomposed efficient long-term robot task planning using large language models." 2025 IEEE International Conference on Robotics and Automation (ICRA). IEEE, 2025.
# Paper link: https://delta-llm.github.io/
# Based on the official code implementation by Y. Liu et al. https://github.com/boschresearch/DELTA

import itertools
import os
from src.context_matters.utils.pddl import (
    get_pddl_domain_actions,
    get_pddl_domain_predicates,
    get_pddl_domain_types
)




################ Domain Generation Prompt ################

def nl_2_pddl_domain(domain_exp: str, domain_qry_name: str, add_obj_exp: str = None, add_obj_qry: str = None,
                     add_act_exp: str = None, add_act_qry: str = None):
    content = "You are an excellent PDDL domain file generator. Given a description of object types, predicates, and action knowledge, you can use it to generate a PDDL domain description file."

    if add_obj_qry is None and add_act_qry is None:
        raise Exception(
            "Additional object and additional action cannot be both None!")
    info_add_obj_exp = ", {}". format(
        ", ".join(add_obj_exp)) if add_obj_exp is not None else ""
    info_add_act_exp = "and the following additional action(s): \n{}".format(
        "\n".join(add_act_exp)) if add_act_exp is not None else ""
    
    info_add_obj_qry = "there are the following new object type(s): {}.".format(
        ", ".join(["\n" + str(obj['type']) + ": " + str(obj['description']) for obj in add_obj_qry])) if add_obj_qry is not None else ""
    info_add_act_qry = "\nBesides the basic actions (move_to, pick, drop), there are the following additional action(s): \n{}".format(
        "\n".join(act['description'] for act in add_act_qry)) if add_act_qry is not None else ""

    prompt = f"""
    For example, a domain has the following object types: agent, room, item{info_add_obj_exp}. 
    The agent can perform the following basic actions:
    move_to(<agent>, <room_1>, <room_2>): <agent> goes from <room_1> to <room_2>. As result, <agent> will leave <room_1> and locates in <room_2>.
    pick(<agent>, <item>, <room>): <agent> picks up an <item> at <room>. <item> must be located in <room>, and <agent> state is hand-free. As result, <agent> state will change to loaded, and <item> left <room>.
    drop(<agent>, <item>, <room>): <agent> drops an <item> at <room>. <agent> is in <room> and has <item> in hand, and <agent> state is loaded. As result, <item> will locate in <room>, <agent> state will change to hand-free.
    {info_add_act_exp}
    
    A PDDL domain file describes the object types, the predicates, and the action knowledge (the preconditions and effects of an action).
    The corresponding PDDL domain file with respect to the previous actions looks like: \n```\n{domain_exp}\n```
    with the first line of code defining the name of the domain.
    
    Now in a new domain named "{domain_qry_name}", {info_add_obj_qry} {info_add_act_qry}\n
    Please provide a new PDDL domain file with respect to this new domain and define the domain name as "{domain_qry_name}" directly without further explanations. Please also keep the comments such as "; Begin actions", "; End actions" etc. in the domain file.
    """

    return content, prompt




################ Scene pruning prompt ################

def nl_prune_item(items_exp: dict, items_qry: dict, goal_exp: str, goal_qry: str, item_keep_exp: list, domain_exp: str = None, domain_qry: str = None):
    act_exp, act_qry = None, None
    if domain_exp is not None:
        act_exp = "and the corresponding action knowledge\n{}".format(
            get_pddl_domain_actions(domain_exp))
    if domain_qry is not None:
        act_qry = "and the new action knowledge\n{}".format(
            get_pddl_domain_actions(domain_qry))

    content = "You are an excellent assistant in pruning items. Given a list of items and a goal description, you can prune the item list by only keeping the relevant items."
    prompt = f"""
    Here is an example of a list of items: {items_exp}
    
    Given an example of a goal description: {goal_exp}, {act_exp} 
    the relevant items for accomplishing the goal are {item_keep_exp}.
    
    Now given a new list of items: {items_qry}
    and a new goal description: {goal_qry}, {act_qry}
    please provide a list of the relevent items from the new item list for accomplishing the new goal directly without further explanations, and keep the same data structure.
    """
    return content, prompt




################ Problem Generation Prompt ################

def sg_2_pddl_problem(domain_name_exp: str, domain_exp: str, problem_exp: str,
                      sg_exp: dict, sg_qry: dict, goal_exp: str, goal_qry: str,
                      domain_qry: str, domain_name_qry: str, initial_robot_location: str):
    content = "You are an excellent PDDL problem file generator. Given a scene graph representation of an environment, a PDDL domain file and a goal description, you can generate a PDDL problem file."
    prompt = f"""
    Here is an example of a scene graph in the form of a nested dictionary in Python:
    ```\n{sg_exp}\n```\n
    The top level keys are the name of the scene are the rooms.
    Each room contains a list of 'items' inside the rooms.
    The agent should be initially marked as free or not holding any object.
    The agent should be initially located in {initial_robot_location}

    Given a goal description e.g., {goal_exp}, and using the pre-defined object types, predicated in the PDDL domain example named {domain_name_exp}:
    ```\n{get_pddl_domain_types(domain_exp)}\n{get_pddl_domain_predicates(domain_exp)}\n```
    A corresponding PDDL problem file can be formulated as follows:
    ```\n{problem_exp}\n```
    The first line defines the name of the problem, usually the scene graph's name.
    The second line refers to the domain it based on.
    The "(:objects )" section lists all the items included in the scene graph with corresponding object types.
    The "(:init )" section lists the positions of the items and the agent, and the attributes of all listed items (e.g. accessible, pickable, turnable etc.).
    The ; Positions part lists the positions of all items and the agent in the scene graph. 
    The "(:goal )" section defines the goal using the goal description given above.

    Now given a new scene graph: \n```\n{sg_qry}\n```
    and a new goal description: {goal_qry}
    and using the object types, predicates from the new PDDL domain file named {domain_name_qry}:
    ```\n{get_pddl_domain_types(domain_qry)}\n{get_pddl_domain_predicates(domain_qry)}\n```
    Please provide a new problem file in PDDL with respect to the new scene graph and goal specification directly without further explanations. Please also keep the comments such as "; Begin goal", "; End goal" etc. in the problem file.
    """
    prompt += "\nThe goal should only consist of the previously defined predicates without any further keyword which not appear in the examples such as 'forall' etc."

    prompt += "\nHIERARCHICAL TYPING RULES:\n\nMake sure that the object type declarations CONFORM THE ACTION PARAMETERS i.e. object_1 is grabbable but action_2 requires a pickable object and object_1 should be run with action_2 then object_1 must be declared as pickable as well.\n"
    prompt += "\n\nADDITIONAL GROUNDING IMPROVEMENT RULES\n\nMake sure that the objects in the generated PDDL problem have a correspondence in the scene graph."
    
    possible_objects = []
    for room, items in sg_qry.items():
        for item in items:
            possible_objects.append(item)

    # Groundability improvement prompt
    prompt += "The only objects allowed in the PDDL problem (:objects) section must appear in the following list:\n{}\n".format(possible_objects)
    prompt += "IMPORTANT: NEVER change the name or syntax of the object names.\n"

    # Relaxation prompt
    prompt += "\n\nADDITIONAL GOAL RELAXATION\n\nIf the goal cannot be accomplished with the given scene graph, translated into PDDL a RELAXED version of the goal by changing its complexity or the objects involved.\n"
    prompt += "EXAMPLES:\n\n- COMPLEXITY RELAXATION: i.e. Find me an apple in the kitchen and then put it on the dining table -> Find me an apple in the kitchen and bring it to the living room\n"
    prompt += "- OBJECT RELAXATION: i.e. Find me a red apple in the kitchen -> Find me a snack\n"

    return content, prompt




################ Problem Decomposition Prompt ################

def decompose_problem(goal_exp: str, subgoal_exp: list, subgoal_pddl_exp: list, item_keep_exp: list,
                      goal_qry: str, problem_exp: str, item_keep_qry: list, problem_qry: str,
                      domain_qry: str, acc_goal: bool = False):
    content = "You are an excellent assistant in decomposing long-term tasks. Given a task goal description and a corresponding PDDL problem file, you can decompose the task into multiple sub-tasks, and generate multiple PDDL sub-problem files correspondingly."
    subgoal_in_lines = ", \n".join(
        ["```" + sp + "```" for sp in subgoal_pddl_exp])
    if acc_goal:
        subgoal_pddl_exp_str = f"""When formulating the corresponding sub-goals in PDDL, to ensure that the previously achieved goals still remain valid, one can autoregressively append new sub-goal to the previous ones as follows:\n{subgoal_in_lines}"""
    else:
        subgoal_pddl_exp_str = f"""The corresponding sub-goal descriptions in PDDL can be formulated as: \n{subgoal_in_lines}"""

    prompt = f"""
    A PDDL problem file has three main sections:
    The "(:objects )" section lists all the object types in the environment.
    The "(:init )" section lists the 'Positions' of the items and the agent, and the 'Attributes' of all listed items (e.g. accessible, pickable, turnable etc.).
    The "(:goal )" section defines the task goal.
    
    For example, given a goal description: {goal_exp}, and the following PDDL problem file:
    ```\n{problem_exp}\n```
    
    The task goal can first be broken down intgroundableo multiple sub-goals: {subgoal_exp}.
    {subgoal_pddl_exp_str}
    
    Now given a new goal description: {goal_qry},
    and a corresponding new PDDL problem file:
    ```\n{problem_qry}n```
    and solely using the following predicates:
    ```\n{get_pddl_domain_predicates(domain_qry)}\n```
    
    Note that the robot can only transport one item at a time.
    And the following item(s) are relevant for accomplishing the goal {item_keep_qry}.
    Please break down the new goal into a list of sub-goals as many as possible, and formulate them in PDDL (where each sub-goal only consists of one predicate). Use ``` to wrap each sub-goal in PDDL.
    """
    return content, prompt


def decompose_problem_chain(goal_exp: str, subgoal_exp: list, subgoal_pddl_exp: list, item_keep_exp: list,
                            goal_qry: str, problem_exp: str, item_keep_qry: list, problem_qry: str,
                            domain_qry: str, acc_goal: bool = False):
    content = "You are an excellent assistant in decomposing long-term tasks. Given a task goal description and a corresponding PDDL problem file, you can decompose the task into multiple sub-tasks, and generate multiple PDDL sub-problem files correspondingly."
    subgoal_in_lines = ", \n".join(
        ["```" + sp + "```" for sp in subgoal_pddl_exp])
    if acc_goal:
        subgoal_pddl_exp_str = f"""When formulating the corresponding sub-goals in PDDL, to ensure that the previously achieved goals still remain valid, one can autoregressively append new sub-goal to the previous ones as follows:\n{subgoal_in_lines}"""
    else:
        subgoal_pddl_exp_str = f"""The corresponding sub-goal descriptions in PDDL can be formulated as: \n{subgoal_in_lines}"""

    prompt = f"""
    For example, given a goal description: {goal_exp}, and the following PDDL problem file:
    ```\n{problem_exp}\n```
    
    The task goal can first be broken down into multiple sub-goals: {subgoal_exp}.
    {subgoal_pddl_exp_str}
    
    Now given a new goal description: {goal_qry},
    and a corresponding new PDDL problem file:
    ```\n{problem_qry}\n```
    and solely using the following predicates:
    ```\n{get_pddl_domain_predicates(domain_qry)}\n```
    
    Note that the robot can only transport one item at a time.
    And the following item(s) are relevant for accomplishing the goal {item_keep_qry}.
    Please break down the new goal into a list of sub-goals as many as possible, and formulate them in PDDL (where each sub-goal only consists of one predicate). Use ``` to wrap each sub-goal in PDDL.
    """
    return content, prompt