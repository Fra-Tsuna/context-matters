import re

from src.context_matters.utils.graph import *
from .pddl_generation import _save_prompt_response
from src.context_matters.logger_cfg import logger


def relax_goal_pipeline(graph, task, agent, workflow_iteration=None, logs_dir=None):
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    system_prompt = open(os.path.join(CURRENT_DIR, "prompt", "object_needed.txt"),"r").read()
    user_prompt = "SCENE: \n" + graph + "\n\nTASK: \n" + task
    objects_needed = agent.llm_call(system_prompt, user_prompt, temperature=0.1, top_p=1)
    logger.info("Answer:", objects_needed)

    ### SCENE CATEGORIZATION
    system_prompt = open(os.path.join(CURRENT_DIR, "prompt", "scene_categorization.txt"),"r").read()
    user_prompt = "SCENE: \n" + graph + "\n\nTASK: \n" + task

    scene_categorization = agent.llm_call(system_prompt, user_prompt, temperature=0.1, top_p=1)
    logger.info("Scene categorization answer:", scene_categorization)

    system_prompt = open(os.path.join(CURRENT_DIR, "prompt", "relaxation_thinking.txt"),"r").read()
    user_prompt = "SCENE: \n" + graph + "\n\nTASK: \n" + task + "\n\nOBJECTS NEEDED: \n" + objects_needed + "\n\nOBJECTS AVAILABLE: \n" + scene_categorization
    relaxation_thinking = agent.llm_call(system_prompt, user_prompt, temperature=0.1, top_p=1)
    logger.info("Relaxation thinking answer:", relaxation_thinking)

    system_prompt = open(os.path.join(CURRENT_DIR, "prompt", "goal_relaxation.txt"),"r").read()
    user_prompt = "Original goal: " + task + "\nRelaxation analysis: \n" + relaxation_thinking
    new_goal = agent.llm_call(system_prompt, user_prompt, temperature=0.1, top_p=1) 
    logger.info("New goal:", new_goal)

    new_goal = new_goal.split("<NEW_GOAL>")[1]
    return new_goal


def dict_replaceable_objects(graph, task, agent, workflow_iteration=None, logs_dir=None):
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    system_prompt = open(os.path.join(CURRENT_DIR, "prompt", "replace_objects.txt"),"r").read()
    user_prompt = "SCENE: \n" + graph + "\n\nTASK: \n" + task

    answer = agent.llm_call(system_prompt, user_prompt, temperature=0.1, top_p=1)
    
    if logs_dir is not None:
        _save_prompt_response(
            prompt=f"System:\n"+system_prompt+"\n\nUser:\n"+user_prompt,
            response=answer,
            prefix="object_replacement",
            suffix=workflow_iteration,
            output_dir=logs_dir
        )

    new_goal = answer.split("<NEW_GOAL>")[1]

    # If no goal is returned, try again
    if not new_goal.strip():

        answer = agent.llm_call(system_prompt, user_prompt, temperature=0.1, top_p=1)
        
        if logs_dir is not None:
            _save_prompt_response(
                prompt=f"System:\n"+system_prompt+"\n\nUser:\n"+user_prompt,
                response=answer,
                prefix="object_replacement",
                suffix=workflow_iteration,
                output_dir=logs_dir
            )

        new_goal = answer.split("<NEW_GOAL>")[1]

        # If still no goal is returned, return the original goal
        if not new_goal.strip():
            return task

    return new_goal

def relax_goal(objects, goal, agent, workflow_iteration=None, logs_dir=None):
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    system_prompt = open(os.path.join(CURRENT_DIR, "prompt", "relax_goal.txt"),"r").read()
    user_prompt = "Objects:\n" + str(objects) + "\nGoal:" + goal
    answer = agent.llm_call(system_prompt, user_prompt, temperature=0.1, top_p=1)

    if logs_dir is not None:
        _save_prompt_response(
            prompt=f"System:\n"+system_prompt+"\n\nUser:\n"+user_prompt,
            response=answer,
            prefix="goal_relaxation",
            suffix=workflow_iteration,
            output_dir=logs_dir
        )

    new_goal = answer.split("<NEW_GOAL>")[1]

    # If no goal is returned, try again
    if not new_goal.strip():
        
        answer = agent.llm_call(system_prompt, user_prompt, temperature=0.1, top_p=1)
        
        if logs_dir is not None:
            _save_prompt_response(
                prompt=f"System:\n"+system_prompt+"\n\nUser:\n"+user_prompt,
                response=answer,
                prefix="goal_relaxation",
                suffix=workflow_iteration,
                output_dir=logs_dir
            )

        new_goal = answer.split("<NEW_GOAL>")[1]

        # If still no goal is returned, return the original goal
        if not new_goal.strip():
            return goal
        
    return new_goal