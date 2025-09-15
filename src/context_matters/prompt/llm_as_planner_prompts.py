def sayplan_plan_exp():
    """
    Provides a  comprehensive example of the LLM's expected output format.
    Reflects the 'cleaning' task from delta_example.py DOMAIN_DATA_EXAMPLE
    """
    output = {
        "chain_of_thought": "I have located the relevant item: mop_11 for cleaning -> First, get the mop and clean the floors of all required rooms (kitchen, bathroom, dining room) -> Next, clean the dirty mop and return it to the kitchen.",
        "reasoning": "I will generate a task plan using the identified scene graph.",
        "plan": """(move_to robot dining_room_1 kitchen_2)
        (pick robot mop_11 kitchen_2)
        (mop_floor robot mop_11 kitchen_2)
        (move_to robot kitchen_2 dining_room_1)
        (mop_floor robot mop_11 dining_room_1)
        (move_to robot dining_room_1 bathroom_3)
        (mop_floor robot mop_11 bathroom_3)
        (move_to robot bathroom_3 dining_room_1)
        (move_to robot dining_room_1 kitchen_2)
        (drop robot mop_11 kitchen_2)
        """
            }
    return output

def sayplan_output_format():
    return {
        "chain_of_thought": "break your problem down into a series of intermediate reasoning steps to help you determine your next command",
        "reasoning": "justify why the next action is important",
        "plan": "high-level task plan, which only consists of actions listed in the ENVIRONMENT FUNCTIONS above. The number of actions (plan length) is always finite."
    }

def sg_2_plan(sg_exp: dict, sg_qry: dict, goal_exp: str, goal_qry: str, agent_starting_position: str, 
              add_obj_exp: str = None, add_obj_qry: str = None, add_act_exp: str = None, add_act_qry: str = None):
    content = "You are an excellent graph planning agent. Given some domain knowledge and a scene graph representation of an environment, you can use it to generate a step-by-step task plan for solving a given goal instruction."

    info_add_obj_exp = ", {}". format(
        ", ".join(add_obj_exp)) if add_obj_exp is not None else ""
    info_add_act_exp = "and the following additional action(s): \n{}".format(
        "\n".join(add_act_exp)) if add_act_exp is not None else ""
    info_add_obj_qry = "there are the following new object type(s): {}.".format(
        ", ".join(add_obj_qry)) if add_obj_qry is not None else ""
    info_add_act_qry = "\nBesides the basic actions (move_to, pick, drop), there are the following additional action(s): \n{}".format(
        "\n".join(add_act_qry)) if add_act_qry is not None else ""
    
      
    output_exp = sayplan_plan_exp()

    prompt = f"""
    EXAMPLE:
    A domain has the following object types: agent, room, item{info_add_obj_exp}. 
    The agent can perform the following basic actions:
    goto(<agent>, <room_1>, <room_2>): <agent> goes from <room_1> to <room_2>, where <room_1> and <room_2> should be neighbors. As result, <agent> will leave <room_1> and locates in <room_2>.
    pick(<agent>, <item>, <room>): <agent> picks up an <item> at <room>. <item> must be accessible and located in <room>, the pick action is in <item>'s affordance, and <agent> state is hand-free. As result, <agent> state will change to loaded, and <item> left <room>.
    drop(<agent>, <item>, <room>): <agent> drops an <item> at <room>. <item> is accessible, the drop action is in <item>'s affordance, <agent> is in <room> and has <item> in hand, and <agent> state is loaded. As result, <item> will locate in <room>, <agent> state will change to hand-free.
    {info_add_act_exp}
    
    Here is an example of a scene graph in the form of a nested dictionary in Python:
    ```\n{sg_exp}\n```\n
    The top level keys are the name of the scene, the rooms, the agents, and possibly the humans.
    Each room contains a dictionary of 'items' inside the rooms, and a list of 'neighbor' (connected) rooms. The 'neighbor' relation is bidirectional, i.e. if kitchen is neighbor of corridor, then corridor is also neighbor of kitchen.
    Each item has three attributes, 'accessible' means if the item can be accessed or not, 'affordance' indicates the affordable actions of this item, 'state' infers whether the item is free, or occupied, e.g. being picked by an agent.
    Each agent has two attributes, the current position and the state.

    Given a goal description e.g., {goal_exp}, and using the previously defined object types and actions, you can generate the output according to the given OUTPUT RESPONSE FORMAT (Python dict, USE DOUBLE QUOTES!):
    {sayplan_output_format()}
    
    Following is an example output with respect to the example scene graph above:\n{output_exp}
    
    QUERY:
    Now in a new domain, {info_add_obj_qry} {info_add_act_qry}\n
    New instruction: {goal_qry}
    The agent starts at {agent_starting_position}
    New 3D scene graph:\n{sg_qry}
    Please generate the output with according to the given OUTPUT RESPONSE FORMAT directly without further explanations. Make sure to use the term "robot" in the plan instead of "agent". Do not generate infinite actions in the plan!
    """

    #Note:
    #In the query, added the agent_starting_position to inform where the agent starts in the scene graph

    return content, prompt
