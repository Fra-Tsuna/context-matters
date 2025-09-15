SEARCH_EXAMPLE_PROMPT = """
    Instruction: Launder the clothes and bring them to bedroom_1.
    3D scene graph: {'bathroom': [], 'closet_1': [], 'closet_2': [], 'corridor_1': [], 'corridor_2': [], 'bedroom_1': [], 'bedroom_2': [], 'kitchen': [], 'living_room': [], 'agent': {'position': 'living_room', 'state': 'hand-free'}}
    Memory (history of previously expanded room nodes): []
    Output: {
        "mode": "search",
        "chain_of_thought": "Explore graph for identifiying relevant items for solving the given task -> perform common sense exploration first -> explore kitchen -> explore bathroom -> explore bedroom_1 -> if items not found then explore other nodes.",
        "reasoning": "I will expand kitchen.",
        "command": ("expand", "kitchen")
    }
    3D scene graph: {'bathroom': [], 'closet_1': [], 'closet_2': [], 'corridor_1': [], 'corridor_2': [], 'bedroom_1': [], 'bedroom_2': [], 'kitchen': [('wash_machine', {'affordance': ['open', 'close', 'turnon', 'turnoff']}), ('microwave', {'affordance': ['open', 'close', 'turnon', 'turnoff']}), ('oven', {'affordance': ['turnon', 'turnoff']}), ('sink_2', {'affordance': []}), ('fridge_3', {'affordance': ['open', 'close']})], 'living_room': [], 'agent': {'position': 'living_room', 'state': 'hand-free'}}
    Memory: ["kitchen"]
    Output: {
        "mode": "search",
        "chain_of_thought": "I have found wash_machine in kitchen which is relevant for solving the task -> leave this room expanded -> still further items missing for the task -> bathroom might have relevant items -> explore bathroom next.",
        "reasoning": "I will expand bathroom next.",
        "command": ("expand", "bathroom")
    }n
    3D scene graph: {'bathroom': [('sink_1', {'affordance': []}), ('toilet', {'affordance': []}), ('detergent', {'affordance': ['pick', 'drop']})], 'closet_1': [], 'closet_2': [], 'corridor_1': [], 'corridor_2': [], 'bedroom_1': [], 'bedroom_2': [], 'kitchen': [('wash_machine', {'affordance': ['open', 'close', 'turnon', 'turnoff']}), ('microwave', {'affordance': ['open', 'close', 'turnon', 'turnoff']}), ('oven', {'affordance': ['turnon', 'turnoff']}), ('sink_2', {'affordance': []}), ('fridge_3', {'affordance': ['open', 'close']})], 'living_room': [], 'agent': {'position': 'living_room', 'state': 'hand-free'}}
    Memory: ["kitchen", "bathroom"]
    Output: {
        "mode": "search",
        "chain_of_thought": "I have found detergent in bathroom -> leave this room expanded -> still further items missing for the task -> bedroom_1 might have clothes -> explore bedroom_1 next.",
        "reasoning": "I will expand bedroom_1 next.",
        "command": ("expand", "bedroom_1")
    }
    3D scene graph: {'bathroom': [('sink_1', {'affordance': []}), ('toilet', {'affordance': []}), ('detergent', {'affordance': ['pick', 'drop']})], 'closet_1': [], 'closet_2': [], 'corridor_1': [], 'corridor_2': [], 'bedroom_1': [('bed_1', {'affordance': []})], 'bedroom_2': [], 'kitchen': [('wash_machine', {'affordance': ['open', 'close', 'turnon', 'turnoff']}), ('microwave', {'affordance': ['open', 'close', 'turnon', 'turnoff']}), ('oven', {'affordance': ['turnon', 'turnoff']}), ('sink_2', {'affordance': []}), ('fridge_3', {'affordance': ['open', 'close']})], 'living_room': [], 'agent': {'position': 'living_room', 'state': 'hand-free'}}
    Memory: ["kitchen", "bathroom", "bedroom_1"]
    Output: {
        "mode": "search",
        "chain_of_thought": "bedroom_1 does not have clothes or anything relevant towards solving this task -> I should contract this room to save token memory -> I will explore other unexplored rooms after doing this.",
        "reasoning": "I will contract bedroom_1 to save memory.",
        "command": ("contract", "bedroom_1")
    }
    3D scene graph: {'bathroom': [('sink_1', {'affordance': []}), ('toilet', {'affordance': []}), ('detergent', {'affordance': ['pick', 'drop']})], 'closet_1': [], 'closet_2': [], 'corridor_1': [], 'corridor_2': [], 'bedroom_1': [], 'bedroom_2': [], 'kitchen': [('wash_machine', {'affordance': ['open', 'close', 'turnon', 'turnoff']}), ('microwave', {'affordance': ['open', 'close', 'turnon', 'turnoff']}), ('oven', {'affordance': ['turnon', 'turnoff']}), ('sink_2', {'affordance': []}), ('fridge_3', {'affordance': ['open', 'close']})], 'living_room': [], 'agent': {'position': 'living_room', 'state': 'hand-free'}}
    Memory: ["kitchen", "bathroom", "bedroom_1"]
    Output: {
        "mode": "search",
        "chain_of_thought": "closet_1, closet_2, corridor_1, corridor_2, bedroom_2, and living_room are unexplored yet -> I will try bedroom_2 next.",
        "reasoning": "I will expand closet_1 as it is unexplored.",
        "command": ("expand", "bedroom_2")
    }
    3D scene graph: {'bathroom': [('sink_1', {'affordance': []}), ('toilet', {'affordance': []}), ('detergent', {'affordance': ['pick', 'drop']})], 'closet_1': [], 'closet_2': [], 'corridor_1': [], 'corridor_2': [], 'bedroom_1': [], 'bedroom_2': [('bed_2', {'affordance': []}), ('book', {'affordance': ['pick', 'drop']})], 'kitchen': [('wash_machine', {'affordance': ['open', 'close', 'turnon', 'turnoff']}), ('microwave', {'affordance': ['open', 'close', 'turnon', 'turnoff']}), ('oven', {'affordance': ['turnon', 'turnoff']}), ('sink_2', {'affordance': []}), ('fridge_3', {'affordance': ['open', 'close']})], 'living_room': [], 'agent': {'position': 'living_room', 'state': 'hand-free'}}
    Memory: ["kitchen", "bathroom", "bedroom_1", "bedroom_2"]
    Output: {
        "mode": "search",
        "chain_of_thought": "bedroom_1 does not have clothes or anything relevant towards solving this task -> I should contract this room to save token memory -> I will explore other unexplored rooms after doing this.",
        "reasoning": "I will contract bedroom_2 to save memory.",
        "command": ("contract", "bedroom_2")
    }
    3D scene graph: {'bathroom': [('sink_1', {'affordance': []}), ('toilet', {'affordance': []}), ('detergent', {'affordance': ['pick', 'drop']})], 'closet_1': [], 'closet_2': [], 'corridor_1': [], 'corridor_2': [], 'bedroom_1': [], 'bedroom_2': [], 'kitchen': [('wash_machine', {'affordance': ['open', 'close', 'turnon', 'turnoff']}), ('microwave', {'affordance': ['open', 'close', 'turnon', 'turnoff']}), ('oven', {'affordance': ['turnon', 'turnoff']}), ('sink_2', {'affordance': []}), ('fridge_3', {'affordance': ['open', 'close']})], 'living_room': [], 'agent': {'position': 'living_room', 'state': 'hand-free'}}
    Memory: ["kitchen", "bathroom", "bedroom_1", "bedroom_2"]
    Output: {
        "mode": "search",
        "chain_of_thought": "closet_1, closet_2, corridor_1, corridor_2, and living_room are unexplored yet -> I will try closet_1 next.",
        "reasoning": "I will expand closet_1 as it is unexplored.",
        "command": ("expand", "closet_1")
    }
    3D scene graph: {'bathroom': [('sink_1', {'affordance': []}), ('toilet', {'affordance': []}), ('detergent', {'affordance': ['pick', 'drop']})], 'closet_1': [('clothes', {'affordance': ['pick', 'drop']})], 'closet_2': [], 'corridor_1': [], 'corridor_2': [], 'bedroom_1': [], 'bedroom_2': [], 'kitchen': [('wash_machine', {'affordance': ['open', 'close', 'turnon', 'turnoff']}), ('microwave', {'affordance': ['open', 'close', 'turnon', 'turnoff']}), ('oven', {'affordance': ['turnon', 'turnoff']}), ('sink_2', {'affordance': []}), ('fridge_3', {'affordance': ['open', 'close']})], 'living_room': [], 'agent': {'position': 'living_room', 'state': 'hand-free'}}
    Memory: ["kitchen", "bathroom", "bedroom_1", "bedroom_2", "closet_1"]
    Output: {
        "mode": "search",
        "chain_of_thought": "I have found clothes in closet_1 -> leave this room expanded -> closet_2, corridor_1, corridor_2, and living_room are unexplored yet -> I will try closet_2 next.",
        "reasoning": "I will expand closet_2 as it is unexplored.",
        "command": ("expand", "closet_2")
    }
    3D scene graph: {'bathroom': [('sink_1', {'affordance': []}), ('toilet', {'affordance': []}), ('detergent', {'affordance': ['pick', 'drop']})], 'closet_1': [('clothes', {'affordance': ['pick', 'drop']})], 'closet_2': [('fridge_1', {'affordance': ['open', 'close']}), ('fridge_2', {'affordance': ['open', 'close']})], 'corridor_1': [], 'corridor_2': [], 'bedroom_1': [], 'bedroom_2': [], 'kitchen': [('wash_machine', {'affordance': ['open', 'close', 'turnon', 'turnoff']}), ('microwave', {'affordance': ['open', 'close', 'turnon', 'turnoff']}), ('oven', {'affordance': ['turnon', 'turnoff']}), ('sink_2', {'affordance': []}), ('fridge_3', {'affordance': ['open', 'close']})], 'living_room': [], 'agent': {'position': 'living_room', 'state': 'hand-free'}}
    Memory: ["kitchen", "bathroom", "bedroom_1", "bedroom_2", "closet_1", "closet_2"]
    Output: {
        "mode": "search",
        "chain_of_thought": "closet_2 does not have anything relevant towards solving this task -> I should contract this room to save token memory -> I will explore other unexplored rooms after doing this.",
        "reasoning": "I will contract closet_2 to save memory.",
        "command": ("contract", "closet_2")
    }
    3D scene graph: {'bathroom': [('sink_1', {'affordance': []}), ('toilet', {'affordance': []}), ('detergent', {'affordance': ['pick', 'drop']})], 'closet_1': [('clothes', {'affordance': ['pick', 'drop']})], 'closet_2': [], 'corridor_1': [], 'corridor_2': [], 'bedroom_1': [], 'bedroom_2': [], 'kitchen': [('wash_machine', {'affordance': ['open', 'close', 'turnon', 'turnoff']}), ('microwave', {'affordance': ['open', 'close', 'turnon', 'turnoff']}), ('oven', {'affordance': ['turnon', 'turnoff']}), ('sink_2', {'affordance': []}), ('fridge_3', {'affordance': ['open', 'close']})], 'living_room': [], 'agent': {'position': 'living_room', 'state': 'hand-free'}}
    Memory: ["kitchen", "bathroom", "bedroom_1", "bedroom_2", "closet_1", "closet_2"]
    Output: {
        "mode": "search",
        "chain_of_thought": "corridor_1, corridor_2, and living_room are unexplored yet -> I will try corridor_1 next.",
        "reasoning": "I will expand corridor_1 as it is unexplored.",
        "command": ("expand", "corridor_1")
    }
    3D scene graph: {'bathroom': [('sink_1', {'affordance': []}), ('toilet', {'affordance': []}), ('detergent', {'affordance': ['pick', 'drop']})], 'closet_1': [('clothes', {'affordance': ['pick', 'drop']})], 'closet_2': [], 'corridor_1': [], 'corridor_2': [], 'bedroom_1': [], 'bedroom_2': [], 'kitchen': [('wash_machine', {'affordance': ['open', 'close', 'turnon', 'turnoff']}), ('microwave', {'affordance': ['open', 'close', 'turnon', 'turnoff']}), ('oven', {'affordance': ['turnon', 'turnoff']}), ('sink_2', {'affordance': []}), ('fridge_3', {'affordance': ['open', 'close']})], 'living_room': [], 'agent': {'position': 'living_room', 'state': 'hand-free'}}
    Memory: ["kitchen", "bathroom", "bedroom_1", "bedroom_2", "closet_1", "closet_2", "corridor_1"]
    Output: {
        "mode": "search",
        "chain_of_thought": "corridor_1 does not have anything relevant towards solving this task -> I should contract this room to save token memory -> I will explore other unexplored rooms after doing this.",
        "reasoning": "I will contract corridor_1 to save memory.",
        "command": ("contract", "corridor_1")
    }
    3D scene graph: {'bathroom': [('sink_1', {'affordance': []}), ('toilet', {'affordance': []}), ('detergent', {'affordance': ['pick', 'drop']})], 'closet_1': [('clothes', {'affordance': ['pick', 'drop']})], 'closet_2': [], 'corridor_1': [], 'corridor_2': [], 'bedroom_1': [], 'bedroom_2': [], 'kitchen': [('wash_machine', {'affordance': ['open', 'close', 'turnon', 'turnoff']}), ('microwave', {'affordance': ['open', 'close', 'turnon', 'turnoff']}), ('oven', {'affordance': ['turnon', 'turnoff']}), ('sink_2', {'affordance': []}), ('fridge_3', {'affordance': ['open', 'close']})], 'living_room': [], 'agent': {'position': 'living_room', 'state': 'hand-free'}}
    Memory: ["kitchen", "bathroom", "bedroom_1", "bedroom_2", "closet_1", "closet_2", "corridor_1"]
    Output: {
        "mode": "search",
        "chain_of_thought": "corridor_2 and living_room are unexplored yet -> I will try corridor_2 next.",
        "reasoning": "I will expand corridor_2 as it is unexplored.",
        "command": ("expand", "corridor_2")
    }
    3D scene graph: {'bathroom': [('sink_1', {'affordance': []}), ('toilet', {'affordance': []}), ('detergent', {'affordance': ['pick', 'drop']})], 'closet_1': [('clothes', {'affordance': ['pick', 'drop']})], 'closet_2': [], 'corridor_1': [], 'corridor_2': [], 'bedroom_1': [], 'bedroom_2': [], 'kitchen': [('wash_machine', {'affordance': ['open', 'close', 'turnon', 'turnoff']}), ('microwave', {'affordance': ['open', 'close', 'turnon', 'turnoff']}), ('oven', {'affordance': ['turnon', 'turnoff']}), ('sink_2', {'affordance': []}), ('fridge_3', {'affordance': ['open', 'close']})], 'living_room': [], 'agent': {'position': 'living_room', 'state': 'hand-free'}}
    Memory: ["kitchen", "bathroom", "bedroom_1", "bedroom_2", "closet_1", "closet_2", "corridor_1", "corridor_2"]
    Output: {
        "mode": "search",
        "chain_of_thought": "corridor_2 does not have anything relevant towards solving this task -> I should contract this room to save token memory -> I will explore other unexplored rooms after doing this.",
        "reasoning": "I will contract corridor_2 to save memory.",
        "command": ("contract", "corridor_2")
    }
    3D scene graph: {'bathroom': [('sink_1', {'affordance': []}), ('toilet', {'affordance': []}), ('detergent', {'affordance': ['pick', 'drop']})], 'closet_1': [('clothes', {'affordance': ['pick', 'drop']})], 'closet_2': [], 'corridor_1': [], 'corridor_2': [], 'bedroom_1': [], 'bedroom_2': [], 'kitchen': [('wash_machine', {'affordance': ['open', 'close', 'turnon', 'turnoff']}), ('microwave', {'affordance': ['open', 'close', 'turnon', 'turnoff']}), ('oven', {'affordance': ['turnon', 'turnoff']}), ('sink_2', {'affordance': []}), ('fridge_3', {'affordance': ['open', 'close']})], 'living_room': [], 'agent': {'position': 'living_room', 'state': 'hand-free'}}
    Memory: ["kitchen", "bathroom", "bedroom_1", "bedroom_2", "closet_1", "closet_2", "corridor_1", "corridor_2"]
    Output: {
        "mode": "search",
        "chain_of_thought": "living_room is unexplored yet -> I will try living_room next.",
        "reasoning": "I will expand living_room as it is unexplored.",
        "command": ("expand", "living_room")
    }
    3D scene graph: {'bathroom': [('sink_1', {'affordance': []}), ('toilet', {'affordance': []}), ('detergent', {'affordance': ['pick', 'drop']})], 'closet_1': [('clothes', {'affordance': ['pick', 'drop']})], 'closet_2': [], 'corridor_1': [], 'corridor_2': [], 'bedroom_1': [], 'bedroom_2': [], 'kitchen': [('wash_machine', {'affordance': ['open', 'close', 'turnon', 'turnoff']}), ('microwave', {'affordance': ['open', 'close', 'turnon', 'turnoff']}), ('oven', {'affordance': ['turnon', 'turnoff']}), ('sink_2', {'affordance': []}), ('fridge_3', {'affordance': ['open', 'close']})], 'living_room': [('tv', {'affordance': ['turnon', 'turnoff']}), ('couch', {'affordance': []})], 'agent': {'position': 'living_room', 'state': 'hand-free'}}
    Memory: ["kitchen", "bathroom", "bedroom_1", "bedroom_2", "closet_1", "closet_2", "corridor_1", "corridor_2", "living_room"]
    Output: {
        "mode": "search",
        "chain_of_thought": "living_room does not have anything relevant towards solving this task -> I should contract this room to save token memory -> I have explored all rooms and found all relevant items for solving the task.",
        "reasoning": "I will contract living_room to save memory.",
        "command": ("contract", "living_room")
    }
    3D scene graph: {'bathroom': [('sink_1', {'affordance': []}), ('toilet', {'affordance': []}), ('detergent', {'affordance': ['pick', 'drop']})], 'closet_1': [('clothes', {'affordance': ['pick', 'drop']})], 'closet_2': [], 'corridor_1': [], 'corridor_2': [], 'bedroom_1': [], 'bedroom_2': [], 'kitchen': [('wash_machine', {'affordance': ['open', 'close', 'turnon', 'turnoff']}), ('microwave', {'affordance': ['open', 'close', 'turnon', 'turnoff']}), ('oven', {'affordance': ['turnon', 'turnoff']}), ('sink_2', {'affordance': []}), ('fridge_3', {'affordance': ['open', 'close']})], 'living_room': [], 'agent': {'position': 'living_room', 'state': 'hand-free'}}
    Memory: ["kitchen", "bathroom", "bedroom_1", "bedroom_2", "closet_1", "closet_2", "corridor_1", "corridor_2", "living_room"]
    Output: {
        "mode": "search",
        "chain_of_thought": "I have found all relevant items for solving the task -> search complete -> switch to planning mode.",
        "reasoning": "I will switch to planning mode.",
        "command": "Switch to planning"
    }
"""


def search_prompt(query_goal, query_collapsed_sg, initial_robot_location):
    content = "You are an excellent graph exploration agent. Given a graph representation of an environment, you can explore the graph by expanding room nodes to find the items of interest."

    prompt = f"""
    ENVIRONMENT API:
    expand(<room>): Reveal items connected to a room node.
    contract(<room>): Hide items to reduce graph size for memory constraints.
    
    OUTPUT RESPONSE FORMAT:
    "mode": "search",
    "chain_of_thought": "break your problem down into a series of intermediate reasoning steps to help you determine your next command",
    "reasoning": "justify why the next action is important",
    "command": (command_name, room_name): "command_name": "expand" or "contract", "room_name": room to perform an operation on."
    
    EXAMPLE: \n{SEARCH_EXAMPLE_PROMPT}
    
    QUERY:
    New instruction: {query_goal}
    New 3D scene graph: \n{query_collapsed_sg}
    Initial robot location: {initial_robot_location}
    Please generate the output with according to the given OUTPUT RESPONSE FORMAT directly without further explanations. Make sure to use the term "robot" in the plan instead of "agent".
    """
    return content, prompt


def export_search_cmd(response: str):
    start_idx = response.find("{")
    end_idx = response.find("}")
    sliced_response = response[start_idx : end_idx + 1]
    output = eval(sliced_response)
    return (
        output["mode"],
        output["chain_of_thought"],
        output["reasoning"],
        output["command"],
    )


PLAN_EXAMPLE_PROMPT = """
    Instruction: Launder the clothes and bring them to bedroom_1.

    3D scene graph: {
        'bathroom': [
            ('detergent', {'affordance': ['pick', 'drop']})
        ],
        'closet_1': [
            ('clothes', {'affordance': ['pick', 'drop']})
        ],
        'closet_2': [],
        'corridor_1': [],
        'corridor_2': [],
        'bedroom_1': [],
        'bedroom_2': [],
        'kitchen': [
            ('wash_machine', {'affordance': ['open', 'close', 'turnon', 'turnoff']})
        ],
        'living_room': [],
        'agent': {
            'position': 'living_room',
            'state': 'hand-free'
        }
    }

    Additional ENVIRONMENT FUNCTIONS:
    * Additional action(s):
    - launder(<agent>, <item_1>, <item_2>, <item_3>, <room>): For laundering.
        Arguments: <agent> (the agent performing the action), <item_1> (the clothes to be laundered), <item_2> (the detergent used for laundering), <item_3> (the wash machine used for laundering), <room> (the room where the laundering takes place).
        Preconditions: <agent> must be in the same room as <item_1>, <item_2>, and <item_3>, <item_1> is dirty, and <agent> must be free (not holding anything).
        Postconditions: <item_1> is clean.

    Output:
    {
        'mode': 'planning',
        'chain_of_thought': 'I have located the relevent items: clothes, detergent, and wash_machine -> generate plan for launder the clothes -> launder the clothes with detergent and wash_machine and bring them to bedroom_1.',
        'reasoning': 'I will generate a task plan using the identified scene graph.',
        'plan': '''(move_to robot living_room corridor_1)
                (move_to robot corridor_1 bathroom)
                (grab robot detergent bathroom)
                (move_to robot bathroom kitchen)
                (drop robot detergent kitchen)
                (move_to robot kitchen closet_1)
                (grab robot clothes closet_1)
                (move_to robot closet_1 kitchen)
                (drop robot clothes kitchen)
                (launder robot clothes detergent wash_machine kitchen)
                (grab robot clothes kitchen)
                (move_to robot kitchen bedroom_1)
                (drop robot clothes bedroom_1)'''
    }
"""

def get_situational_actions(task: str) -> str:
    if task == "dining_setup":
        actions = """
        Additional ENVIRONMENT FUNCTIONS:
        * Additional action(s):
        - put_on(<agent>, <item>, <surface>, <room>): This action allows a robot to place a grabbable object onto a surface in a room.
            Arguments: <agent> (the robot that will place the object), <item> (the grabbable object to be placed), <surface> (the surface where the object will be placed), <room> (the room where the action takes place).
            Preconditions: the <agent>, the <surface>, and the <item> must be in the same room, and the <agent> must be holding the <item>.
            Postconditions: the <agent> is no longer holding the <item>, the <item> is on the <surface>, and the <agent> is free.
        - take_from(<agent>, <item>, <surface>, <room>): This action allows a robot to take a grabbable object from a surface in a room.
            Arguments: <agent> (the robot that will take the object), <item> (the grabbable object to be taken), <surface> (the surface where the object is located), <room> (the room where the action takes place).
            Preconditions: the <agent>, the <surface>, and the <item> must be in the same room, and the <agent> must be free (not holding anything).
            Postconditions: the <agent> is holding the <item>, the <item> is no longer on the <surface>, and the <agent> is no longer free.
        """
    elif task == "house_cleaning":
        actions = """
        Additional ENVIRONMENT FUNCTIONS:
        * Additional action(s):
        - throw_away(<agent>, <item>, <bin>, <room>): This action allows a robot to throw away an item into a bin in a room.
            Arguments: <agent> (the robot that will throw away the item), <item> (the item to be thrown away), <bin> (the bin where the item will be thrown away), <room> (the room where the action takes place).
            Preconditions: the <agent>, the <item>, and the <bin> must be in the same room, and the <agent> must be holding the <item>.
            Postconditions: The <agent> is no longer holding the <item> and is free, the <item> is marked as thrashed, and the <item> is no longer in the <room>.
        - mop_floor(<agent>, <mop>, <room>): This action allows a robot to mop the floor in a room.
            Arguments: <agent> (the robot that will mop the floor), <mop> (the mop to be used), <room> (the room where the action takes place).
            Preconditions: the <agent>, the <mop>, and the <room> must be in the same room, and the <agent> must be holding the clean <mop>, and the floor in the <room> must be dirty.
            Postconditions: The floor in the <room> is clean, the <mop> is dirty, and the <mop> is no longer clean.
        - clean_mop(<agent>, <mop>,): This action allows a robot to clean a dirty mop.
            Arguments: <agent> (the robot that will clean the mop), <mop> (the dirty mop to be cleaned).
            Preconditions: The <agent> must be holding a dirty <mop>.
            Postconditions: The <mop> is clean and no longer dirty.
        """
    elif task == "laundry":
        actions = """
        Additional ENVIRONMENT FUNCTIONS:
        * Additional action(s):
        - open(<agent>, <washing_machine>, <room>): The robot opens a washing machine in a room.
            Arguments: <agent> (the robot), <washing_machine> (the washing machine), <room> (the room).
            Preconditions: The <agent> and the <washing_machine> are in the same room <room>, and the <washing_machine> is closed.
            Postconditions: The <washing_machine> is open.
        - close(<agent>, <washing_machine>, <room>): The robot closes a washing machine in a room.
            Arguments: <agent> (the robot), <washing_machine> (the washing machine), <room> (the room).
            Preconditions: The <agent> and the <washing_machine> are in the same room <room>, and the <washing_machine> is open.
            Postconditions: The <washing_machine> is closed.
        - refill(<agent>, <washing_machine>, <room>, <cleaning_supply>): The robot refills a washing machine with a cleaning supply in a room.
            Arguments: <agent> (the robot), <washing_machine> (the washing machine), <room> (the room), <cleaning_supply> (the cleaning supply).
            Preconditions: The <agent> and the <washing_machine> are in the same <room>, the <agent> is holding the <cleaning_supply>, and the <washing_machine> is empty.
            Postconditions: The <washing_machine> is refilled and no longer empty.
        - put_inside(<agent>, <cleanable_object>, <washing_machine>, <room>): The robot puts a cleanable object inside a washing machine in a room.
            Arguments: <agent> (the robot), <cleanable_object> (the cleanable object), <washing_machine> (the washing machine), <room> (the room).
            Preconditions: The <agent> and the <washing_machine> are in the same <room>, the <agent> is holding the <cleanable_object>, and the <washing_machine> is open.
            Postconditions: The <agent> is no longer holding the <cleanable_object>, the <cleanable_object> is inside the <washing_machine>, and the <agent> is free.
        - wash(<agent>, <cleanable_object>, <washing_machine>, <room>): The robot washes a cleanable object inside a washing machine in a room.
            Arguments: <agent> (the robot), <cleanable_object> (the cleanable object), <washing_machine> (the washing machine), <room> (the room).
            Preconditions: The <agent> and the <washing_machine> are in the same <room>, the <cleanable_object> is inside the <washing_machine>, the <washing_machine> is closed, and the <washing_machine> is refilled.
            Postconditions: The <cleanable_object> is clean and no longer dirty.
        """
    elif task == "office_setup":
        actions = """
        Additional ENVIRONMENT FUNCTIONS:
        * Additional action(s):
        - push(<agent>, <container>, <starting_room>, <destination_room>): The robot pushes a movable container from one room to another.
            Arguments: <agent> (the robot), <container> (the container to be pushed), <starting_room> (the starting room), <destination_room> (the destination room).
            Preconditions: the <agent> and the <container> are in the <starting_room>, and the <container> is empty.
            Postconditions: the <agent> and the <container> are no longer in the <starting_room> and are now in the <destination_room>.
        - put_in(<agent>, <object>, <container>, <room>): The robot places an object into a movable container.
            Arguments: <agent> (the robot), <object> (the object to be placed), <container> (the container), <room> (the room where the container is located).
            Preconditions: The <agent>, the <object>, and the <container> must be in the same <room>, and the <agent> must be holding the <object>.
            Postconditions: The <agent> is no longer holding the <object>, the <object> is inside the <container>, and the <agent> is free.
        - take_out(<agent>, <object>, <container>, <room>): The robot removes an object from a movable container.
            Arguments: <agent> (the robot), <object> (the object to be removed), <container> (the container), <room> (the room where the container is located).
            Preconditions: The <agent> and the <container> are in the same <room>, the <object> is inside the <container>, and the <agent> is free to grab.
            Postconditions: The <object> is no longer inside the <container>, the <agent> is holding the <object>, and the <agent> is no longer free.
        """
    elif task == "pc_assembly":
        actions = """
        Additional ENVIRONMENT FUNCTIONS:
        * Additional action(s):
        - put_on(<agent>, <item>, <surface>, <room>): This action allows a robot to place a grabbable object onto a surface in a room.
            Arguments: <agent> (the robot that will place the object), <item> (the grabbable object to be placed), <surface> (the surface where the object will be placed), <room> (the room where the action takes place).
            Preconditions: the <agent>, the <surface>, and the <item> must be in the same room, and the <agent> must be holding the <item>.
            Postconditions: the <agent> is no longer holding the <item>, the <item> is on the <surface>, and the <agent> is free.
        - take_from(<agent>, <item>, <surface>, <room>): This action allows a robot to take a grabbable object from a surface in a room.
            Arguments: <agent> (the robot that will take the object), <item> (the grabbable object to be taken), <surface> (the surface where the object is located), <room> (the room where the action takes place).
            Preconditions: the <agent>, the <surface>, and the <item> must be in the same room, and the <agent> must be free (not holding anything).
            Postconditions: the <agent> is holding the <item>, the <item> is no longer on the <surface>, and the <agent> is no longer free.
        - put_in(<agent>, <object>, <container>, <room>): The robot places an object into a movable container.
            Arguments: <agent> (the robot), <object> (the object to be placed), <container> (the container), <room> (the room where the container is located).
            Preconditions: The <agent>, the <object>, and the <container> must be in the same <room>, and the <agent> must be holding the <object>.
            Postconditions: The <agent> is no longer holding the <object>, the <object> is inside the <container>, and the <agent> is free.
        - take_out(<agent>, <object>, <container>, <room>): The robot removes an object from a movable container.
            Arguments: <agent> (the robot), <object> (the object to be removed), <container> (the container), <room> (the room where the container is located).
            Preconditions: The <agent> and the <container> are in the same <room>, the <object> is inside the <container>, and the <agent> is free to grab.
            Postconditions: The <object> is no longer inside the <container>, the <agent> is holding the <object>, and the <agent> is no longer free.
        """
    elif task == "general":
        actions = """
        Additional ENVIRONMENT FUNCTIONS:
        * Additional action(s):
        - put_on(<agent>, <item>, <surface>, <room>): This action allows a robot to place a grabbable object onto a surface in a room.
            Arguments: <agent> (the robot that will place the object), <item> (the grabbable object to be placed), <surface> (the surface where the object will be placed), <room> (the room where the action takes place).
            Preconditions: the <agent>, the <surface>, and the <item> must be in the same room, and the <agent> must be holding the <item>.
            Postconditions: the <agent> is no longer holding the <item>, the <item> is on the <surface>, and the <agent> is free.
        - take_from(<agent>, <item>, <surface>, <room>): This action allows a robot to take a grabbable object from a surface in a room.
            Arguments: <agent> (the robot that will take the object), <item> (the grabbable object to be taken), <surface> (the surface where the object is located), <room> (the room where the action takes place).
            Preconditions: the <agent>, the <surface>, and the <item> must be in the same room, and the <agent> must be free (not holding anything).
            Postconditions: the <agent> is holding the <item>, the <item> is no longer on the <surface>, and the <agent> is no longer free.
        """
    else:
        raise ValueError(f"Unknown task: {task}")

    return actions


def plan_prompt(goal_qry, sg_qry, add_env_func_qry):
    content = "You are an excellent graph planning agent. Given a graph representation of an environment, you can use this graph to generate a step-by-step task plan that the agent can follow to solve a given instruction."

    prompt = f"""
    We have now switched to planning mode.
    
    ENVIRONMENT FUNCTIONS:
    There are the following object types in the environment: agent, room, item.
    Note that robot is an instance of agent, and the robot can perform the following basic actions:
    - move_to(<agent>, <room_1>, <room_2>): This action moves the agent from one room to another. \
        Arguments: <agent> (the <agent> to be moved), <room_1> (the room the <agent> is currently in), <room_2> (the room the <agent> will move to).\
        Preconditions: the <agent> must be in the <room_1> room. \
        Postconditions: the <agent> is no longer in the <room_1> room and is now in the <room_2> room.
    - grab(<agent>, <item>, <room>): This action allows a <agent> to grab a grabbable <item> in a <room>. \
        Arguments: <agent> (the <agent> that will grab the object), <item> (the grabbable object to be grabbed), <room> (the room where the object is located). \
        Preconditions: the <agent> and the <item> must be in the same room, and the <agent> must be free (not holding anything). \
        Postconditions: the <agent> is holding the <item>, and the <item> is no longer in the <room>.
    - drop(<agent>, <item>, <room>): This action allows a <agent> to drop a grabbable <item> it is holding in a <room>. \
        Arguments: <agent> (the <agent> that will drop the object), <item> (the grabbable object to be dropped), <room> (the room where the object will be dropped). \
        Preconditions: the <agent> must be in the <room> and holding the <item>. \
        Postconditions: the <agent> is no longer holding the <item>, the <agent> is free, and the <item> is now in the <room>.

    OUTPUT RESPONSE FORMAT:
    "mode": "planning",
    "chain_of_thought": "break your problem down into a series of intermediate reasoning steps to help you determine your next command",
    "reasoning": "justify why the next action is important",
    "plan": "high-level task plan, which only consists of actions listed in the ENVIRONMENT FUNCTIONS above"
    
    EXAMPLE: \n{PLAN_EXAMPLE_PROMPT}
    
    QUERY:
    New instruction: {goal_qry}
    Previously explored 3D scene graph: \n{sg_qry}
    {add_env_func_qry}
    Please generate the output with according to the given OUTPUT RESPONSE FORMAT directly without further explanations. Make sure to use the term "robot" in the plan instead of "agent".
    """
    
    return content, prompt

# def replan_prompt(err_info: str):
#     content = "You are an excellent replanner. Given an error message, you can fix the previous plan to recover from the error."
#     prompt = f"""
#     Scene Graph Simulator (Feedback): {err_info}\n
#     Fix the plan.
#     """
    
#     return content, prompt

def replan_prompt():
    content = "You are an excellent replanner. If you reached this state it means that the previous plan failed to execute successfully."
    prompt = f"""
    Plan unsuccessfull. Fix the plan.
    """
    
    return content, prompt

# TODO think about states