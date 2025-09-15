from pathlib import Path
from typing import Dict, Set, List, Tuple, Union
import random
from collections import defaultdict

import numpy as np

from agent import llm_call, local_llm_call

random.seed(412)
np.random.seed(423)

def read_graph_from_path(path: Path) -> Dict:
    """
    Read 3DSG from file (.npz) and returns it stored in a dictionary
    
    :param path: Path to the .npz file 
    :return: Dictionary containing the 3DSG
    """
    
    #print(path)
    assert (
        isinstance(path, Path)
    ), "Input file is not a Path"
    assert (
        str(path).endswith(".npz")
    ), "Input file is not .npz object"
    
    graph = np.load(path, allow_pickle=True)['output'].item()
    
    keeps = set(["object", "room"])
    graph = filter_graph(graph, keeps)
    
    return graph

def add_descriptions_to_objects(graph):
    """
    Adds the "description" property to each object in the graph.

    :param graph: Dictionary containing the 3DSG
    :return: Updated graph with descriptions for the objects
    """
    print("Adding descriptions to objects")
    for obj_id, obj in graph["object"].items():
        print("the object is: ", obj["class_"])
        obj["description"] = local_llm_call("Given an object, provide a brief description (max 30 words) including color, material, and what it can be used for. Format: 'color object_name made of material, used for action'. Be concise.", 
                        question="the object is " + obj["class_"])
        print("the description is: ", obj["description"])
    return graph

def save_graph(graph: Dict, path: str):
    """
    Save the 3DSG graph to a file
    
    :param graph: Dictionary containing the 3DSG
    :param path: Path to the file
    """
    print(path)
    np.savez(path, output=graph)

def save_graph_json(graph: Dict, path: str):
    """
    Save the 3DSG graph to a JSON file with detailed room and object information
    
    :param graph: Dictionary containing the 3DSG
    :param path: Path to the JSON file (without extension)
    """
    import json
    
    rooms = graph.get("room", {})
    objects = graph.get("object", {})
    
    # Create structured JSON data
    json_data = {
        "scene_info": {
            "total_rooms": len(rooms),
            "total_objects": len(objects)
        },
        "rooms": {},
        "objects": {}
    }
    
    # Process rooms
    for room_id, room_info in rooms.items():
        json_data["rooms"][room_id] = {
            "id": room_id,
            "category": room_info.get('scene_category', 'unknown'),
            "label": f"{room_info.get('scene_category', 'UnnamedRoom')}_{room_id}",
            "bbox": room_info.get('bbox', []).tolist() if hasattr(room_info.get('bbox', []), 'tolist') else room_info.get('bbox', []),
            "objects_in_room": []
        }
    
    # Process objects
    for obj_id, obj_info in objects.items():
        parent_room = obj_info.get('parent_room')
        obj_data = {
            "id": obj_id,
            "class": obj_info.get('class_', 'unknown'),
            "label": f"{obj_info.get('class_', 'UnnamedObject')}_{obj_id}",
            "parent_room": parent_room,
            "description": obj_info.get('description', ''),
            "bbox": obj_info.get('bbox', []).tolist() if hasattr(obj_info.get('bbox', []), 'tolist') else obj_info.get('bbox', []),
            "action_affordance": obj_info.get('action_affordance', [])
        }
        
        json_data["objects"][obj_id] = obj_data
        
        # Add object to its parent room
        if parent_room and parent_room in json_data["rooms"]:
            json_data["rooms"][parent_room]["objects_in_room"].append(obj_id)
    
    # Save to JSON file
    json_path = f"{path}.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)
    
    print(f"Saved JSON scene graph to: {json_path}")


def filter_graph(graph: Dict, labels: Set[str]) -> Dict:
    """
    Filter the 3DSG graph to only include objects with labels in the 'labels' set
    
    :param graph: Dictionary containing the 3DSG
    :param labels: Set of labels (strings) to consider, e.g. {"room", "object"}
    :return new_graph: Dictionary containing the filtered 3DSG
    """
    new_graph = {}
    for key, item in graph.items():
        if key in labels:
            new_graph[key] = item
    return new_graph

def get_verbose_scene_graph(graph: Dict, as_string: bool = True) -> Union[str, Dict]:
    """
    Given a 3DSG, return a verbose, discursive description of the scene graph, or a dict.

    :param graph: Dictionary containing the 3DSG
    :param as_string: Whether to output as a string or dict
    :return: String with the verbose scene graph or dict mapping rooms to objects
    """
    rooms = graph.get("room", {})
    objects = graph.get("object", {})

    # 1. Create labels for rooms and objects
    room_id_to_label = {
        r_id: f"{info.get('scene_category', 'UnnamedRoom')}_{r_id}"
        for r_id, info in rooms.items()
    }
    obj_id_to_label = {
        o_id: f"{info.get('class_', 'UnnamedObject')}_{o_id}"
        for o_id, info in objects.items()
    }

    # 2. Group objects by their parent room
    room_to_objects = defaultdict(list)
    for o_id, info in objects.items():
        r_id = info.get('parent_room')
        if r_id in rooms:
            label = obj_id_to_label[o_id]
            desc = info.get('description', None)
            room_to_objects[r_id].append((label, desc))

    if not as_string:
        # Return raw dict
        return {
            room_id_to_label[r_id]: room_to_objects.get(r_id, [])
            for r_id in rooms
        }

    # 3. Build discursive string
    # List of room labels in order
    labels = [room_id_to_label[r_id] for r_id in rooms]
    # Intro sentence
    output = []
    output.append(
        "The 3DSG is made of these rooms: " + ", ".join(labels) + "."
    )

    # One sentence per room
    for r_id in rooms:
        room_label = room_id_to_label[r_id]
        items = room_to_objects.get(r_id, [])
        if items:
            # Describe objects
            parts = [f"{name} ({desc})" if desc is not None else f"{name}" for name, desc in items]
            line = f"The {room_label} contains " + ", ".join(parts) + "."
        else:
            line = f"The {room_label} has no objects."
        output.append(line)

    # Join into single paragraph
    return "\n".join(output)

def add_objects(graph, dict_objects):
    """
    Adds random objects to random rooms in the graph.

    :param graph: Dictionary containing the 3DSG
    :param dict_objects: Dictionary of objects to add, where keys are categories and values are lists of object names
    :return: Updated graph with new objects
    """
    
    possible_objects = []
    for _, obj_value in dict_objects.items():
        for obj in obj_value:
            possible_objects.append({"class_": obj, "action_affordance": []})

    # Find max ID
    if graph["object"]:
        max_id = max(graph["object"].keys())
    else:
        max_id = 0

    # get existing rooms
    room_ids = list(graph["room"].keys())

    # add random objects
    for obj_data in possible_objects:
        obj_id = max_id + 1
        obj_type = obj_data["class_"]
        room_id = random.choice(room_ids)  # get random room

        # Get room center and add random epsilon
            
        room_center = graph["room"][room_id].get("location", np.zeros(3))
        epsilon = np.random.uniform(-0.5, 0.5, 3)
        location = np.array(room_center + epsilon)

        # create new object
        new_object = {
            "id": obj_id,
            "class_": obj_data["class_"],
            "action_affordance": obj_data["action_affordance"],
            "location": location,
            "parent_room": room_id,
        }

        # add to the graph
        graph["object"][obj_id] = new_object
        max_id += 1  # update max_id

    return graph


def choose_random_room(graph):
    return random.choice(get_room_names(graph))


def get_room_names(graph):
    return [room_data["scene_category"] + "_" + str(room_id) for room_id, room_data in graph["room"].items()]


def add_objects_to_specific_rooms(graph, objects_config):
    """
    Adds objects to specific rooms in the graph based on the new configuration format.
    
    :param graph: Dictionary containing the 3DSG
    :param objects_config: Dictionary with format:
        {
            "object_name": {"rooms": ["room_type1", "room_type2"], "count": int},
            ...
        }
    :return: Updated graph with new objects
    """
    
    # Find max ID
    if graph["object"]:
        max_id = max(graph["object"].keys())
    else:
        max_id = 0

    # Get all rooms with their types
    room_mapping = {}  # room_type -> list of room_ids
    for room_id, room_data in graph["room"].items():
        room_type = room_data["scene_category"]
        if room_type not in room_mapping:
            room_mapping[room_type] = []
        room_mapping[room_type].append(room_id)

    # Add objects based on configuration
    for obj_name, config in objects_config.items():
        target_room_types = config["rooms"]
        count = config["count"]
        
        # Find available rooms of the specified types
        available_rooms = []
        for room_type in target_room_types:
            if room_type in room_mapping:
                available_rooms.extend(room_mapping[room_type])
        
        if not available_rooms:
            print(f"Warning: No rooms of types {target_room_types} found for object {obj_name}")
            continue
            
        # Add the specified number of objects
        for _ in range(count):
            obj_id = max_id + 1
            
            # Choose a room (randomly from available rooms if multiple)
            room_id = random.choice(available_rooms)
            
            # Get room center and add random epsilon
            room_center = graph["room"][room_id].get("location", np.zeros(3))
            epsilon = np.random.uniform(-0.5, 0.5, 3)
            location = np.array(room_center + epsilon)

            # Create new object
            new_object = {
                "id": obj_id,
                "class_": obj_name,
                "action_affordance": [],
                "location": location,
                "parent_room": room_id,
            }

            # Add to the graph
            graph["object"][obj_id] = new_object
            max_id += 1  # update max_id

    return graph