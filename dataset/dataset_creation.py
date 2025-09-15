import os
from pprint import pprint
import json

from dataset_utils import *
import argparse

DATASET_DIR = os.path.dirname(os.path.abspath(__file__))

def generate_splits(selected_dataset_splits, DESCRIPTION=False):
    for root_dir, dirs, files in os.walk(DATASET_DIR):
        for file in files:
            # Load JSON task configuration file
            if file.endswith(".json") and file.split(".")[0] in selected_dataset_splits:
                print(f"Generating split for task: {file}")
                path = os.path.join(root_dir, file)

                # Get file name without extension
                task_name = os.path.splitext(file)[0]

                # Load json path
                with open(path) as f:
                    data = json.load(f)
                    problems = data["problems"]
                    f.close()

                # Create a dir for the current task, this dir will contain all the scenes related to this task
                domain_dir = os.path.join(task_name)
                os.makedirs(domain_dir, exist_ok=True)

                for problem_id, problem_data in problems.items():
                    objects = problem_data["objects"]
                    graphs = problem_data["graph"]
                    for graph_id in graphs:
                        print(graph_id)

                        scene_dir = os.path.join(domain_dir, graph_id)
                        os.makedirs(scene_dir, exist_ok=True)
                        
                        problem_dir = os.path.join(scene_dir, problem_id)
                        if not os.path.exists(problem_dir):
                            os.makedirs(problem_dir, exist_ok=True)

                            path_graph = os.path.join("3dscenegraph", "3DSceneGraph_" + graph_id + ".npz")
                            print(path_graph)

                            graph = read_graph_from_path(Path(path_graph))
                            # pprint(graph)
                            print(problem_data["description"])

                            # Check if objects use the new format (to_add with specific rooms) or old format
                            if "to_add" in objects and isinstance(objects["to_add"], dict) and objects["to_add"]:
                                # New format: objects with specific rooms
                                graph = add_objects_to_specific_rooms(graph, objects["to_add"])
                            elif objects and any(isinstance(v, list) for v in objects.values()):
                                # Old format: random placement
                                graph = add_objects(graph, objects)
                            
                            if DESCRIPTION:
                                graph = add_descriptions_to_objects(graph)

                            #save_graph(graph, os.path.join(problem_dir, graph_id+"_"+problem_id))
                            save_graph(graph, os.path.join(problem_dir, graph_id))
                            
                            # Save JSON version of the scene graph
                            save_graph_json(graph, os.path.join(problem_dir, graph_id))

                            task_path = os.path.join(problem_dir, "task.txt")
                            with open(task_path, "w") as f:
                                f.write(problem_data["goal"])
                                f.close()
                            
                            description_path = os.path.join(problem_dir, "description.txt")
                            with open(description_path, "w") as f:
                                f.write(problem_data["description"])
                                f.close()

                            init_loc_path = os.path.join(problem_dir, "init_loc.txt")
                            with open(init_loc_path, mode="w") as f:
                                random_loc = choose_random_room(graph)
                                print(random_loc)
                                f.write(random_loc)
                            
                            print("\n\n")
                        else:
                            print("Already exists")

if __name__=="__main__":
    
    DATASET_SPLITS = [
        "general",
        "dining_setup",
        "laundry",
        "office_setup",
        "house_cleaning",
        "dining_setup",
        "pc_assembly"
    ]

    parser = argparse.ArgumentParser(description="Generate dataset splits.")
    parser.add_argument("--description", action="store_true", help="Add descriptions to objects in the graph.")
    args = parser.parse_args()

    DESCRIPTION = args.description

    generate_splits(DATASET_SPLITS, DESCRIPTION)