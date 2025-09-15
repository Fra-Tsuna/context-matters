### Dataset creation instruction
Download the original 3D Scene Graphs from [3D Scene Graph](https://3dscenegraph.stanford.edu/) and place the `.npz` files into
the `3dscenegraph` folder, which should be located at the same directory level as the `dataset_creation.py` script. 
Export your OpenAI API key:
```bash
export OPENAI_API_KEY=<YOUR KEY>
```
After placing the files, you can generate the enhanced dataset by executing the following command:
```bash
python dataset_creation.py --description (optional)
```
The `--description` parameter is optional and allows you to enrich
the 3D Scene Graph with detailed descriptions for each object,
including attributes such as color and material. These descriptions
are generated using a Large Language Model (LLM) prompt.

### Workflow
The`dataset_creation.py` script takes in input the `.json` file in the
root directory. The `.json` file contains a parameter indicating the
input graph and for that graph a set of objects. Each object is
positioned randomly in one of the room of the graph without specify
a precise location inside the room.
The final output will be a graph containing the rooms with the
objects inside them and for each object a description
