# Install
We recommend using [VS Code + devcontainer](https://code.visualstudio.com/docs/devcontainers/containers). You need to install the devcontainers extension to VS Code. After that, when you open this workspace in Code, there will be a pop-up window asking if you want to re-open it in devcontainer. If you missed it, use cmd+shift+p, and select 
"Dev Containers: Reopen in Container". Make sure you have a docker environment running. 

Once the workspace is reopened inside a container (to make sure it is inside a container, launch a terminal and check the user name and node name), you may need to manually install two extensions to the "new VS code", i.e., Python (ms-python.python) and Jupyter (ms-toolsai.jupyter).

You need a valid token for OpenAI API, stored in ./openai.credential. Other LLM models will be supported later.

# Structure
- [./inswitch](inswitch/): generic components for inSwitch
- [./usecases](usecases/): data, scripts, components, etc., for different use cases

# Run
Most of the running examples and demos are in Jupyter Notebooks. To run them, you need to select a kernal - choose a python environment called "base (Python 3.12.7)". The version may vary. It is the under path "/opt/conda/bin/python".

If you face problem running python scripts, you may also try to select this same "base" python as the Python Interpreter (this can be done also via cmd+shift+p).

Open [http://localhost:5000](http://localhost:5000) to view it in your browser.

# Frontend

A basic **React frontend** is available to interact with the API and test its endpoints easily.

## Repo

[https://github.com/baptisterambour/intent-front](https://github.com/baptisterambour/intent-front)

## How to run it

```bash
git clone https://github.com/baptisterambour/intent-front.git
cd intent-front
npm install
npm start
```

# GraphDB

The API interacts with a GraphDB instance for storing and querying RDF data.

- GraphDB runs on: [http://localhost:7200](http://localhost:7200)
- Default repository name: `intent-db`
- SPARQL endpoint: `http://localhost:7200/repositories/intent-db`
- Workbench (UI): [http://localhost:7200](http://localhost:7200)

Make sure GraphDB is running before starting the API. You can use Docker to launch it, for example:

```bash
docker run -d --name graphdb -p 7200:7200 ontotext/graphdb:10.6.4
```