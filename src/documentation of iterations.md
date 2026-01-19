# Guidelines of saving versions
As a general rule, code of versions that were unsuccesfull to adhere to the minimum requirements is not kept or maintained. It might also change to adapt to later versions.

# Agents

### 1. simple_agent

*prompt*: simple_prompt v1

*tools*: doc_chunk_tool


### 2. standalone_extraction_agent

*prompt*: extract_ISA v1

*tools*: doc_chunk_tool, validate_ISA_schema, schema_getter_tools


# Workflows

### 1. simple_workflow

Uses simple_agent_v1 to simply extract the title and authors from a document.

### 2. standalone_agent_workflow

Uses simple_agent_v2 to attempt to extract ISA metadata given a provided document.

*notes*: this is also prototype version 5.

*state*: v1

### 3. stateful_graph_agent

Implement a stateful graph of one planner node, a validation, a schema fetcher, a document reader and an ISA composer node.

*notes*: This is also prototype version 6. 
*state*: v2

In version 6.1 the planner had no access to a histroy of messages, so it did not know what was done before. It kept making a simple mistake over and over again. Logs of this version can be found under output/stateful_graph/version 6.1 - without history.

For version 6.2 we update the state to include a history and give this to the planner. We see that this prevents the planner from repeating the same mistakes again. However, we notice that the planner and the ISA_updater argue about what to do. The planner instructs the ISA_update to draft something with empty fields to have something to begin with. The ISA_update refuses this because it has not received any evidence yet. In attempt to solve this, we will include a field in the prompt of the ISA_updater with instructions from the planner. This hopefully results in better communication between the agents.

*prompt used*: planner v1, ISA_updater v1

Version 6.3
*prompt used*: planner system-v2. inference v1-, ISA_updater system-v2, inference-v2.

We incorporated a way for the planner to give instructions to the ISA schema updater. Observations:
- Token limit exceeded for planner prompt. We will solve this by limiting the 'evidence' field in the prompt to only the keywords that were used to search for the evidence. We will also remove some whitespace.
- The planner agent instructs to perform 'search_document' action for about the same query. To solve this, we will give the planner prompt stricter instructions on how to act based on the given information, limiting the freedom of the agent.
- The schema getter functions are not really necessary in the overall workflow, because the isa_updater agent already has access to this tool individually. So in the next iteration, we will remove this as an option for the planner to choose. But then how does the planner know what to search in the document? Maybe this is a design flaw: the planner does not know what to search in the document, because it does not read the schema's. We will solve this problem by wrapping an agent around the schema getter functions. The agent will read the schema and summarize for the planner what kind of information needs to be extracted in order to complete the isa metadata. Since this a major change in the system design, we will move to version 7.

Version 7.0
*prompt used*: planner system-v3, policy-v1, inference-v2, ISA_updater system-v3, inference v-2, schema_agent system-v1, inference-v1.

What changed:
- Agent around schema getter functions, which can answer planner's questions about the schema
- Changed planner system and inference prompts. Give stricter instructions on how to search document, remove whitespace, limit evidence field
- state - remove schema cache
- For planner made a split between system and policy prompt.

Observations:
- planner agent is going to the same steps (schema agent) with the same argument multiple times. Waste of resources and not progressing. 
- The planner agent does not seem to properly break down the task, but instead asks for the entire schema all at once.

Version 8.0
*prompt used*: 
- isa_overview system-v1, inference v1
- isa_updater system-v3, inference-v2
- planner system-v4, policy-v2, inference-v3
- task decomposer, system-v1, inference-v1.

What changed:
Added nodes:
- isa_overview -> to introduce what isa is to the next node - Agent call
- document_overview -> to give a summary of the document and identify the main objects to be documented. - LLM call
- task_decomposer -> to construct atomic tasks from the document_overview output - LLM call
- task_progresser -> responsible for marking tasks as complete and moving on to next task - deterministic node
Removed nodes:
- schema_agent -> planner agent does not need to be aware of schema's in this configuration.

The main change is that we start identifying atomatic subtasks to execute to complete the entire task. The planner agent is now responsible for executing only the identified subtasks, instead of being responsible for the overall system. Implemented the heuristic of task decomposition.


Observations:
- The choices of the planner agent are unreliable, prematurely ending the task, or choosing the same task over and over again. In the next version, 8.1, some guardrails will be implemented based on the previous message. We will make the planner node partly deterministic.
- The document overview agent does not always identify the same amount of studies and assays.

Version 8.1
*prompt used*:
- isa_overview system-v1, inference v1
- isa_updater system-v3, inference-v2
- planner system-v5, policy-v3, inference-v2
- task decomposer, system-v1, inference-v1.

What changed:
- Added constraints to planner agent. Some transactions are now automatic, so that the LLM does not have the authority to change this. All tasks completed --> STOP. Update draft --> validate, validate + succes = finish task

Observations:
- Able to properly iterate through the task steps because of automatic transactions
- No real feedback yet on the content of the isa metadata instead of just the structure, the content might be faulty
- Trouble with onotology annotations --> this is not possible yet and the system just makes things up
- A lot of 'mistakes' stem from misunderstanding of the purpose of different ISA objects. In next version add official ISA abstract model specification to node ISA_overview, instead of just JSON specifications of ISA objects.

Version 8.2
*prompts used*
- isa_overview system-v1, inference v1
- isa_updater system-v3, inference-v2
- planner system-v5, policy-v3, inference-v2
- task decomposer, system-v1, inference-v1.

What changed:
- added offical ISA abstract model specification to node ISA_overview, instead of just JSON specifications of ISA objects. In hope that this leads to better understanding of ISA objects functions.


# Tools

### 1. get_doc_chunk

Given a query, retrieves k chunks from the contextual document which are semantically close to the query.

### 2. validate_ISA_schema

Given an ISA JSON object, validates it against the official ISA schema. Returns a validation report.

### 3. schema_getter_tools

For each object included in the ISA schema, a tool is created to fetch the JSON schema of this object.

# Prompts

### Simple
These are general prompts instructing agents to answer questions.

### extract_ISA
These are prompt which instruct agents to extract ISA. Currently there is one version of this prompt type. This prompt includes general steps an agent should perform in order to properly extract ISA metadata.


