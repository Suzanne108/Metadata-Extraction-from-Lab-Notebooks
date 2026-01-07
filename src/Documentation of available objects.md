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
*prompt used*: planner v2, ISA_updater v2.


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


