

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

*performance*:


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


