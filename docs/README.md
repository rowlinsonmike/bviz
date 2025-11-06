## Bedrock Vision

<p align="center">
<img style="border-radius:8px;" alt="example" src="https://github.com/rowlinsonmike/bviz/raw/refs/heads/main/docs/assets/diagram.png"/>
</p>

### Overview
> 🚧 Actively development 🚧

Bedrock Vision is a project that visualizes Bedrock Converse API conversations. It creates a PDF that contains a diagram of the conversation flow and associated text for each item, including tool calls and results.

### Installation
> [Diagrams](https://diagrams.mingrammer.com/docs/getting-started/installation) library relies on graphviz. macOS users using Homebrew can install Graphviz via `brew install graphviz`. Similarly, Windows users with Chocolatey installed can run `choco install graphviz`.

To install Bedrock Vision, use pip:

```bash
pip install bviz
```

### How to Use
1. Ensure you have the `diagrams` library installed.
2. Execute the cli against a bedrock messages json file:
   ```bash
   bviz -p /path/to/json -n diagram_name
   ```

- `-p` specifies the path to the JSON file containing the Bedrock conversation.
- `-n` specifies the name of the generated diagram.
