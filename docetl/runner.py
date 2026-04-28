"""
The DSLRunner module implements a declarative pipeline execution engine with a pull-based
evaluation model. Key architectural decisions include:

Design Patterns:
- Pull-based DAG: Operations are lazily evaluated only when their outputs are needed,
  enabling efficient resource utilization and caching
- Dependency Injection: Operations receive their dependencies through a standardized interface,
  making the system modular and testable
- Builder Pattern: Pipeline construction is separated from execution, allowing validation
  and optimization before runtime

Core Features:
- Transparent Caching: Automatic checkpointing and reuse of intermediate results
- Cost Tracking: Built-in tracking of operation costs for optimization
- Schema Validation: Type checking and schema validation at both build and runtime
- Extensible Operations: New operations can be added by implementing the operation interface

The architecture prioritizes:
1. Separation of Concerns: Building, validation, and execution are distinct phases
2. Flexibility: Support for both streaming and batch processing patterns
3. Observability: Rich logging and cost tracking throughout execution
4. Performance: Lazy evaluation and caching optimize resource usage
"""

from __future__ import annotations

import functools
import hashlib
import json
import os
import time
from collections import defaultdict
from typing import Any

from dotenv import load_dotenv
from pydantic import BaseModel
from rich.markup import escape
from rich.panel import Panel

from docetl.config_wrapper import ConfigWrapper
from docetl.containers import OpContainer, StepBoundary
from docetl.dataset import Dataset, create_parsing_tool_map
from docetl.operations import get_operation, get_operations
from docetl.operations.base import BaseOperation
from docetl.optimizer import Optimizer
from docetl.storage import StorageBackend, canonical_json_hash, get_default_backend, storage_path_join

from . import schemas
from .utils import classproperty

load_dotenv()


class DSLRunner(ConfigWrapper):
    """
    DSLRunner orchestrates pipeline execution by building and traversing a DAG of OpContainers.
    The runner uses a two-phase approach:

    1. Build Phase:
       - Parses YAML config into a DAG of OpContainers
       - Each operation becomes a node connected to its dependencies
       - Special handling for equijoins which have two parent nodes
       - Validates operation syntax and schema compatibility

    2. Execution Phase:
       - Starts from the final operation and pulls data through the DAG
       - Handles caching/checkpointing of intermediate results
       - Tracks costs and execution metrics
       - Manages dataset loading and result persistence

    The separation between build and execution phases allows for:
    - Pipeline validation before any execution
    - Cost estimation and optimization
    - Partial pipeline execution for testing
    """

    @classproperty
    def schema(cls):
        # Accessing the schema loads all operations, so only do this
        # when we actually need it...
        # Yes, this means DSLRunner.schema isn't really accessible to
        # static type checkers. But it /is/ available for dynamic
        # checking, and for generating json schema.

        OpType = functools.reduce(
            lambda a, b: a | b, [op.schema for op in get_operations().values()]
        )
        # More pythonic implementation of the above, but only works in python 3.11:
        # OpType = Union[*[op.schema for op in get_operations().values()]]

        class Pipeline(BaseModel):
            config: dict[str, Any] | None
            parsing_tools: list[schemas.ParsingTool] | None
            datasets: dict[str, schemas.Dataset] | None = None
            retrievers: dict[str, Any] | None
            operations: list[OpType]
            pipeline: schemas.PipelineSpec

        return Pipeline

    @classproperty
    def json_schema(cls):
        return cls.schema.model_json_schema()

    def __init__(self, config: dict, max_threads: int | None = None, **kwargs):
        """
        Initialize the DSLRunner with a YAML configuration file.

        Args:
            max_threads (int, optional): Maximum number of threads to use. Defaults to None.
        """
        super().__init__(
            config,
            base_name=kwargs.pop("base_name", None),
            yaml_file_suffix=kwargs.pop("yaml_file_suffix", None),
            max_threads=max_threads,
            **kwargs,
        )
        self.total_cost = 0
        self.total_token_usage = defaultdict(
            lambda: {"prompt_tokens": 0, "completion_tokens": 0}
        )
        self._initialize_state()
        self._setup_parsing_tools()
        self._setup_retrievers()
        self._build_operation_graph(config)
        self._compute_operation_hashes()

        # Run initial validation
        self._from_df_accessors = kwargs.get("from_df_accessors", False)
        if not self._from_df_accessors:
            self.syntax_check()

    def _initialize_state(self) -> None:
        """Initialize basic runner state and datasets"""
        self.datasets = {}
        self.intermediate_dir = (
            self.config.get("pipeline", {}).get("output", {}).get("intermediate_dir")
        )
        # Storage backend — uses env vars DOCETL_STORAGE_URL / DOCETL_CACHE_DIR by default
        self.storage = get_default_backend()

    def _setup_parsing_tools(self) -> None:
        """Set up parsing tools from configuration"""
        self.parsing_tool_map = create_parsing_tool_map(
            self.config.get("parsing_tools", None)
        )

    def _setup_retrievers(self) -> None:
        """Instantiate retrievers from configuration (lazy index creation)."""
        from docetl.retrievers.lancedb import LanceDBRetriever

        self.retrievers: dict[str, Any] = {}
        retrievers_cfg = self.config.get("retrievers", {}) or {}
        for name, rconf in retrievers_cfg.items():
            if not isinstance(rconf, dict):
                raise ValueError(f"Invalid retriever '{name}' configuration")
            if rconf.get("type") != "lancedb":
                raise ValueError(
                    f"Unsupported retriever type '{rconf.get('type')}' for '{name}'. Only 'lancedb' is supported."
                )
            required = ["dataset", "index_dir", "index_types"]
            for key in required:
                if key not in rconf:
                    raise ValueError(
                        f"Retriever '{name}' missing required key '{key}'."
                    )
            # Defaults
            rconf.setdefault("query", {"top_k": 5})
            rconf.setdefault("build_index", "if_missing")

            self.retrievers[name] = LanceDBRetriever(self, name, rconf)

    def _build_operation_graph(self, config: dict) -> None:
        """Build the DAG of operations from configuration"""
        self.config = config
        self.op_container_map = {}
        self.last_op_container = None

        for step in self.config["pipeline"]["steps"]:
            self._validate_step(step)

            if step.get("input"):
                self._add_scan_operation(step)
            elif step["operations"] and isinstance(step["operations"][0], dict):
                self._add_equijoin_operation(step)
            else:
                # No input specified and not an equijoin: use synthetic empty dataset [{}]
                self._add_empty_scan_operation(step)

            self._add_step_operations(step)
            self._add_step_boundary(step)

    def _validate_step(self, step: dict) -> None:
        """Validate step configuration"""
        assert "name" in step.keys(), f"Step {step} does not have a name"
        assert "operations" in step.keys(), f"Step {step} does not have `operations`"

    def _add_scan_operation(self, step: dict) -> None:
        """Add a scan operation for input datasets"""
        scan_op_container = OpContainer(
            f"{step['name']}/scan_{step['input']}",
            self,
            {
                "type": "scan",
                "dataset_name": step["input"],
                "name": f"scan_{step['input']}",
            },
        )
        self.op_container_map[f"{step['name']}/scan_{step['input']}"] = (
            scan_op_container
        )
        if self.last_op_container:
            scan_op_container.add_child(self.last_op_container)
        self.last_op_container = scan_op_container

    def _add_empty_scan_operation(self, step: dict) -> None:
        """Add a scan operation for a synthetic empty dataset [{}]"""
        dataset_name = "__empty__"
        scan_op_container = OpContainer(
            f"{step['name']}/scan_{dataset_name}",
            self,
            {
                "type": "scan",
                "dataset_name": dataset_name,
                "name": f"scan_{dataset_name}",
            },
        )
        self.op_container_map[f"{step['name']}/scan_{dataset_name}"] = scan_op_container
        if self.last_op_container:
            scan_op_container.add_child(self.last_op_container)
        self.last_op_container = scan_op_container

    def _add_equijoin_operation(self, step: dict) -> None:
        """Add an equijoin operation with its scan operations"""
        equijoin_operation_name = list(step["operations"][0].keys())[0]
        left_dataset_name = list(step["operations"][0].values())[0]["left"]
        right_dataset_name = list(step["operations"][0].values())[0]["right"]

        left_scan_op_container = OpContainer(
            f"{step['name']}/scan_{left_dataset_name}",
            self,
            {
                "type": "scan",
                "dataset_name": left_dataset_name,
                "name": f"scan_{left_dataset_name}",
            },
        )
        if self.last_op_container:
            left_scan_op_container.add_child(self.last_op_container)
        right_scan_op_container = OpContainer(
            f"{step['name']}/scan_{right_dataset_name}",
            self,
            {
                "type": "scan",
                "dataset_name": right_dataset_name,
                "name": f"scan_{right_dataset_name}",
            },
        )
        if self.last_op_container:
            right_scan_op_container.add_child(self.last_op_container)
        equijoin_op_container = OpContainer(
            f"{step['name']}/{equijoin_operation_name}",
            self,
            self.find_operation(equijoin_operation_name),
            left_name=left_dataset_name,
            right_name=right_dataset_name,
        )

        equijoin_op_container.add_child(left_scan_op_container)
        equijoin_op_container.add_child(right_scan_op_container)

        self.last_op_container = equijoin_op_container
        self.op_container_map[f"{step['name']}/{equijoin_operation_name}"] = (
            equijoin_op_container
        )
        self.op_container_map[f"{step['name']}/scan_{left_dataset_name}"] = (
            left_scan_op_container
        )
        self.op_container_map[f"{step['name']}/scan_{right_dataset_name}"] = (
            right_scan_op_container
        )

    def _add_step_operations(self, step: dict) -> None:
        """Add operations for a step"""
        # Skip first op only for equijoin (first op is a dict, not a string)
        is_equijoin = (
            step.get("input") is None
            and step["operations"]
            and isinstance(step["operations"][0], dict)
        )
        op_start_idx = 1 if is_equijoin else 0

        for operation_name in step["operations"][op_start_idx:]:
            if not isinstance(operation_name, str):
                raise ValueError(
                    f"Operation {operation_name} in step {step['name']} should be a string. "
                    "If you intend for it to be an equijoin, don't specify an input in the step."
                )

            op_container = OpContainer(
                f"{step['name']}/{operation_name}",
                self,
                self.find_operation(operation_name),
            )
            op_container.add_child(self.last_op_container)
            self.last_op_container = op_container
            self.op_container_map[f"{step['name']}/{operation_name}"] = op_container

    def _add_step_boundary(self, step: dict) -> None:
        """Add a step boundary node"""
        step_boundary = StepBoundary(
            f"{step['name']}/boundary",
            self,
            {"type": "step_boundary", "name": f"{step['name']}/boundary"},
        )
        step_boundary.add_child(self.last_op_container)
        self.op_container_map[f"{step['name']}/boundary"] = step_boundary
        self.last_op_container = step_boundary

    def _compute_operation_hashes(self) -> None:
        """Compute hashes for operations to enable caching"""
        op_map = {op["name"]: op for op in self.config["operations"]}
        self.step_op_hashes = defaultdict(dict)

        for step in self.config["pipeline"]["steps"]:
            for idx, op in enumerate(step["operations"]):
                op_name = op if isinstance(op, str) else list(op.keys())[0]

                all_ops_until_and_including_current = (
                    [op_map[prev_op] for prev_op in step["operations"][:idx]]
                    + [op_map[op_name]]
                    + [self.config.get("system_prompt", {})]
                )

                for op in all_ops_until_and_including_current:
                    if "model" not in op:
                        op["model"] = self.default_model

                all_ops_str = json.dumps(all_ops_until_and_including_current)
                self.step_op_hashes[step["name"]][op_name] = hashlib.sha256(
                    all_ops_str.encode()
                ).hexdigest()

    def get_output_path(self, require=False):
        output_path = self.config.get("pipeline", {}).get("output", {}).get("path")
        if output_path:
            if not (
                output_path.lower().endswith(".json")
                or output_path.lower().endswith(".csv")
            ):
                raise ValueError(
                    f"Output path '{output_path}' is not a JSON or CSV file. Please provide a path ending with '.json' or '.csv'."
                )
        elif require:
            raise ValueError(
                "No output path specified in the configuration. Please provide an output path ending with '.json' or '.csv' in the configuration to use the save() method."
            )

        return output_path

    def syntax_check(self):
        """
        Perform a syntax check on all operations defined in the configuration.
        """
        self.console.log("[yellow]Checking operations...[/yellow]")

        # Just validate that it's a json file if specified
        self.get_output_path()
        current = self.last_op_container

        try:
            # Walk the last op container to check syntax
            op_containers = []
            if self.last_op_container:
                op_containers = [self.last_op_container]

            while op_containers:
                current = op_containers.pop(0)
                syntax_result = current.syntax_check()
                self.console.log(syntax_result, end="")
                # Add all children to the queue
                op_containers.extend(current.children)
        except Exception as e:
            raise ValueError(
                f"Syntax check failed for operation '{current.name}': {str(e)}"
            )

        self.console.log("[green]✓ All operations passed syntax check[/green]")

    def print_query_plan(self, show_boundaries=False):
        """
        Print a visual representation of the entire query plan using indentation and arrows.
        Operations are color-coded by step to show the pipeline structure while maintaining
        dependencies between steps.
        """
        if not self.last_op_container:
            self.console.log("\n[bold]Pipeline Steps:[/bold]")
            self.console.log(
                Panel("No operations in pipeline", title="Query Plan", width=100)
            )
            self.console.log()
            return

        def _print_op(
            op: OpContainer, indent: int = 0, step_colors: dict[str, str] | None = None
        ) -> str:
            # Handle boundary operations based on show_boundaries flag
            if isinstance(op, StepBoundary):
                if show_boundaries:
                    output = []
                    indent_str = "  " * indent
                    step_name = op.name.split("/")[0]
                    color = step_colors.get(step_name, "white")
                    output.append(
                        f"{indent_str}[{color}][bold]{op.name}[/bold][/{color}]"
                    )
                    output.append(f"{indent_str}Type: step_boundary")
                    if op.children:
                        output.append(f"{indent_str}[yellow]▼[/yellow]")
                        for child in op.children:
                            output.append(_print_op(child, indent + 1, step_colors))
                    return "\n".join(output)
                elif op.children:
                    return _print_op(op.children[0], indent, step_colors)
                return ""

            # Build the string for the current operation with indentation
            indent_str = "  " * indent
            output = []

            # Color code the operation name based on its step
            step_name = op.name.split("/")[0]
            color = step_colors.get(step_name, "white")
            output.append(f"{indent_str}[{color}][bold]{op.name}[/bold][/{color}]")
            output.append(f"{indent_str}Type: {op.config['type']}")

            # Add schema if available
            if "output" in op.config and "schema" in op.config["output"]:
                output.append(f"{indent_str}Output Schema:")
                for field, field_type in op.config["output"]["schema"].items():
                    escaped_type = escape(str(field_type))
                    output.append(
                        f"{indent_str}  {field}: [bright_white]{escaped_type}[/bright_white]"
                    )

            # Add children
            if op.children:
                if op.is_equijoin:
                    output.append(f"{indent_str}[yellow]▼ LEFT[/yellow]")
                    output.append(_print_op(op.children[0], indent + 1, step_colors))
                    output.append(f"{indent_str}[yellow]▼ RIGHT[/yellow]")
                    output.append(_print_op(op.children[1], indent + 1, step_colors))
                else:
                    output.append(f"{indent_str}[yellow]▼[/yellow]")
                    for child in op.children:
                        output.append(_print_op(child, indent + 1, step_colors))

            return "\n".join(output)

        # Get all step boundaries and extract unique step names
        step_boundaries = [
            op
            for name, op in self.op_container_map.items()
            if isinstance(op, StepBoundary)
        ]
        step_boundaries.sort(key=lambda x: x.name)

        # Create a color map for steps - using distinct colors
        colors = ["cyan", "magenta", "green", "yellow", "blue", "red"]
        step_names = [b.name.split("/")[0] for b in step_boundaries]
        step_colors = {
            name: colors[i % len(colors)] for i, name in enumerate(step_names)
        }

        # Print the legend
        self.console.log("\n[bold]Pipeline Steps:[/bold]")
        for step_name, color in step_colors.items():
            self.console.log(f"[{color}]■[/{color}] {step_name}")

        # Print the full query plan starting from the last step boundary
        query_plan = _print_op(self.last_op_container, step_colors=step_colors)
        self.console.log(Panel(query_plan, title="Query Plan", width=100))
        self.console.log()

    def find_operation(self, op_name: str) -> dict:
        for operation_config in self.config["operations"]:
            if operation_config["name"] == op_name:
                return operation_config
        raise ValueError(f"Operation '{op_name}' not found in configuration.")

    def load_run_save(self) -> float:
        """
        Execute the entire pipeline defined in the configuration.
        """
        output_path = self.get_output_path(require=True)

        # Print the query plan
        self.print_query_plan()

        start_time = time.time()

        if self.last_op_container:
            self.load()
            self.console.rule("[bold]Pipeline Execution[/bold]")
            output, _, _, _ = self.last_op_container.next()
            self.save(output)

        execution_time = time.time() - start_time

        # Print execution summary
        token_usage_lines = ""
        if self.total_token_usage:
            token_usage_lines = "\n[bold]Token Usage:[/bold]\n"
            total_prompt = 0
            total_completion = 0
            total_cached = 0
            for model, usage in sorted(self.total_token_usage.items()):
                prompt = usage["prompt_tokens"]
                completion = usage["completion_tokens"]
                cached = usage.get("cached_tokens", 0)
                total_prompt += prompt
                total_completion += completion
                total_cached += cached
                line = f"  {model}: [cyan]{prompt:,}[/cyan] input"
                if cached:
                    line += f" ([dim]{cached:,} cached[/dim])"
                line += f", [cyan]{completion:,}[/cyan] output"
                token_usage_lines += line + "\n"
            if len(self.total_token_usage) > 1:
                total_line = f"  [bold]Total: [cyan]{total_prompt:,}[/cyan] input"
                if total_cached:
                    total_line += f" ([dim]{total_cached:,} cached[/dim])"
                total_line += f", [cyan]{total_completion:,}[/cyan] output[/bold]"
                token_usage_lines += total_line + "\n"

        summary = (
            f"Cost: [green]${self.total_cost:.2f}[/green]\n"
            f"Time: {execution_time:.2f}s\n"
            + token_usage_lines
            + (
                f"Cache: [dim]{self.intermediate_dir}[/dim]\n"
                if self.intermediate_dir
                else ""
            )
            + f"Output: [dim]{output_path}[/dim]"
        )
        self.console.log(Panel(summary, title="Execution Summary"))

        return self.total_cost

    def load(self) -> None:
        """
        Load all datasets defined in the configuration.
        """
        datasets = {}
        self.console.rule("[bold]Loading Datasets[/bold]")

        for name, dataset_config in self.config.get("datasets", {}).items():
            if dataset_config["type"] == "file":
                datasets[name] = Dataset(
                    self,
                    "file",
                    dataset_config["path"],
                    source="local",
                    parsing=dataset_config.get("parsing", []),
                    user_defined_parsing_tool_map=self.parsing_tool_map,
                )
                self.console.log(
                    f"[green]✓[/green] Loaded dataset '{name}' from {dataset_config['path']}"
                )
            elif dataset_config["type"] == "memory":
                datasets[name] = Dataset(
                    self,
                    "memory",
                    dataset_config["path"],
                    source="local",
                    parsing=dataset_config.get("parsing", []),
                    user_defined_parsing_tool_map=self.parsing_tool_map,
                )
                self.console.log(
                    f"[green]✓[/green] Loaded dataset '{name}' from in-memory data"
                )
            else:
                raise ValueError(f"Unsupported dataset type: {dataset_config['type']}")

        self.datasets = {
            name: (
                dataset
                if isinstance(dataset, Dataset)
                else Dataset(self, "memory", dataset)
            )
            for name, dataset in datasets.items()
        }
        # Always register a synthetic empty dataset for pipelines without input data
        self.datasets["__empty__"] = Dataset(self, "memory", [{}])
        self.console.log()

    def save(self, data: list[dict]) -> None:
        """
        Save the final output of the pipeline.
        """
        self.get_output_path(require=True)

        output_config = self.config["pipeline"]["output"]
        if output_config["type"] == "file":
            output_path = output_config["path"]
            parent = storage_path_join(*output_path.replace("\\", "/").split("/")[:-1])
            if parent:
                self.storage.makedirs(parent, exist_ok=True)
            if output_path.lower().endswith(".json"):
                with self.storage.open(output_path, "w") as file:
                    json.dump(data, file, indent=2)
            else:  # CSV
                import csv
                import io as _io

                buf = _io.StringIO()
                writer = csv.DictWriter(buf, fieldnames=data[0].keys())
                limited_data = [
                    {k: d.get(k, None) for k in data[0].keys()} for d in data
                ]
                writer.writeheader()
                writer.writerows(limited_data)
                with self.storage.open(output_path, "w") as file:
                    file.write(buf.getvalue())
            self.console.log(
                f"[green]✓[/green] Saved to [dim]{output_path}[/dim]\n"
            )
        else:
            raise ValueError(
                f"Unsupported output type: {output_config['type']}. Supported types: file"
            )

    def _checkpoint_path(
        self, step_name: str, operation_name: str, upstream_data_hash: str | None
    ) -> str | None:
        """
        Compute content-addressed checkpoint filename.

        When upstream_data_hash is available the filename encodes both op-config
        and upstream data, making it safe to cache forever (no separate registry).
        Falls back to config-only hash when no upstream hash is provided.
        """
        if (
            step_name not in self.step_op_hashes
            or operation_name not in self.step_op_hashes[step_name]
        ):
            return None

        op_config_hash = self.step_op_hashes[step_name][operation_name]
        if upstream_data_hash:
            combined = canonical_json_hash([upstream_data_hash, op_config_hash])
        else:
            combined = op_config_hash
        short = combined[:24]
        return storage_path_join(
            self.intermediate_dir, step_name, f"{operation_name}_{short}.json"
        )

    def _load_from_checkpoint_if_exists(
        self,
        step_name: str,
        operation_name: str,
        upstream_data_hash: str | None = None,
    ) -> tuple[list[dict] | None, str | None]:
        """
        Returns (data, data_hash) if a valid checkpoint exists, else (None, None).

        Content-addressed filenames: a hit means the file exists under the hash
        path; no separate registry (.docetl_intermediate_config.json) is needed.
        """
        if self.intermediate_dir is None or self.config.get("bypass_cache", False):
            return None, None

        checkpoint_path = self._checkpoint_path(
            step_name, operation_name, upstream_data_hash
        )
        if checkpoint_path is None:
            return None, None

        if not self.storage.exists(checkpoint_path):
            return None, None

        ds_key = f"{step_name}_{operation_name}"
        if ds_key not in self.datasets:
            self.datasets[ds_key] = Dataset(self, "file", checkpoint_path, "local")

        self.console.log(
            f"[green]✓[/green] [italic]Loaded checkpoint for operation '{operation_name}' "
            f"in step '{step_name}' from {checkpoint_path}[/italic]"
        )

        loaded = self.datasets[ds_key].load()
        loaded_hash = canonical_json_hash(loaded)
        return loaded, loaded_hash

    def clear_intermediate(self) -> None:
        """
        Clear the intermediate directory.
        """
        if self.intermediate_dir:
            self.storage.rm(self.intermediate_dir, recursive=True)
            return

        raise ValueError("Intermediate directory not set. Cannot clear intermediate.")

    def _save_checkpoint(
        self,
        step_name: str,
        operation_name: str,
        data: list[dict],
        upstream_data_hash: str | None = None,
    ) -> str:
        """
        Save a content-addressed checkpoint and return the data hash.

        The checkpoint filename encodes both op-config and upstream data so it
        is valid forever — no separate registry file required.
        """
        checkpoint_path = self._checkpoint_path(
            step_name, operation_name, upstream_data_hash
        )
        if checkpoint_path is None:
            return canonical_json_hash(data)

        parent = storage_path_join(*checkpoint_path.split("/")[:-1]) if "/" in checkpoint_path else ""
        if parent:
            self.storage.makedirs(parent, exist_ok=True)

        with self.storage.open(checkpoint_path, "w") as f:
            json.dump(data, f)

        self.console.log(
            f"[green]✓ [italic]Intermediate saved for operation '{operation_name}' "
            f"in step '{step_name}' at {checkpoint_path}[/italic][/green]"
        )
        return canonical_json_hash(data)

    def should_optimize(
        self, step_name: str, op_name: str, **kwargs
    ) -> tuple[str, float, list[dict[str, Any]], list[dict[str, Any]]]:
        self.load()

        # Augment the kwargs with the runner's config if not already provided
        kwargs["litellm_kwargs"] = self.config.get("optimizer_config", {}).get(
            "litellm_kwargs", {}
        )
        kwargs["rewrite_agent_model"] = self.config.get("optimizer_config", {}).get(
            "rewrite_agent_model", "gpt-5.1"
        )
        kwargs["judge_agent_model"] = self.config.get("optimizer_config", {}).get(
            "judge_agent_model", "gpt-4o-mini"
        )

        builder = Optimizer(self, **kwargs)
        self.optimizer = builder
        result = builder.should_optimize(step_name, op_name)
        return result

    def optimize(
        self,
        save: bool = False,
        return_pipeline: bool = True,
        **kwargs,
    ) -> tuple[dict | "DSLRunner", float]:

        if not self.last_op_container:
            raise ValueError("No operations in pipeline. Cannot optimize.")

        self.load()

        # Augment the kwargs with the runner's config if not already provided
        kwargs["litellm_kwargs"] = self.config.get("optimizer_config", {}).get(
            "litellm_kwargs", {}
        )
        kwargs["rewrite_agent_model"] = self.config.get("optimizer_config", {}).get(
            "rewrite_agent_model", "gpt-5.1"
        )
        kwargs["judge_agent_model"] = self.config.get("optimizer_config", {}).get(
            "judge_agent_model", "gpt-4o-mini"
        )

        save_path = kwargs.get("save_path", None)
        # Pop the save_path from kwargs
        kwargs.pop("save_path", None)

        builder = Optimizer(
            self,
            **kwargs,
        )
        self.optimizer = builder
        llm_api_cost = builder.optimize()
        operations_cost = self.total_cost
        self.total_cost += llm_api_cost

        # Log the cost of optimization
        self.console.log(
            f"[green italic]💰 Total cost: ${self.total_cost:.4f}[/green italic]"
        )
        self.console.log(
            f"[green italic]  ├─ Operation execution cost: ${operations_cost:.4f}[/green italic]"
        )
        self.console.log(
            f"[green italic]  └─ Optimization cost: ${llm_api_cost:.4f}[/green italic]"
        )

        if save:
            if kwargs.get("save_path"):
                save_path = kwargs["save_path"]
                builder.save_optimized_config(save_path)
                self.optimized_config_path = save_path
            else:
                save_path = f"{self.base_name}_opt.yaml"
                builder.save_optimized_config(save_path)
                self.optimized_config_path = save_path

        if return_pipeline:
            return (
                DSLRunner(builder.clean_optimized_config(), self.max_threads),
                self.total_cost,
            )

        return builder.clean_optimized_config(), self.total_cost

    def _run_operation(
        self,
        op_config: dict[str, Any],
        input_data: list[dict[str, Any]] | dict[str, Any],
        return_instance: bool = False,
        is_build: bool = False,
    ) -> list[dict[str, Any]] | tuple[list[dict[str, Any]], BaseOperation, float]:
        """
        Run a single operation based on its configuration.

        This method creates an instance of the appropriate operation class and executes it.
        It also updates the total operation cost.

        Args:
            op_config (dict[str, Any]): The configuration of the operation to run.
            input_data (list[dict[str, Any]]): The input data for the operation.
            return_instance (bool, optional): If True, return the operation instance along with the output data.

        Returns:
            list[dict[str, Any]] | tuple[list[dict[str, Any]], BaseOperation, float]:
            If return_instance is False, returns the output data.
            If return_instance is True, returns a tuple of the output data, the operation instance, and the cost.
        """
        operation_class = get_operation(op_config["type"])

        oc_kwargs = {
            "runner": self,
            "config": op_config,
            "default_model": self.config["default_model"],
            "max_threads": self.max_threads,
            "console": self.console,
            "status": self.status,
        }
        operation_instance = operation_class(**oc_kwargs)
        if op_config["type"] == "equijoin":
            output_data, cost = operation_instance.execute(
                input_data["left_data"], input_data["right_data"]
            )
        elif op_config["type"] == "filter":
            output_data, cost = operation_instance.execute(input_data, is_build)
        else:
            output_data, cost = operation_instance.execute(input_data)

        self.total_cost += cost

        if return_instance:
            return output_data, operation_instance
        else:
            return output_data

    def _flush_partial_results(
        self, operation_name: str, batch_index: int, data: list[dict]
    ) -> None:
        """
        Save partial (batch-level) results from an operation to a directory named
        '<operation_name>_batches' inside the intermediate directory.

        Args:
            operation_name (str): The name of the operation, e.g. 'extract_medications'.
            batch_index (int): Zero-based index of the batch.
            data (list[dict]): Batch results to write to disk.
        """
        if not self.intermediate_dir:
            return

        op_batches_dir = storage_path_join(self.intermediate_dir, f"{operation_name}_batches")
        self.storage.makedirs(op_batches_dir, exist_ok=True)

        checkpoint_path = storage_path_join(op_batches_dir, f"batch_{batch_index}.json")

        with self.storage.open(checkpoint_path, "w") as f:
            json.dump(data, f)

        self.console.log(
            f"[green]✓[/green] [italic]Partial checkpoint saved for '{operation_name}', "
            f"batch {batch_index} at '{checkpoint_path}'[/italic]"
        )
