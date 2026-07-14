#!/usr/bin/env python3
"""Dump the tool schema a running Gradio MCP server exposes.

Gradio builds its MCP tool schema from each API function's docstring + type hints, but its
docstring parser is primitive and silently mangles anything not shaped exactly right. The only
way to know what an MCP client actually sees is to read the live schema. This script fetches it
and prints, per tool: the name, the description, and each input argument's description -- so you
can confirm your docstring survived the parser intact (URLs present, arg descriptions complete,
tool named what you expect).

Prerequisite: launch the app with ``demo.launch(mcp_server=True)`` (needs gradio>=5.28 and the
``gradio[mcp]`` extra). Then run this against the running server.

Usage:
    python dump_mcp_schema.py                       # http://127.0.0.1:7860
    python dump_mcp_schema.py --port 7860
    python dump_mcp_schema.py --url http://127.0.0.1:7860
    python dump_mcp_schema.py --tool detect_objects  # only show one tool

Stdlib only -- no third-party deps, runs anywhere.
"""
import argparse
import json
import sys
import urllib.error
import urllib.request


def fetch_schema(base_url: str, timeout: float = 15.0):
    schema_url = base_url.rstrip("/") + "/gradio_api/mcp/schema"
    try:
        with urllib.request.urlopen(schema_url, timeout=timeout) as resp:
            return json.loads(resp.read())
    except urllib.error.URLError as e:
        sys.exit(
            f"ERROR: could not reach {schema_url}\n"
            f"  ({e})\n"
            "  Is the app running and launched with demo.launch(mcp_server=True)?\n"
            "  (mcp_server needs gradio>=5.28 and the gradio[mcp] extra installed.)"
        )
    except json.JSONDecodeError as e:
        sys.exit(f"ERROR: {schema_url} did not return valid JSON: {e}")


def print_tool(tool: dict) -> None:
    print("=" * 78)
    print(f"TOOL: {tool.get('name', '<unnamed>')}")
    print("-" * 78)
    print("DESCRIPTION:")
    print(f"  {tool.get('description', '(none)')}")
    props = (tool.get("inputSchema") or {}).get("properties") or {}
    print("ARGUMENTS:")
    if not props:
        print("  (none)")
    for name, spec in props.items():
        desc = spec.get("description", "(no description)")
        typ = spec.get("type", "?")
        print(f"  - {name} ({typ}): {desc}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--url", help="Base URL of the running Gradio app.")
    parser.add_argument("--port", type=int, default=7860, help="Port (default 7860) if --url omitted.")
    parser.add_argument("--host", default="127.0.0.1", help="Host (default 127.0.0.1) if --url omitted.")
    parser.add_argument("--tool", help="Only print the tool whose name contains this string.")
    parser.add_argument("--raw", action="store_true", help="Print the raw JSON schema instead of a summary.")
    args = parser.parse_args()

    base_url = args.url or f"http://{args.host}:{args.port}"
    schema = fetch_schema(base_url)

    if args.raw:
        print(json.dumps(schema, indent=2))
        return

    tools = schema if isinstance(schema, list) else schema.get("tools", [])
    if args.tool:
        tools = [t for t in tools if args.tool in t.get("name", "")]
    if not tools:
        sys.exit(f"No tools found (filter={args.tool!r}). The app may expose no MCP tools.")

    for tool in tools:
        print_tool(tool)


if __name__ == "__main__":
    main()
