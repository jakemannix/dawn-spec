#!/usr/bin/env python
"""
Generate Python code from Protocol Buffer definitions.

This script compiles the .proto files for the ACP and AGP protocols
into Python modules for use in the DAWN implementation.
"""
import os
import sys
import subprocess
from pathlib import Path

# Directory paths
PROTO_DIR = Path("proto")
OUTPUT_DIR = Path("src/proto")

# Ensure output directory exists
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def generate_proto(proto_file: Path) -> None:
    """
    Generate Python code from a .proto file.
    
    Args:
        proto_file: Path to the .proto file
    """
    print(f"Generating Python code for {proto_file}...")
    
    # Construct the command
    command = [
        "python", "-m", "grpc_tools.protoc",
        f"--proto_path={PROTO_DIR}",
        f"--python_out={OUTPUT_DIR}",
        f"--grpc_python_out={OUTPUT_DIR}",
        str(proto_file)
    ]
    
    # Run the command
    try:
        subprocess.run(command, check=True)
        print(f"Successfully generated code for {proto_file}")
    except subprocess.CalledProcessError as e:
        print(f"Error generating code for {proto_file}: {e}")
        sys.exit(1)


def fix_imports() -> None:
    """Fix relative imports in generated files."""
    print("Fixing imports in generated files...")
    
    for py_file in OUTPUT_DIR.glob("**/*.py"):
        with open(py_file, "r") as f:
            content = f.read()
        
        # Fix import statements
        if "import" in content:
            content = content.replace("from proto", "from src.proto")
        
        with open(py_file, "w") as f:
            f.write(content)


def create_init_files() -> None:
    """Create __init__.py files for Python packages."""
    print("Creating __init__.py files...")
    
    # Create __init__.py in the output directory
    init_file = OUTPUT_DIR / "__init__.py"
    if not init_file.exists():
        with open(init_file, "w") as f:
            f.write('"""Generated Protocol Buffer code for DAWN protocols."""\n')


def main() -> None:
    """Run the code generation process."""
    print("=== Generating DAWN Protocol Code ===")
    
    # Find all .proto files
    proto_files = list(PROTO_DIR.glob("*.proto"))
    
    if not proto_files:
        print(f"No .proto files found in {PROTO_DIR}")
        sys.exit(1)
    
    print(f"Found {len(proto_files)} .proto files: {', '.join(f.name for f in proto_files)}")
    
    # Generate code for each .proto file
    for proto_file in proto_files:
        generate_proto(proto_file)
    
    # Fix imports and create package structure
    fix_imports()
    create_init_files()
    
    print("=== Protocol Code Generation Complete ===")


if __name__ == "__main__":
    main()