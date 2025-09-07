# Create the src directory structure
import os

project_dir = "document_qna"
src_dir = os.path.join(project_dir, "src")

# Define the directory structure
directories = [
    "src",
    "src/ingestion",
    "src/chunking", 
    "src/embeddings",
    "src/vectorstore",
    "src/retrieval",
    "src/llm",
    "src/utils",
    "tests",
    "demo",
    "demo/sample_docs"
]

# Create directories
for directory in directories:
    full_path = os.path.join(project_dir, directory)
    if not os.path.exists(full_path):
        os.makedirs(full_path)
        print(f"Created directory: {directory}")

# Create __init__.py files for Python packages
init_files = [
    "src/__init__.py",
    "src/ingestion/__init__.py",
    "src/chunking/__init__.py",
    "src/embeddings/__init__.py",
    "src/vectorstore/__init__.py", 
    "src/retrieval/__init__.py",
    "src/llm/__init__.py",
    "src/utils/__init__.py"
]

for init_file in init_files:
    full_path = os.path.join(project_dir, init_file)
    with open(full_path, "w") as f:
        f.write("# Package initialization\n")
    print(f"Created: {init_file}")

print("\n✅ Created complete directory structure")