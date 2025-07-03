"""UV script sharing commands for hfjobs."""

import os
import re
import sys
from pathlib import Path
from typing import Optional, List, Dict, Any

from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError

from . import BaseCommand


class ScriptsCommand(BaseCommand):
    """Manage UV scripts on Hugging Face Hub."""

    @staticmethod
    def register_subcommand(parser):
        """Register UV script subcommands."""
        scripts_parser = parser.add_parser(
            "scripts",
            help="Share and manage UV scripts on Hugging Face Hub",
            description="Commands for sharing UV scripts as Hugging Face datasets"
        )
        
        subparsers = scripts_parser.add_subparsers(
            dest="scripts_command",
            help="Scripts commands",
            required=True
        )
        
        # Init command
        init_parser = subparsers.add_parser(
            "init",
            help="Initialize a new UV script repository",
            description="Create a Hugging Face dataset repository for sharing UV scripts"
        )
        init_parser.add_argument(
            "repo",
            help="Repository name (e.g., 'username/my-script' or just 'my-script')"
        )
        init_parser.add_argument(
            "script",
            nargs="?",
            help="UV script to upload (creates template if not provided)"
        )
        init_parser.add_argument(
            "--private",
            action="store_true",
            help="Make the repository private"
        )
        
        # Push command
        push_parser = subparsers.add_parser(
            "push",
            help="Push a UV script to an existing repository",
            description="Update or add UV scripts to a repository"
        )
        push_parser.add_argument(
            "script",
            help="UV script to push"
        )
        push_parser.add_argument(
            "--repo",
            help="Repository to push to (uses last initialized repo if not specified)"
        )

    def run(self, args):
        """Execute scripts command."""
        if args.scripts_command == "init":
            self._init_repo(args)
        elif args.scripts_command == "push":
            self._push_script(args)

    def _init_repo(self, args):
        """Initialize a new UV script repository."""
        api = HfApi()
        
        # Ensure repo name includes username
        repo_id = args.repo
        if "/" not in repo_id:
            user_info = api.whoami()
            username = user_info["name"]
            repo_id = f"{username}/{repo_id}"
        
        # Create repository
        print(f"Creating repository: {repo_id}")
        try:
            create_repo(
                repo_id,
                repo_type="dataset",
                private=args.private,
                exist_ok=False
            )
        except Exception as e:
            if "already exists" in str(e):
                print(f"Error: Repository {repo_id} already exists")
                return
            raise
        
        # Upload script or create template
        if args.script:
            script_path = Path(args.script)
            if not script_path.exists():
                print(f"Error: Script not found: {args.script}")
                return
            
            script_name = script_path.name
            print(f"Uploading script: {script_name}")
            
            # Read script content
            with open(script_path, 'r') as f:
                script_content = f.read()
            
            # Upload script
            api.upload_file(
                path_or_fileobj=script_content.encode(),
                path_in_repo=script_name,
                repo_id=repo_id,
                repo_type="dataset"
            )
        else:
            # Create a template script
            script_name = "script.py"
            script_content = self._create_template_script()
            
            print(f"Creating template script: {script_name}")
            api.upload_file(
                path_or_fileobj=script_content.encode(),
                path_in_repo=script_name,
                repo_id=repo_id,
                repo_type="dataset"
            )
        
        # Create README
        print("Creating README with usage instructions")
        readme_content = self._create_readme(repo_id, script_name, script_content)
        
        api.upload_file(
            path_or_fileobj=readme_content.encode(),
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="dataset"
        )
        
        # Save last repo for push command
        self._save_last_repo(repo_id)
        
        print(f"\n✅ Script published to: https://huggingface.co/datasets/{repo_id}")
        print(f"\nRun your script with:")
        print(f"hfjobs run ghcr.io/astral-sh/uv:python3.12 \\")
        print(f"  uv run https://huggingface.co/datasets/{repo_id}/resolve/main/{script_name} \\")
        print(f"  <your-args>")

    def _push_script(self, args):
        """Push a script to an existing repository."""
        api = HfApi()
        
        # Get repository
        repo_id = args.repo or self._get_last_repo()
        if not repo_id:
            print("Error: No repository specified and no previous repository found")
            print("Use --repo to specify a repository or run 'hfjobs scripts init' first")
            return
        
        # Check script exists
        script_path = Path(args.script)
        if not script_path.exists():
            print(f"Error: Script not found: {args.script}")
            return
        
        # Check repository exists
        try:
            api.repo_info(repo_id, repo_type="dataset")
        except RepositoryNotFoundError:
            print(f"Error: Repository not found: {repo_id}")
            return
        
        script_name = script_path.name
        
        # Read script content
        with open(script_path, 'r') as f:
            script_content = f.read()
        
        print(f"Uploading: {script_name}")
        
        # Upload script
        api.upload_file(
            path_or_fileobj=script_content.encode(),
            path_in_repo=script_name,
            repo_id=repo_id,
            repo_type="dataset"
        )
        
        # Update README
        print("Updating README...")
        try:
            # Download existing README
            readme_path = api.hf_hub_download(
                repo_id=repo_id,
                filename="README.md",
                repo_type="dataset"
            )
            with open(readme_path, 'r') as f:
                readme_content = f.read()
            
            # Update README if this is a new script
            if script_name not in readme_content:
                readme_content = self._update_readme_for_new_script(
                    readme_content, repo_id, script_name, script_content
                )
                
                api.upload_file(
                    path_or_fileobj=readme_content.encode(),
                    path_in_repo="README.md",
                    repo_id=repo_id,
                    repo_type="dataset"
                )
        except Exception as e:
            print(f"Warning: Could not update README: {e}")
        
        print(f"✅ Script added to repository")
        print(f"View at: https://huggingface.co/datasets/{repo_id}")

    def _create_template_script(self) -> str:
        """Create a template UV script."""
        return '''# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "datasets",
#     "tqdm",
# ]
# ///
"""Template UV script for hfjobs.

This is a template script. Customize it for your needs!

Usage:
    python script.py <input> <output> [--option value]
"""

import argparse
from datasets import load_dataset
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser(description="Template UV script")
    parser.add_argument("input", help="Input dataset")
    parser.add_argument("output", help="Output dataset")
    parser.add_argument("--text-column", default="text", help="Text column name")
    
    args = parser.parse_args()
    
    print(f"Loading dataset: {args.input}")
    dataset = load_dataset(args.input, split="train")
    
    # Your processing logic here
    print(f"Processing {len(dataset)} examples...")
    
    # Example: simple transformation
    def process_example(example):
        # Add your transformation logic
        return example
    
    processed = dataset.map(process_example, desc="Processing")
    
    print(f"Pushing to: {args.output}")
    processed.push_to_hub(args.output)
    
    print("✅ Done!")


if __name__ == "__main__":
    main()
'''

    def _create_readme(self, repo_id: str, script_name: str, script_content: str) -> str:
        """Create README content for the repository."""
        # Extract script info
        deps = self._extract_dependencies(script_content)
        description = self._extract_description(script_content)
        
        readme = f"""---
tags:
- hfjobs-uv-script
- uv
- python
---

# {repo_id.split('/')[-1]}

A UV script for hfjobs.

## Usage

```bash
hfjobs run ghcr.io/astral-sh/uv:python3.12 \\
  uv run https://huggingface.co/datasets/{repo_id}/resolve/main/{script_name} \\
  <your-args>
```

## Script Details

**Script:** `{script_name}`
"""
        
        if description:
            readme += f"\n**Description:** {description}\n"
        
        if deps:
            readme += "\n**Dependencies:**\n"
            for dep in deps:
                readme += f"- {dep}\n"
        
        readme += """
---
*Created with [hfjobs](https://github.com/huggingface/hfjobs)*
"""
        
        return readme

    def _update_readme_for_new_script(
        self, readme: str, repo_id: str, script_name: str, script_content: str
    ) -> str:
        """Update README when adding a new script."""
        # Extract script info
        deps = self._extract_dependencies(script_content)
        description = self._extract_description(script_content)
        
        # Find where to insert new script info
        if "## Scripts" not in readme:
            # Add Scripts section before the footer
            footer_marker = "---\n*Created with"
            if footer_marker in readme:
                before_footer = readme.split(footer_marker)[0]
                footer = footer_marker + readme.split(footer_marker)[1]
            else:
                before_footer = readme
                footer = "\n---\n*Created with [hfjobs](https://github.com/huggingface/hfjobs)*\n"
            
            scripts_section = "\n## Scripts\n\n"
        else:
            # Insert into existing Scripts section
            parts = readme.split("## Scripts")
            before_scripts = parts[0] + "## Scripts"
            
            # Find next section or footer
            remaining = parts[1]
            next_section_match = re.search(r'\n## ', remaining)
            footer_match = re.search(r'\n---\n\*Created with', remaining)
            
            if next_section_match:
                scripts_content = remaining[:next_section_match.start()]
                after_scripts = remaining[next_section_match.start():]
            elif footer_match:
                scripts_content = remaining[:footer_match.start()]
                after_scripts = remaining[footer_match.start():]
            else:
                scripts_content = remaining
                after_scripts = ""
            
            before_footer = before_scripts + scripts_content
            footer = after_scripts
            scripts_section = ""
        
        # Add new script info
        script_info = f"""
### {script_name}
"""
        if description:
            script_info += f"{description}\n\n"
        
        script_info += f"""```bash
hfjobs run ghcr.io/astral-sh/uv:python3.12 \\
  uv run https://huggingface.co/datasets/{repo_id}/resolve/main/{script_name}
```
"""
        
        if deps:
            script_info += "\n**Dependencies:** " + ", ".join(deps) + "\n"
        
        return before_footer + scripts_section + script_info + footer

    def _extract_dependencies(self, script_content: str) -> List[str]:
        """Extract dependencies from UV script header."""
        deps = []
        in_deps = False
        
        for line in script_content.split('\n'):
            if 'dependencies = [' in line:
                in_deps = True
                continue
            if in_deps:
                if ']' in line:
                    break
                dep_match = re.search(r'"([^"]+)"', line)
                if dep_match:
                    deps.append(dep_match.group(1))
        
        return deps

    def _extract_description(self, script_content: str) -> Optional[str]:
        """Extract description from script docstring."""
        # Look for docstring
        docstring_match = re.search(r'"""(.*?)"""', script_content, re.DOTALL)
        if docstring_match:
            lines = docstring_match.group(1).strip().split('\n')
            if lines:
                # Return first non-empty line
                for line in lines:
                    line = line.strip()
                    if line:
                        return line
        return None

    def _save_last_repo(self, repo_id: str):
        """Save the last repository for future push commands."""
        config_dir = Path.home() / ".hfjobs"
        config_dir.mkdir(exist_ok=True)
        
        config_file = config_dir / "last_uv_repo"
        config_file.write_text(repo_id)

    def _get_last_repo(self) -> Optional[str]:
        """Get the last repository used."""
        config_file = Path.home() / ".hfjobs" / "last_uv_repo"
        if config_file.exists():
            return config_file.read_text().strip()
        return None