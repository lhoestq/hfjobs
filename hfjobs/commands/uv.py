"""UV commands for hfjobs."""

import hashlib
import os
import re
import sys
from argparse import Namespace
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any

from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError

from . import BaseCommand
from .run import RunCommand


class UvCommand(BaseCommand):
    """Manage UV scripts on Hugging Face Hub."""

    @staticmethod
    def register_subcommand(parser):
        """Register UV subcommands."""
        uv_parser = parser.add_parser(
            "uv",
            help="Share and manage UV scripts on Hugging Face Hub",
            description="Commands for sharing UV scripts as Hugging Face datasets"
        )
        
        subparsers = uv_parser.add_subparsers(
            dest="uv_command",
            help="UV commands",
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
        init_parser.set_defaults(func=UvCommand)
        
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
        push_parser.set_defaults(func=UvCommand)
        
        # Sync command
        sync_parser = subparsers.add_parser(
            "sync",
            help="Sync local scripts to repository",
            description="Sync all Python scripts from local directory to HF repository"
        )
        sync_parser.add_argument(
            "files",
            nargs="*",
            help="Specific files to sync (default: all .py files)"
        )
        sync_parser.add_argument(
            "--dry-run",
            action="store_true",
            help="Show what would be synced without uploading"
        )
        sync_parser.set_defaults(func=UvCommand)
        
        # Run command
        run_parser = subparsers.add_parser(
            "run",
            help="Run a UV script on HF infrastructure",
            description="Upload and execute a UV script using hfjobs"
        )
        run_parser.add_argument("script", help="UV script to run")
        run_parser.add_argument("script_args", nargs="*", help="Arguments for the script", default=[])
        run_parser.add_argument("--repo", help="Repository for the script")
        run_parser.add_argument("--python", default="3.12", help="Python version")
        run_parser.add_argument("--flavor", default="cpu-basic", help="Hardware flavor")
        run_parser.add_argument("-e", "--env", action="append", help="Environment variables")
        run_parser.add_argument("-s", "--secret", action="append", help="Secret environment variables")
        run_parser.add_argument("--timeout", help="Max duration")
        run_parser.add_argument("-d", "--detach", action="store_true", help="Run in background")
        run_parser.add_argument("--token", help="HF token")
        run_parser.set_defaults(func=UvCommand)

    def __init__(self, args):
        """Initialize the command with parsed arguments."""
        self.args = args

    def run(self):
        """Execute UV command."""
        if self.args.uv_command == "init":
            self._init_repo(self.args)
        elif self.args.uv_command == "push":
            self._push_script(self.args)
        elif self.args.uv_command == "sync":
            self._sync_scripts(self.args)
        elif self.args.uv_command == "run":
            self._run_script(self.args)

    def _init_repo(self, args):
        """Initialize a new UV script repository."""
        api = HfApi()
        
        # Ensure repo name includes username
        repo_id = args.repo
        if "/" not in repo_id:
            user_info = api.whoami()
            username = user_info["name"]
            repo_id = f"{username}/{repo_id}"
        
        # Create local directory
        local_dir = Path(repo_id.split('/')[-1])
        if local_dir.exists():
            print(f"Error: Directory '{local_dir}' already exists")
            return
        
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
        
        # Create local directory structure
        print(f"Creating local directory: {local_dir}")
        local_dir.mkdir(parents=True)
        config_dir = local_dir / ".hfjobs"
        config_dir.mkdir()
        
        # Save config
        config_file = config_dir / "config"
        config_file.write_text(f"repo={repo_id}\n")
        
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
            
            # Save to local directory
            local_script = local_dir / script_name
            local_script.write_text(script_content)
            
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
            
            # Save to local directory
            local_script = local_dir / script_name
            local_script.write_text(script_content)
            
            api.upload_file(
                path_or_fileobj=script_content.encode(),
                path_in_repo=script_name,
                repo_id=repo_id,
                repo_type="dataset"
            )
        
        # Create README
        print("Creating README with usage instructions")
        readme_content = self._create_readme(repo_id, script_name, script_content)
        
        # Save README locally
        local_readme = local_dir / "README.md"
        local_readme.write_text(readme_content)
        
        api.upload_file(
            path_or_fileobj=readme_content.encode(),
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="dataset"
        )
        
        # Save last repo for push command
        self._save_last_repo(repo_id)
        
        print(f"\n✅ Created local directory: {local_dir}")
        print(f"✅ Script published to: https://huggingface.co/datasets/{repo_id}")
        print(f"\nRun your script with:")
        print(f"hfjobs run ghcr.io/astral-sh/uv:python3.12 \\")
        print(f"  uv run https://huggingface.co/datasets/{repo_id}/resolve/main/{script_name} \\")
        print(f"  <your-args>")
        print(f"\nLocal directory: cd {local_dir}")

    def _push_script(self, args):
        """Push a script to an existing repository."""
        api = HfApi()
        
        # Check if we're in a local directory with config
        local_dir = None
        config_file = Path(".hfjobs/config")
        if config_file.exists():
            # We're in a scripts directory
            local_dir = Path.cwd()
            # Read repo from config
            config_content = config_file.read_text()
            for line in config_content.splitlines():
                if line.startswith("repo="):
                    repo_id = line.split("=", 1)[1]
                    break
        else:
            # Get repository from args or last repo
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
        
        # Save locally if in a local directory
        if local_dir and script_path.parent != local_dir:
            local_script = local_dir / script_name
            local_script.write_text(script_content)
            print(f"Saved locally: {local_script}")
        
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

    def _sync_scripts(self, args):
        """Sync local scripts to remote repository."""
        api = HfApi()
        
        # Check if we're in a scripts directory with config
        config_file = Path(".hfjobs/config")
        if not config_file.exists():
            # Try parent directory
            config_file = Path("../.hfjobs/config")
            if not config_file.exists():
                print("Error: Not in a UV directory. Run from within a directory created by 'hfjobs uv init'")
                return
        
        # Read config
        config_content = config_file.read_text()
        repo_id = None
        for line in config_content.splitlines():
            if line.startswith("repo="):
                repo_id = line.split("=", 1)[1]
                break
        
        if not repo_id:
            print("Error: Could not find repository in config")
            return
        
        # Get local directory
        local_dir = config_file.parent.parent
        
        # Find files to sync
        if args.files:
            # Specific files provided
            files_to_sync = []
            for file_pattern in args.files:
                files_to_sync.extend(local_dir.glob(file_pattern))
        else:
            # Default to all Python files
            files_to_sync = list(local_dir.glob("*.py"))
        
        if not files_to_sync:
            print("No files to sync")
            return
        
        print(f"Repository: {repo_id}")
        print(f"Files to sync:")
        for file in files_to_sync:
            print(f"  - {file.name}")
        
        if args.dry_run:
            print("\n--dry-run specified, not uploading")
            return
        
        # Upload each file
        for file_path in files_to_sync:
            if file_path.is_file() and not file_path.name.startswith('.'):
                print(f"\nUploading: {file_path.name}")
                with open(file_path, 'r') as f:
                    content = f.read()
                
                api.upload_file(
                    path_or_fileobj=content.encode(),
                    path_in_repo=file_path.name,
                    repo_id=repo_id,
                    repo_type="dataset"
                )
        
        # Update README if there are new scripts
        print("\nUpdating README...")
        try:
            # Download current README
            readme_path = api.hf_hub_download(
                repo_id=repo_id,
                filename="README.md",
                repo_type="dataset"
            )
            with open(readme_path, 'r') as f:
                readme_content = f.read()
            
            # Check if any scripts are missing from README
            updated = False
            for file_path in files_to_sync:
                if file_path.suffix == '.py' and file_path.name not in readme_content:
                    with open(file_path, 'r') as f:
                        script_content = f.read()
                    readme_content = self._update_readme_for_new_script(
                        readme_content, repo_id, file_path.name, script_content
                    )
                    updated = True
            
            if updated:
                # Save updated README locally
                local_readme = local_dir / "README.md"
                local_readme.write_text(readme_content)
                
                # Upload updated README
                api.upload_file(
                    path_or_fileobj=readme_content.encode(),
                    path_in_repo="README.md",
                    repo_id=repo_id,
                    repo_type="dataset"
                )
                print("✅ README updated")
        except Exception as e:
            print(f"Warning: Could not update README: {e}")
        
        print(f"\n✅ Sync complete: https://huggingface.co/datasets/{repo_id}")

    def _run_script(self, args):
        """Run a UV script on HF infrastructure."""
        api = HfApi()
        
        # Check script exists
        script_path = Path(args.script)
        if not script_path.exists():
            print(f"Error: Script not found: {args.script}")
            return
        
        # Determine repository
        repo_id = self._determine_repository(args)
        is_ephemeral = args.repo is None and not Path(".hfjobs/config").exists()
        
        # Create repo if needed
        try:
            api.repo_info(repo_id, repo_type="dataset")
            print(f"Using existing repository: {repo_id}")
        except RepositoryNotFoundError:
            print(f"Creating repository: {repo_id}")
            create_repo(repo_id, repo_type="dataset", exist_ok=True)
        
        # Upload script
        print(f"Uploading {script_path.name}...")
        with open(script_path, 'r') as f:
            script_content = f.read()
        
        # For MVP, just use original filename
        filename = script_path.name
        
        api.upload_file(
            path_or_fileobj=script_content.encode(),
            path_in_repo=filename,
            repo_id=repo_id,
            repo_type="dataset"
        )
        
        script_url = f"https://huggingface.co/datasets/{repo_id}/resolve/main/{filename}"
        repo_url = f"https://huggingface.co/datasets/{repo_id}"
        print(f"✓ Script uploaded to: {script_url}")
        print(f"✓ Repository: {repo_url}")
        
        # Create and upload README
        if is_ephemeral:
            print(f"✓ Temporary repository created: {repo_id}")
            # Create minimal README for ephemeral repo
            readme_content = self._create_minimal_readme(repo_id, filename, script_content)
        else:
            # For persistent repos, check if README exists and update it
            try:
                # Try to download existing README
                readme_path = api.hf_hub_download(
                    repo_id=repo_id,
                    filename="README.md",
                    repo_type="dataset"
                )
                with open(readme_path, 'r') as f:
                    existing_readme = f.read()
                # Update existing README with new script
                readme_content = self._update_readme_with_script(repo_id, filename, script_content, existing_readme)
            except Exception:
                # No existing README, create new one
                readme_content = self._create_readme(repo_id, filename, script_content)
        
        # Upload README
        api.upload_file(
            path_or_fileobj=readme_content.encode(),
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="dataset"
        )
        
        # Prepare docker image
        docker_image = f"ghcr.io/astral-sh/uv:python{args.python}-bookworm-slim"
        
        # Build command
        command = ["uv", "run", script_url] + args.script_args
        
        # Create RunCommand args
        run_args = Namespace(
            dockerImage=docker_image,
            command=command,
            env=args.env,
            secret=args.secret,
            env_file=None,  # Not supported in MVP
            secret_env_file=None,  # Not supported in MVP
            flavor=args.flavor,
            timeout=args.timeout,
            detach=args.detach,
            token=args.token
        )
        
        print("Starting job on HF infrastructure...")
        RunCommand(run_args).run()

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
        description = self._extract_description(script_content) or "UV script"
        
        readme = f"""---
tags:
- hfjobs-uv-script
- uv
- python
viewer: false
---

# {repo_id.split('/')[-1]}

A collection of UV scripts for hfjobs.

## Usage

Run any script using:
```bash
hfjobs uv run <script_name> --repo {repo_id.split('/')[-1]}
```

## Scripts

<!-- AUTO-GENERATED SCRIPTS LIST - DO NOT EDIT MANUALLY -->
| Script | Description | Command |
|--------|-------------|---------|
| [{script_name}](./blob/main/{script_name}) | {description} | `hfjobs uv run {script_name} --repo {repo_id.split('/')[-1]}` |
<!-- END AUTO-GENERATED SCRIPTS LIST -->

## Learn More

Learn more about UV scripts in the [UV documentation](https://docs.astral.sh/uv/guides/scripts/).

---
*Created with [hfjobs](https://github.com/huggingface/hfjobs)*
"""
        
        return readme

    def _create_minimal_readme(self, repo_id: str, script_name: str, script_content: str) -> str:
        """Create minimal README content for ephemeral repositories."""
        # Extract script info
        description = self._extract_description(script_content)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")
        
        readme = f"""---
tags:
- hfjobs-uv-script
- ephemeral
- uv
- python
viewer: false
---

# Ephemeral UV Script Repository

This is a temporary repository created by `hfjobs uv run` for one-time script execution.

**Script:** `{script_name}`  
**Created:** {timestamp}
"""
        
        if description:
            readme += f"**Description:** {description}\n"
        
        readme += f"""
## Direct Execution

This script was executed using:
```bash
hfjobs uv run {script_name}
```

## Script URL

```
https://huggingface.co/datasets/{repo_id}/resolve/main/{script_name}
```

---
*Created with [hfjobs](https://github.com/huggingface/hfjobs)*
"""
        
        return readme

    def _update_readme_with_script(self, repo_id: str, script_name: str, script_content: str, existing_readme: str) -> str:
        """Update existing README with a new script entry."""
        description = self._extract_description(script_content) or "UV script"
        
        # Check if script is already in the table
        if f"| [{script_name}]" in existing_readme:
            # Script already exists, don't add duplicate
            return existing_readme
        
        # Find the auto-generated section
        start_marker = "<!-- AUTO-GENERATED SCRIPTS LIST - DO NOT EDIT MANUALLY -->"
        end_marker = "<!-- END AUTO-GENERATED SCRIPTS LIST -->"
        
        start_idx = existing_readme.find(start_marker)
        end_idx = existing_readme.find(end_marker)
        
        if start_idx == -1 or end_idx == -1:
            # Markers not found, fallback to creating new README
            return self._create_readme(repo_id, script_name, script_content)
        
        # Extract the table content
        table_start = existing_readme.find("\n", start_idx) + 1
        table_end = existing_readme.rfind("\n", 0, end_idx)
        
        # Add new row to the table
        new_row = f"| [{script_name}](./blob/main/{script_name}) | {description} | `hfjobs uv run {script_name} --repo {repo_id.split('/')[-1]}` |"
        
        # Reconstruct README with new script
        updated_readme = (
            existing_readme[:table_end] + 
            "\n" + new_row + 
            existing_readme[table_end:]
        )
        
        return updated_readme

    def _update_readme_for_new_script(
        self, readme: str, repo_id: str, script_name: str, script_content: str
    ) -> str:
        """Update README when adding a new script."""
        # Extract script info
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

    def _determine_repository(self, args) -> str:
        """Determine which repository to use for the script."""
        api = HfApi()
        
        # Check local directory first
        config_file = Path(".hfjobs/config")
        if config_file.exists():
            config_content = config_file.read_text()
            for line in config_content.splitlines():
                if line.startswith("repo="):
                    repo_id = line.split("=", 1)[1]
                    print(f"Using repository from local config: {repo_id}")
                    return repo_id
        
        # Use provided repo
        if args.repo:
            repo_id = args.repo
            if "/" not in repo_id:
                username = api.whoami()["name"]
                repo_id = f"{username}/{repo_id}"
            return repo_id
        
        # Create ephemeral repo
        username = api.whoami()["name"]
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        
        # Simple hash for uniqueness
        script_hash = hashlib.md5(
            Path(args.script).read_bytes()
        ).hexdigest()[:8]
        
        return f"{username}/hfjobs-uv-run-{timestamp}-{script_hash}"