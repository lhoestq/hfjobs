"""UV run command for hfjobs - execute UV scripts on HF infrastructure."""

import hashlib
from argparse import Namespace
from datetime import datetime
from pathlib import Path

from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError

from . import BaseCommand
from .run import RunCommand


class UvCommand(BaseCommand):
    """Run UV scripts on Hugging Face infrastructure."""

    @staticmethod
    def register_subcommand(parser):
        """Register UV run subcommand."""
        uv_parser = parser.add_parser(
            "uv",
            help="Run UV scripts on Hugging Face infrastructure",
            description="Execute UV scripts (Python scripts with inline dependencies) using hfjobs",
        )

        subparsers = uv_parser.add_subparsers(
            dest="uv_command", help="UV commands", required=True
        )

        # Run command only
        run_parser = subparsers.add_parser(
            "run",
            help="Run a UV script on HF infrastructure",
            description="Upload and execute a UV script using hfjobs",
        )
        run_parser.add_argument("script", help="UV script to run")
        run_parser.add_argument(
            "script_args", nargs="*", help="Arguments for the script", default=[]
        )
        run_parser.add_argument(
            "--repo",
            help="Repository name for the script (creates ephemeral if not specified)",
        )
        run_parser.add_argument(
            "--flavor", default="cpu-basic", help="Hardware flavor (default: cpu-basic)"
        )
        run_parser.add_argument(
            "-e", "--env", action="append", help="Environment variables"
        )
        run_parser.add_argument(
            "-s", "--secret", action="append", help="Secret environment variables"
        )
        run_parser.add_argument("--timeout", help="Max duration (e.g., 30s, 5m, 1h)")
        run_parser.add_argument(
            "-d", "--detach", action="store_true", help="Run in background"
        )
        run_parser.add_argument("--token", help="HF token")
        run_parser.set_defaults(func=UvCommand)

    def __init__(self, args):
        """Initialize the command with parsed arguments."""
        self.args = args

    def run(self):
        """Execute UV command."""
        if self.args.uv_command == "run":
            self._run_script(self.args)

    def _run_script(self, args):
        """Run a UV script on HF infrastructure."""
        api = HfApi()

        # Check script exists
        script_path = Path(args.script)
        if not script_path.exists():
            print(f"Error: Script not found: {args.script}")
            return

        # Determine repository
        repo_id = self._determine_repository(args, api)
        is_ephemeral = args.repo is None

        # Create repo if needed
        try:
            api.repo_info(repo_id, repo_type="dataset")
            if not is_ephemeral:
                print(f"Using existing repository: {repo_id}")
        except RepositoryNotFoundError:
            print(f"Creating repository: {repo_id}")
            create_repo(repo_id, repo_type="dataset", exist_ok=True)

        # Upload script
        print(f"Uploading {script_path.name}...")
        with open(script_path, "r") as f:
            script_content = f.read()

        filename = script_path.name

        api.upload_file(
            path_or_fileobj=script_content.encode(),
            path_in_repo=filename,
            repo_id=repo_id,
            repo_type="dataset",
        )

        script_url = (
            f"https://huggingface.co/datasets/{repo_id}/resolve/main/{filename}"
        )
        repo_url = f"https://huggingface.co/datasets/{repo_id}"

        print(f"✓ Script uploaded to: {repo_url}/blob/main/{filename}")

        # Create and upload minimal README
        readme_content = self._create_minimal_readme(repo_id, filename, is_ephemeral)
        api.upload_file(
            path_or_fileobj=readme_content.encode(),
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="dataset",
        )

        if is_ephemeral:
            print(f"✓ Temporary repository created: {repo_id}")

        # Prepare docker image (always use Python 3.12)
        docker_image = "ghcr.io/astral-sh/uv:python3.12-bookworm-slim"

        # Build command
        command = ["uv", "run", script_url] + args.script_args

        # Create RunCommand args
        run_args = Namespace(
            dockerImage=docker_image,
            command=command,
            env=args.env,
            secret=args.secret,
            env_file=None,
            secret_env_file=None,
            flavor=args.flavor,
            timeout=args.timeout,
            detach=args.detach,
            token=args.token,
        )

        print("Starting job on HF infrastructure...")
        RunCommand(run_args).run()

    def _determine_repository(self, args, api):
        """Determine which repository to use for the script."""
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
        script_hash = hashlib.md5(Path(args.script).read_bytes()).hexdigest()[:8]

        return f"{username}/hfjobs-uv-run-{timestamp}-{script_hash}"

    def _create_minimal_readme(self, repo_id, script_name, is_ephemeral):
        """Create minimal README content."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")

        if is_ephemeral:
            # Ephemeral repository README
            readme = f"""---
tags:
- hfjobs-uv-script
- ephemeral
---

# UV Script: {script_name}

Executed via `hfjobs uv run` on {timestamp}

## Run this script

```bash
hfjobs run ghcr.io/astral-sh/uv:python3.12-bookworm-slim \\
  uv run https://huggingface.co/datasets/{repo_id}/resolve/main/{script_name}
```

---
*Created with [hfjobs](https://github.com/huggingface/hfjobs)*
"""
        else:
            # Named repository README
            repo_name = repo_id.split("/")[-1]
            readme = f"""---
tags:
- hfjobs-uv-script
---

# {repo_name}

UV scripts repository

## Scripts
- `{script_name}` - Added {timestamp}

## Run

```bash
hfjobs uv run {script_name} --repo {repo_name}
```

---
*Created with [hfjobs](https://github.com/huggingface/hfjobs)*
"""

        return readme
