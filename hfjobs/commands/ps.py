import json
from argparse import _SubParsersAction, Namespace
from typing import Optional

import requests
from huggingface_hub import whoami
from huggingface_hub.utils import build_hf_headers

from . import BaseCommand
from ._cli_utils import tabulate


class PsCommand(BaseCommand):

    @staticmethod
    def register_subcommand(parser: _SubParsersAction) -> None:
        run_parser = parser.add_parser("ps", help="List Jobs")
        run_parser.add_argument(
            "-a", "--all", action="store_true", help="Show all Jobs (default shows just running)"
        )
        run_parser.add_argument(
            "--token", type=str, help="A User Access Token generated from https://huggingface.co/settings/tokens"
        )
        run_parser.set_defaults(func=PsCommand)

    def __init__(self, args: Namespace) -> None:
        self.all: bool = args.all
        self.token: Optional[str] = args.token or None

    def run(self) -> None:
        username = whoami(self.token)["name"]
        headers = build_hf_headers(token=self.token, library_name="hfjobs")
        resp = requests.get(
            f"https://huggingface.co/api/jobs/{username}",
            headers=headers,
        )
        resp.raise_for_status()
        headers = ["JOB ID", "IMAGE", "COMMAND", "CREATED", "STATUS"]
        rows = [
            [
                job["metadata"]["id"],
                job["compute"]["spec"]["extra"]["input"].get(
                    "dockerImage",
                    "hf.co/spaces/" + job["compute"]["spec"]["extra"]["input"]["spaceId"]
                ),
                json.dumps(" ".join(job["compute"]["spec"]["extra"]["command"])),
                job["metadata"]["created_at"],
                job["compute"]["status"]["stage"]
            ]
            for job in resp.json()
            if self.all or job["compute"]["status"]["stage"] in ("RUNNING", "UPDATING")
        ]
        print(tabulate(rows, headers))
