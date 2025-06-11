import json
import time
from argparse import _SubParsersAction, Namespace
from typing import Optional

import requests
from huggingface_hub import whoami
from huggingface_hub.utils import build_hf_headers

from . import BaseCommand


class CancelCommand(BaseCommand):

    @staticmethod
    def register_subcommand(parser: _SubParsersAction) -> None:
        run_parser = parser.add_parser("cancel", help="Cancel a Job")
        run_parser.add_argument(
            "job_id", type=str, help="Job ID"
        )
        run_parser.add_argument(
            "--token", type=str, help="A User Access Token generated from https://huggingface.co/settings/tokens"
        )
        run_parser.set_defaults(func=CancelCommand)

    def __init__(self, args: Namespace) -> None:
        self.job_id: str = args.job_id
        self.token: Optional[str] = args.token or None

    def run(self) -> None:
        username = whoami(self.token)["name"]
        headers = build_hf_headers(token=self.token, library_name="hfjobs")
        requests.post(
            f"https://huggingface.co/api/jobs/{username}/{self.job_id}/cancel",
            headers=headers,
        ).raise_for_status()
