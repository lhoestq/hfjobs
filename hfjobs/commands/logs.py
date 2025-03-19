import json
import time
from argparse import _SubParsersAction, Namespace
from typing import Optional

import requests
from huggingface_hub import whoami
from huggingface_hub.utils import build_hf_headers

from . import BaseCommand


class LogsCommand(BaseCommand):

    @staticmethod
    def register_subcommand(parser: _SubParsersAction) -> None:
        run_parser = parser.add_parser("logs", help="Fetch the logs of a Job")
        run_parser.add_argument(
            "job_id", type=str, help="Job ID"
        )
        run_parser.add_argument(
            "-t", "--timestamps", action="store_true", help="Show timestamps"
        )
        run_parser.add_argument(
            "--token", type=str, help="A User Access Token generated from https://huggingface.co/settings/tokens"
        )
        run_parser.set_defaults(func=LogsCommand)

    def __init__(self, args: Namespace) -> None:
        self.job_id: str = args.job_id
        self.timestamps: bool = args.timestamps
        self.token: Optional[str] = args.token or None

    def run(self) -> None:
        username = whoami(self.token)["name"]
        headers = build_hf_headers(token=self.token, library_name="hfjobs")
        requests.get(
            f"https://huggingface.co/api/jobs/{username}/{self.job_id}",
            headers=headers,
        ).raise_for_status()

        logging_finished = False
        job_finished = False
        # We need to retry because sometimes the /logs-stream doesn't return logs when the job just started.
        # For example it can return only two lines: one for "Job started" and one empty line.
        while True:
            resp = requests.get(
                f"https://huggingface.co/api/jobs/{username}/{self.job_id}/logs-stream",
                headers=headers,
                stream=True,
                timeout=10,
            )
            log = None
            for line in resp.iter_lines():
                line = line.decode("utf-8")
                if line and line.startswith("data: {"):
                    data = json.loads(line[len("data: "):])
                    # timestamp = data["timestamp"]
                    if not data["data"].startswith("===== Job started"):
                        log = data["data"]
                        print(log)
            logging_finished |= log is not None
            if logging_finished or job_finished:
                break
            job_status = requests.get(
                f"https://huggingface.co/api/jobs/{username}/{self.job_id}",
                headers=headers,
            ).json()
            if "status" in job_status and job_status["status"]["stage"] not in ("RUNNING", "UPDATING"):
                job_finished = True
            time.sleep(1)
