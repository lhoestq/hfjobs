import time
from argparse import _SubParsersAction, Namespace
from typing import Optional

import json
import requests
from huggingface_hub import whoami
from huggingface_hub.utils import build_hf_headers

from . import BaseCommand


class RunCommand(BaseCommand):

    @staticmethod
    def register_subcommand(parser: _SubParsersAction) -> None:
        run_parser = parser.add_parser("run", help="Run a Job")
        run_parser.add_argument(
            "dockerImage", type=str, help="The Docker image to use."
        )
        run_parser.add_argument(
            "-e", "--env", action="append", help="Read in a file of environment variables."
        )
        run_parser.add_argument(
            "--flavor", type=str, help="Flavor for the hardware, as in HF Spaces.", default="cpu-basic"
        )
        run_parser.add_argument(
            "-d", "--detach", action="store_true", help="Run the Job in the background and print the Job ID.", 
        )
        run_parser.add_argument(
            "--token", type=str, help="A User Access Token generated from https://huggingface.co/settings/tokens"
        )
        run_parser.add_argument(
            "command", nargs="...", help="The command to run."
        )
        run_parser.set_defaults(func=RunCommand)

    def __init__(self, args: Namespace) -> None:
        self.docker_image: str = args.dockerImage
        self.environment: dict[str, str] = {
            x.split("=", 1)[0]: x.split("=", 1)[1]
            for x in args.env or []
        }
        self.flavor: str = args.flavor
        self.detach: bool = args.detach
        self.token: Optional[str] = args.token or None
        self.command: list[str] = args.command


    def run(self) -> None:
        input_json = {
            "command":  self.command,
            "arguments": [],
            "environment": self.environment,
            "flavor": self.flavor
        }
        for prefix in (
            "https://huggingface.co/spaces/",
            "https://hf.co/spaces/",
            "huggingface.co/spaces/",
            "hf.co/spaces/",
        ):
            if self.docker_image.startswith(prefix):
                input_json["spaceId"] = self.docker_image[len(prefix):]
                break
        else:
            input_json["dockerImage"] = self.docker_image
        username = whoami(self.token)["name"]
        headers = build_hf_headers(token=self.token, library_name="hfjobs")
        resp = requests.post(
            f"https://huggingface.co/api/jobs/{username}",
            json=input_json,
            headers=headers,
        )
        resp.raise_for_status()

        job_id = resp.json()["metadata"]["job_id"]
        if self.detach:
            print(job_id)
            return

        logging_finished = False
        job_finished = False
        # We need to retry because sometimes the /logs-stream doesn't return logs when the job just started.
        # For example it can return only two lines: one for "Job started" and one empty line.
        while True:
            resp = requests.get(
                f"https://huggingface.co/api/jobs/{username}/{job_id}/logs-stream",
                headers=headers,
                stream=True,
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
                f"https://huggingface.co/api/jobs/{username}/{job_id}",
                headers=headers,
            ).json()
            if "status" in job_status and job_status["status"]["stage"] not in ("RUNNING", "UPDATING"):
                job_finished = True
            time.sleep(1)
