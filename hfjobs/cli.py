from argparse import ArgumentParser

from .commands.ps import PsCommand
from .commands.run import RunCommand

def main():
    
    parser = ArgumentParser("hfjobs", usage="hfjobs <command> [<args>]")
    commands_parser = parser.add_subparsers(help="hfjobs command helpers")

    # Register commands
    PsCommand.register_subcommand(commands_parser)
    RunCommand.register_subcommand(commands_parser)

    # Let's go
    args = parser.parse_args()
    if not hasattr(args, "func"):
        parser.print_help()
        exit(1)

    # Run
    service = args.func(args)
    service.run()

if __name__ == "__main__":
    main()