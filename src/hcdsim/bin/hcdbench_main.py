import argparse
from hcdsim.bench import cnclass, cndetect, hccnstable, hconsetacc, hconsetcn, subdetect, mirrorsubclone,hccnchange,hcPhasing

def main():
    """
    Main entry point for the CLI tool.
    """

    parser = argparse.ArgumentParser(description="HCDSIM Benchmark Utility")
    subparsers = parser.add_subparsers(dest="command", help="Subcommands")

    # Add subcommands
    subdetect.add_subdetect_subparser(subparsers)
    cndetect.add_cndetect_subparser(subparsers)
    hconsetacc.add_hconsetacc_subparser(subparsers)
    hconsetcn.add_hconsetcn_subparser(subparsers)
    hccnchange.add_hccnchange_subparser(subparsers)
    hccnstable.add_hccnstable_subparser(subparsers)
    mirrorsubclone.add_mirrorsubclone_subparser(subparsers)
    cnclass.add_cnclass_subparser(subparsers)
    hcPhasing.add_hcPhasing_subparser(subparsers)



    # Parse arguments
    args = parser.parse_args()

    # Execute corresponding logic based on the command
    if args.command == "subdetect": 
        subdetect.run(args)
    elif args.command == "cndetect":
        cndetect.run(args)
    elif args.command == "hconsetacc":
        hconsetacc.run(args)
    elif args.command == "hconsetcn":
        hconsetcn.run(args)
    elif args.command == "hccnchange":
        hccnchange.run(args)
    elif args.command == "hccnstable":
        hccnstable.run(args)
    elif args.command == "mirrorsubclone":
        mirrorsubclone.run(args)
    elif args.command == "cnclass":
        cnclass.run(args)
    elif args.command == "hcPhasing":
        hcPhasing.run(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()