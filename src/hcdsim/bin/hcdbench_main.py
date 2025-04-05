import argparse
from hcdsim.bench import subdetect, cnaccuracy,hcidentify,cnretain,mirrorsubclone,cnstates,hcparent,hcCNchange

def main():
    """
    Main entry point for the CLI tool.
    """

    parser = argparse.ArgumentParser(description="HCDSIM Benchmark Utility")
    subparsers = parser.add_subparsers(dest="command", help="Subcommands")

    # Add subcommands
    subdetect.add_subdetect_subparser(subparsers)
    cnaccuracy.add_cnaccuracy_subparser(subparsers)
    hcidentify.add_hcidentify_subparser(subparsers)
    hcparent.add_hcparent_subparser(subparsers)
    hcCNchange.add_hcCNchange_subparser(subparsers)
    cnretain.add_cnretain_subparser(subparsers)
    mirrorsubclone.add_mirrorsubclone_subparser(subparsers)
    cnstates.add_cnstates_subparser(subparsers)


    # Parse arguments
    args = parser.parse_args()

    # Execute corresponding logic based on the command
    if args.command == "subdetect": 
        subdetect.run(args)
    elif args.command == "cnaccuracy":
        cnaccuracy.run(args)
    elif args.command == "hcidentify":
        hcidentify.run(args)
    elif args.command == "hcparent":
        hcparent.run(args)
    elif args.command == "hcCNchange":
        hcCNchange.run(args)
    elif args.command == "cnretain":
        cnretain.run(args)
    elif args.command == "mirrorsubclone":
        mirrorsubclone.run(args)
    elif args.command == "cnstates":
        cnstates.run(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()