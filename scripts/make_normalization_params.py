import os
import sys

import click

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from data.DataLoader import PFlowDataset


@click.command()
@click.option(
    "--inputfilename",
    type=str,
    default="/data/multiai/data3/HyperGraph-2212.01328/singleQuarkJet_train.root",
)
@click.option(
    "--outputconfigname",
    type=str,
    default="data/normalization/params.json",
)
def main(**args):

    dataset = PFlowDataset(
        filename=args["inputfilename"],
        num_read=-1,
        add_vars_for_visualization=True,
    )
    dataset.dump_normalization_params(outputname=args["outputconfigname"])


if __name__ == "__main__":
    main()
