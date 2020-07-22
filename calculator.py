import json
import logging
import sys
from argparse import ArgumentParser
from pathlib import Path
from typing import NamedTuple

import pandas as pd

# Assumptions:
# * All compute time for a given compute specification can be done on a single machine
# * A month of compute time is 730 hours
# * All compute is completed within a month
#   * For 1/3 years reserved instances, we give the compute costs in hour only, be aware you will need to pay the full
#     730h/month that a reserved instance cost x 12 months x number of years selected

# TODO: Figure out why it picks linux-d32sv4-standard and linux-d32v4-standard when they are the same price and same spec
# TODO: Support reservation costs estimates
# TODO: Support other requirements (maximize RAM, then CPU)
# TODO: Find costs around a given budget
# TODO: Better match CPU models?
# TODO: Consider specific compute offers (e.g., DB)
# TODO: For 1/3 years reserved, compute the total payment required
# TODO: For 1/3 years reserved,

pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)
pd.set_option('display.width', None)

logger = logging.getLogger(__name__)

current_directory = Path(__file__).parent

file = current_directory / "azure-virtual-machines.json"
logger.debug(f"Loading {file}")
with Path(file).open("r") as f:
    azure_virtual_machines = json.load(f)

regions = azure_virtual_machines["regions"]
regions_slug = [region["slug"] for region in regions]

file = current_directory / "azure-currencies.json"
logger.debug(f"Loading {file}")
with Path(file).open("r") as f:
    azure_currencies = json.load(f)

argument_parser = ArgumentParser()
argument_parser.add_argument(
    "query",
    help="Compute resources requirements (cpu,mem,gpu,gpu_mem,total_hours) "
         "in CSV format, use - to read from stdin",
)
default_region = "us-west"
argument_parser.add_argument(
    "--region",
    help=f"Region(s) where the services will be located (default: {default_region})",
    default=[default_region],
    choices=regions_slug,
    nargs="+",
)
default_currency = "usd"
argument_parser.add_argument(
    "--currency",
    help=f"Currency used to provide total estimated cost (default: {default_currency}",
    default=default_currency,
    choices=azure_currencies.keys(),
)
default_tier = "standard"
argument_parser.add_argument(
    "--tier",
    help=f"Tier(s) to use for compute (default: {default_tier})",
    default=[default_tier],
    choices=["basic", "lowpriority", "standard"],
    nargs="+",
)
default_pricing_scheme = "perhour"
argument_parser.add_argument(
    "--pricing-scheme",
    help=f"Pricing scheme(s) to consider (default: {default_pricing_scheme})",
    default=[default_pricing_scheme],
    choices=["perhour", "perhourspot", "perhouroneyearreserved", "perhourthreeyearreserved"],
    nargs="+",
)
argument_parser.add_argument(
    "--log-level",
    default=logging.getLevelName(logging.INFO),
    choices=logging._levelToName.values(),
)

args = argument_parser.parse_args()

logging.basicConfig(level=args.log_level)

logger.info(f"Regions: {args.region}")
logger.info(f"Tiers: {args.tier}")
logger.info(f"Pricing schemes: {args.pricing_scheme}")
logger.info(f"Currency: {args.currency}")

offers = azure_virtual_machines["offers"]
linux_offers = {
    offer_name: offer
    for offer_name, offer in offers.items()
    if offer_name.startswith("linux")
}
logger.debug(f"{len(linux_offers)} linux offers found.")

tiers = tuple(args.tier)
# Prepare a dataframe of compute resources
offer_rows = []
for offer_name, linux_offer in linux_offers.items():
    if not offer_name.endswith(tiers):
        continue

    for pricing_scheme in args.pricing_scheme:
        per_hour_pricing = linux_offer["prices"].get(pricing_scheme)

        if not per_hour_pricing:
            continue

        for region in args.region:
            region_price = per_hour_pricing.get(region)

            if not region_price:
                continue

            offer_rows += [
                {
                    "region": region,
                    "offer": offer_name,
                    "pricing_scheme": pricing_scheme,
                    "cpu": linux_offer["cores"],
                    "mem": linux_offer["ram"],
                    # TODO: Read GPU info from the gpu field
                    "gpu": 1 if linux_offer.get("gpu") is not None else 0,
                    "gpu_mem": 12 if linux_offer.get("gpu") is not None else 0,
                    "price": region_price["value"],
                }
            ]

compute_df = pd.DataFrame.from_records(offer_rows)
logger.debug(
    f"Created compute_df with {len(compute_df)} entries. "
    f"Some entries may have been filtered due to the unavailability "
    f"of per hour pricing or the compute not being available in the selected region."
)

file = sys.stdin if args.query == "-" else args.query
query = pd.read_csv(file)
logger.info(f"Your query:\n{query}")


class ComputeQuery(NamedTuple):
    cpu: int
    mem: int
    gpu: int
    gpu_mem: int


def find_closest_match(compute_df: pd.DataFrame, compute_query: ComputeQuery):
    filter_cpu = compute_df["cpu"] >= compute_query.cpu
    filter_mem = compute_df["mem"] >= compute_query.mem
    filter_gpu = compute_df["gpu"] >= compute_query.gpu
    filter_gpu_mem = compute_df["gpu_mem"] >= compute_query.gpu_mem
    filtered_df = compute_df[filter_cpu & filter_mem & filter_gpu & filter_gpu_mem]

    if len(filtered_df) == 0:
        logger.error(f"Cannot find compute that will fit compute query {compute_query}")
        return None

    return filtered_df.sort_values("price").iloc[0]


logger.debug("Closest matching compute:")
# For each query, find the closest matching compute that fits all the requirements

unaccounted_costs = []
matches = []
total_cost = 0
for compute_query in query.itertuples(index=False, name="ComputeQuery"):
    logger.debug(f"Match for {compute_query}")
    match = find_closest_match(compute_df, compute_query)

    if match is None:
        unaccounted_costs += [compute_query]
        continue

    match["total_hours"] = compute_query.total_hours
    match["total"] = match["price"] * compute_query.total_hours
    matches += [match]

    logger.debug(match)
    compute_cost = match["price"] * compute_query.total_hours
    logger.debug(f"Compute cost: {compute_cost}")
    total_cost += compute_cost

matches = pd.DataFrame.from_records(matches)
matches = matches.groupby(["region", "offer", "pricing_scheme", "cpu", "mem", "gpu", "gpu_mem", "price"]).sum()
matches = matches.sort_values(["total"])
matches = matches.reset_index()

logger.info(f"Detailed costs (sorted from lowest to highest total):\n{matches}")

total_cost = round(total_cost, 2)
logger.debug(f"Total cost (USD): {total_cost}")

currency_details = azure_currencies[args.currency]
total_cost_in_currency = round(currency_details["conversion"] * total_cost, 2)

logger.info(f"Total cost ({currency_details['name']}): {total_cost_in_currency}")

if len(unaccounted_costs) > 0:
    logger.info(
        f"Could not account for the following compute queries: {unaccounted_costs}"
    )
