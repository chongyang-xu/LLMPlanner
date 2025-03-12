import os
import json

import time
import numpy as np
import gradio as gr
from PIL import Image
from pathlib import Path

from palimpzest.sets import Dataset
from palimpzest.constants import OptimizationStrategy

from palimpzest.core.data.datasources import UserSource
from palimpzest.core.lib.schemas import Schema
from palimpzest.core.lib.fields import BooleanField, ImageFilepathField, ListField, NumericField, StringField
from palimpzest.core.elements.records import DataRecord

from palimpzest.policy import MaxQuality
from palimpzest.query import (
    Execute,
    NoSentinelPipelinedParallelExecution,
    NoSentinelPipelinedSingleThreadExecution,
    NoSentinelSequentialSingleThreadExecution,
)

from palimpzest.datamanager.datamanager import DataDirectory

os.makedirs("profiling-data", exist_ok=True)

datasetid = "real-estate-eval-10"
workload = "real-estate"
visualize = False
verbose = False
profile = False
policy = MaxQuality()
execution_engine = NoSentinelPipelinedSingleThreadExecution
"""
    executor = args.executor
    if executor == "sequential":
        execution_engine = NoSentinelSequentialSingleThreadExecution
    elif executor == "pipelined":
        execution_engine = NoSentinelPipelinedSingleThreadExecution
    elif executor == "parallel":
        execution_engine = NoSentinelPipelinedParallelExecution
    else:
        print("Executor not supported for this demo")
"""

# Addresses far from MIT; we use a simple lookup like this to make the
# experiments re-producible w/out needed a Google API key for geocoding lookups
FAR_AWAY_ADDRS = [
    "Melcher St",
    "Sleeper St",
    "437 D St",
    "Seaport Blvd",
    "50 Liberty Dr",
    "Telegraph St",
    "Columbia Rd",
    "E 6th St",
    "E 7th St",
    "E 5th St",
]


def within_two_miles_of_mit(record: dict):
    # NOTE: I'm using this hard-coded function so that folks w/out a
    #       Geocoding API key from google can still run this example
    try:
        return not any([
            street.lower() in record["address"].lower()
            for street in FAR_AWAY_ADDRS
        ])
    except Exception:
        return False


def in_price_range(record: dict):
    try:
        price = record["price"]
        if isinstance(price, str):
            price = price.strip()
            price = int(price.replace("$", "").replace(",", ""))
        return 6e5 < price <= 2e6
    except Exception:
        return False


class RealEstateListingFiles(Schema):
    """The source text and image data for a real estate listing."""

    listing = StringField(desc="The name of the listing")
    text_content = StringField(
        desc="The content of the listing's text description")
    image_filepaths = ListField(
        element_type=ImageFilepathField,
        desc="A list of the filepaths for each image of the listing",
    )


class TextRealEstateListing(RealEstateListingFiles):
    """Represents a real estate listing with specific fields extracted from its text."""

    address = StringField(desc="The address of the property")
    price = NumericField(desc="The listed price of the property")


class ImageRealEstateListing(RealEstateListingFiles):
    """Represents a real estate listing with specific fields extracted from its text and images."""

    is_modern_and_attractive = BooleanField(
        desc=
        "True if the home interior design is modern and attractive and False otherwise"
    )
    has_natural_sunlight = BooleanField(
        desc=
        "True if the home interior has lots of natural sunlight and False otherwise"
    )


class RealEstateListingSource(UserSource):

    def __init__(self, dataset_id, listings_dir):
        super().__init__(RealEstateListingFiles, dataset_id)
        self.listings_dir = listings_dir
        self.listings = sorted(os.listdir(self.listings_dir))

    def copy(self):
        return RealEstateListingSource(self.dataset_id, self.listings_dir)

    def __len__(self):
        return len(self.listings)

    def get_size(self):
        return sum(
            file.stat().st_size for file in Path(self.listings_dir).rglob("*"))

    def get_item(self, idx: int):
        # fetch listing
        listing = self.listings[idx]

        # create data record
        dr = DataRecord(self.schema, source_id=listing)
        dr.listing = listing
        dr.image_filepaths = []
        listing_dir = os.path.join(self.listings_dir, listing)
        for file in os.listdir(listing_dir):
            if file.endswith(".txt"):
                with open(os.path.join(listing_dir, file), "rb") as f:
                    dr.text_content = f.read().decode("utf-8")
            elif file.endswith(".png"):
                dr.image_filepaths.append(os.path.join(listing_dir, file))

        return dr


workload = "real-estate"
# datasetid="real-estate-eval-100" for paper evaluation
data_filepath = f"/home/cxu/ws/llm/LLMPlanner/scripts/pz/testdata/{datasetid}"
user_dataset_id = f"{datasetid}-user"
DataDirectory().register_user_source(
    src=RealEstateListingSource(user_dataset_id, data_filepath),
    dataset_id=user_dataset_id,
)

plan = Dataset(user_dataset_id, schema=RealEstateListingFiles)
plan = plan.convert(TextRealEstateListing, depends_on="text_content")
plan = plan.convert(ImageRealEstateListing, depends_on="image_filepaths")

plan = plan.filter(
    "The interior is modern and attractive, and has lots of natural sunlight",
    depends_on=["is_modern_and_attractive", "has_natural_sunlight"],
)
plan = plan.filter(within_two_miles_of_mit, depends_on="address")
plan = plan.filter(in_price_range, depends_on="price")

start_time = time.time()
# execute pz plan
records, execution_stats = Execute(
    plan,
    policy,
    nocache=True,
    optimization_strategy=OptimizationStrategy.PARETO,
    execution_engine=execution_engine,
    verbose=verbose,
)
end_time = time.time()
print("Elapsed time:", end_time - start_time)
# 79.18507885932922 s
# 69.80556654930115
# 69.78277802467346
"""
addr: 362-366 Commonwealth Ave Unit 4C, Boston, MA, 02115, price: $609,900
        pz/testdata/real-estate-eval-10/listing7/img1.png,
        pz/testdata/real-estate-eval-10/listing7/img2.png,
        pz/testdata/real-estate-eval-10/listing7/img3.png
"""

# save statistics
if profile:
    stats_path = f"profiling-data/{workload}-profiling.json"
    execution_stats_dict = execution_stats.to_json()
    with open(stats_path, "w") as f:
        json.dump(execution_stats_dict, f)

# visualize output in Gradio
if visualize:
    from palimpzest.utils.demo_helpers import print_table

plan_str = list(execution_stats.plan_strs.values())[-1]

fst_imgs, snd_imgs, thrd_imgs, addrs, prices = [], [], [], [], []

for record in records:
    addrs.append(record.address)
    prices.append(record.price)
    print(f"addr: {record.address}, price: {record.price}")
    path_str = ""
    for idx, img_name in enumerate(["img1.png", "img2.png", "img3.png"]):
        path = os.path.join(data_filepath, record.listing, img_name)
        img = Image.open(path)
        path_str = path_str + path + ", "
        img_arr = np.asarray(img)
        if idx == 0:
            fst_imgs.append(img_arr)
        elif idx == 1:
            snd_imgs.append(img_arr)
        elif idx == 2:
            thrd_imgs.append(img_arr)
    print(f"\t{path_str}")

with gr.Blocks() as demo:
    fst_img_blocks, snd_img_blocks, thrd_img_blocks, addr_blocks, price_blocks = [], [], [], [], []
    for fst_img, snd_img, thrd_img, addr, price in zip(fst_imgs, snd_imgs,
                                                       thrd_imgs, addrs,
                                                       prices):
        with gr.Row(equal_height=True):
            with gr.Column():
                fst_img_blocks.append(gr.Image(value=fst_img))
            with gr.Column():
                snd_img_blocks.append(gr.Image(value=snd_img))
            with gr.Column():
                thrd_img_blocks.append(gr.Image(value=thrd_img))
        with gr.Row():
            with gr.Column():
                addr_blocks.append(gr.Textbox(value=addr, info="Address"))
            with gr.Column():
                price_blocks.append(gr.Textbox(value=price, info="Price"))

    plan_str = list(execution_stats.plan_strs.values())[0]
    gr.Textbox(value=plan_str, info="Query Plan")

demo.launch()
