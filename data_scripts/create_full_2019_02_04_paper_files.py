import pandas as pd
import numpy as np
from pathlib import Path

BASE = Path("data")
FHVHV = BASE / "tlc_nyc/fhvhv"

RAW_FILES = [
    FHVHV / "fhvhv_tripdata_2019-02.parquet",
    FHVHV / "fhvhv_tripdata_2019-03.parquet",
    FHVHV / "fhvhv_tripdata_2019-04.parquet",
]

OLD_INFLOW = FHVHV / "clean_aggregate_tlc_inflow_2019-02_2019-03_B_Q_10min.parquet"

OUT_INFLOW = FHVHV / "clean_aggregate_tlc_inflow_2019-02_2019-04_B_Q_10min.parquet"
OUT_OUTFLOW = FHVHV / "clean_aggregate_tlc_outflow_2019-02_2019-04_B_Q_10min.parquet"
OUT_REALTIME = BASE / "realtime_nyc/clean_aggregate_realtime_2019-02_2019-04_B_Q_10min.parquet"

REALTIME_EDGE_FILE = BASE / "realtime_nyc/graph/realtime_edges_final_v1_B_Q.csv"

START = "2019-02-01 00:00:00"
END = "2019-04-30 23:50:00"
FREQ = "10min"

print("Reading old inflow index to preserve zone set:", OLD_INFLOW)
old_inflow = pd.read_parquet(OLD_INFLOW)

if not isinstance(old_inflow.index, pd.MultiIndex):
    raise ValueError("Expected old inflow file to have MultiIndex.")

time_name = old_inflow.index.names[0]
zone_name = old_inflow.index.names[1]

zones = old_inflow.index.get_level_values(1).unique().sort_values().astype(int)
times = pd.date_range(start=START, end=END, freq=FREQ)

print("Number of zones:", len(zones))
print("Number of time buckets:", len(times))

full_index = pd.MultiIndex.from_product(
    [times, zones],
    names=[time_name, zone_name]
)

inflow_parts = []
outflow_parts = []

for raw_file in RAW_FILES:
    if not raw_file.exists():
        raise FileNotFoundError(raw_file)

    print("\nReading raw:", raw_file)

    df = pd.read_parquet(
        raw_file,
        columns=["pickup_datetime", "dropoff_datetime", "PULocationID", "DOLocationID"]
    )

    df = df.dropna(subset=["pickup_datetime", "dropoff_datetime", "PULocationID", "DOLocationID"])
    df["PULocationID"] = df["PULocationID"].astype(int)
    df["DOLocationID"] = df["DOLocationID"].astype(int)
    df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"])
    df["dropoff_datetime"] = pd.to_datetime(df["dropoff_datetime"])

    # Inflow: count arrivals to zones.
    inflow_df = df[df["DOLocationID"].isin(zones)].copy()
    inflow_df["time_bucket"] = inflow_df["dropoff_datetime"].dt.floor(FREQ)
    inflow_counts = (
        inflow_df.groupby(["time_bucket", "DOLocationID"])
        .size()
        .rename("inflow_volume")
        .to_frame()
    )
    inflow_parts.append(inflow_counts)

    # Outflow: count departures from zones.
    outflow_df = df[df["PULocationID"].isin(zones)].copy()
    outflow_df["time_bucket"] = outflow_df["pickup_datetime"].dt.floor(FREQ)
    outflow_counts = (
        outflow_df.groupby(["time_bucket", "PULocationID"])
        .size()
        .rename("outflow")
        .to_frame()
    )
    outflow_parts.append(outflow_counts)

    print("Rows raw:", len(df))
    print("Rows inflow zone filtered:", len(inflow_df))
    print("Rows outflow zone filtered:", len(outflow_df))

print("\nCombining inflow...")
inflow = pd.concat(inflow_parts).groupby(level=[0, 1]).sum()
inflow.index.names = [time_name, zone_name]
inflow = inflow.reindex(full_index, fill_value=0)
inflow["inflow_volume"] = inflow["inflow_volume"].astype("float32")
inflow.to_parquet(OUT_INFLOW)

print("Saved:", OUT_INFLOW)
print("Shape:", inflow.shape)
print("Columns:", list(inflow.columns))

print("\nCombining outflow...")
outflow = pd.concat(outflow_parts).groupby(level=[0, 1]).sum()
outflow.index.names = [time_name, zone_name]
outflow = outflow.reindex(full_index, fill_value=0)
outflow["outflow"] = outflow["outflow"].astype("float32")
outflow.to_parquet(OUT_OUTFLOW)

print("Saved:", OUT_OUTFLOW)
print("Shape:", outflow.shape)
print("Columns:", list(outflow.columns))

print("\nCreating proxy realtime speed file...")
edges = pd.read_csv(REALTIME_EDGE_FILE)

node_values = []
for col in edges.columns:
    vals = pd.to_numeric(edges[col], errors="coerce").dropna().astype(int).tolist()
    node_values.extend(vals)

road_nodes = sorted(set(node_values))
if not road_nodes:
    raise ValueError("No road node IDs found in realtime edge file.")

rt_index = pd.MultiIndex.from_product(
    [times, road_nodes],
    names=["time_bucket", "road_node_id"]
)

time_positions = np.arange(len(times), dtype=np.float32)
daily_cycle = 5.0 * np.sin(2.0 * np.pi * time_positions / 144.0)

speed_values = []
for t_idx in range(len(times)):
    base_speed = 30.0 + daily_cycle[t_idx]
    for node in road_nodes:
        node_adjustment = (node % 7) * 0.3
        speed_values.append(base_speed + node_adjustment)

speed = pd.DataFrame(
    {"speed": np.asarray(speed_values, dtype=np.float32)},
    index=rt_index
)

OUT_REALTIME.parent.mkdir(parents=True, exist_ok=True)
speed.to_parquet(OUT_REALTIME)

print("Saved:", OUT_REALTIME)
print("Shape:", speed.shape)
print("Road nodes:", len(road_nodes))

print("\nDONE: Created February-April paper-mode clean files.")
