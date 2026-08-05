"""Geographic regions used to scope both training data and evaluation.

TRAIN_REGIONS are new, coastal-leaning centers (a demo-scale stand-in for
AstroLoc's real training scope: "worldwide minus the held-out eval regions").
They're deliberately distinct from the 6 EarthLoc/AstroLoc benchmark regions
this repo already evaluates on (scripts/evaluate.py::REGIONS, reproduced here
as EVAL_REGIONS) so recall@k on those regions stays a reasonably meaningful
zero-shot check, not train-on-the-test-set. Some geographic proximity to an
eval region is tolerated (see closest-distance note below) -- AstroLoc's own
training set is "worldwide" and does not literally excise everything near its
6 benchmark regions either, it just holds those specific areas out.

Picked for coastal/high-saliency content per the existing repo's own saliency
finding (memory: argus-phase0 companion analysis ranked Napa/Toshka Lakes
highest for coastline content, Amazon/Gobi lowest for uniform texture) --
mirrors that logic by choosing archipelagos/coastlines rather than deep
interior/desert/rainforest.
"""

TRAIN_REGIONS = {
    # name: (center_lat, center_lon)
    "SE_Asia": (12, 122),  # Philippines archipelago
    "Australia_East": (-25, 150),  # Great Barrier Reef coast
    "Mediterranean": (37, 25),  # Aegean/Greek islands
    "Japan": (36, 138),  # Japanese coastline
    "West_Africa": (5, 0),  # Gulf of Guinea coast
    "Caribbean": (18, -70),  # Caribbean islands
}

# Same six regions and centers as scripts/evaluate.py::REGIONS (kept as a
# separate copy here, not an import, so astroloc/ stays self-contained and
# doesn't create a two-way dependency between it and the main pipeline).
EVAL_REGIONS = {
    "Alps": (45, 10),
    "Texas": (30, -95),
    "Toshka Lakes": (23, 30),
    "Amazon": (-3, -60),
    "Napa": (38, -122),
    "Gobi": (40, 105),
}

TRAIN_QUERY_RADIUS_KM = 2000  # how far a GAPE photo's nadir point may be from a train center
TRAIN_DB_RADIUS_KM = 2000  # how far a reference tile's nadir point may be from a train center
TRAIN_DB_YEAR = 2021  # single year, matches config.yaml's eval.db_year convention
