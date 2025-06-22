from detector import detect_objects
from annotation import annotate_parking_spots
from utils import load_zones, compute_occupancy, save_occupancy_csv, save_occupancy_json, post_occupancy

def main():
    # 1) Annotate once (or comment out if zones.json already exists)
    # Only run this when defining zones for the first time or updating
    # result = annotate_parking_spots("data/EmptyTestLot.jpg", "data/zones.json", lot_id=0)
    # if result is None:
    #     return  # annotation cancelled, exit

    # 2) Detect objects on a full lot image
    # TEST FILES NAMES
        # EmptyTestLot
        # FullTestLot
        # FullTestLotWithGlare
        # HalfFullTestLot
        # MostlyEmptyTestLot
        # 1_TestingLot
    detections = detect_objects("data/1_TestingLot.jpg")

    # 3) Load zones and compute occupancy
    lot_id, zones = load_zones("data/zones.json")
    occupancy = compute_occupancy(detections, zones, lot_id)

    # 3.1) Output summary to console
    total_spots = len(occupancy)
    full_spots = sum(1 for spot in occupancy if spot['taken'])
    print(f"Summary: {full_spots} of {total_spots} spots are occupied.")

    # 4) Save for POST (choose CSV or JSON)
    save_occupancy_csv(occupancy, "data/occupancy.csv")
    # save_occupancy_json(occupancy, "data/occupancy.json")

    # 5) Post occupancy data to remote service
    post_occupancy(occupancy)

if __name__ == "__main__":
    main()