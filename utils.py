import json
import csv
import numpy as np
import io
import requests

SERVICE_URL = "https://8aa0-152-117-67-119.ngrok-free.app/data"  # endpoint expecting JSON list

def post_occupancy(occupancy_list, url=SERVICE_URL):
    """Send occupancy data as CSV via HTTP POST."""
    try:
        # Convert occupancy_list to CSV in-memory
        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=['lot_id', 'spot_id', 'taken'])
        writer.writeheader()
        writer.writerows(occupancy_list)

        csv_data = output.getvalue()

        headers = {
            "Content-Type": "text/csv"
        }

        response = requests.post(url, data=csv_data.encode('utf-8'), headers=headers)
        response.raise_for_status()
        print(f"Posted occupancy CSV successfully. Status: {response.status_code}")

    except requests.exceptions.RequestException as e:
        print(f"HTTP error posting occupancy data: {e}")



def generate_cluster_points(x1, y1, x2, y2, grid_size=3, margin_ratio=0.3):
    """
    Generate a grid of points inside the bounding box around the center.
    - grid_size: Number of points per axis (3 = 3x3 = 9 points)
    - margin_ratio: How tight the cluster is around the center (0.0 = full box, 0.3 = centered cluster)
    """
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    bw = x2 - x1
    bh = y2 - y1

    margin_x = bw * margin_ratio / 2
    margin_y = bh * margin_ratio / 2

    xs = np.linspace(cx - margin_x, cx + margin_x, grid_size)
    ys = np.linspace(cy - margin_y, cy + margin_y, grid_size)

    points = [(int(x), int(y)) for x in xs for y in ys]
    return points

def load_zones(json_path):
    """Load zones JSON and return lot_id and list of zones."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data['lot_id'], data['zones']


def point_in_poly(x, y, poly):
    """
    Ray casting algorithm for testing if a point is inside a polygon.
    poly: list of (x,y) vertices.
    """
    num = len(poly)
    j = num - 1
    inside = False
    for i in range(num):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if ((yi > y) != (yj > y)) and (
                x < (xj - xi) * (y - yi) / (yj - yi + 1e-9) + xi):
            inside = not inside
        j = i
    return inside


def compute_occupancy(detections, zones, lot_id, grid_size=3, margin_ratio=0.3):
    """
    detections: list of {'class', 'confidence', 'box': [x1,y1,x2,y2]}
    zones: list of {'spot_id', 'coords': [[x,y], ...]}
    Returns list of {'lot_id','spot_id','taken'} with refined cluster point checks near detection center.
    """
    # Initialize all spots as empty
    status = {z['spot_id']: False for z in zones}

    for det in detections:
        x1, y1, x2, y2 = det['box']
        cluster_points = generate_cluster_points(x1, y1, x2, y2, grid_size, margin_ratio)

        for px, py in cluster_points:
            for z in zones:
                if point_in_poly(px, py, z['coords']):
                    status[z['spot_id']] = True
                    break  # Stop checking this zone if one point matched

    # Build result list
    result = []
    for z in zones:
        result.append({
            'lot_id': lot_id,
            'spot_id': z['spot_id'],
            'taken': status[z['spot_id']]
        })
    return result

def save_occupancy_csv(occupancy_list, csv_path):
    """Save occupancy info to CSV with headers LOT_ID,SPOT_ID,TAKEN."""
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=['lot_id','spot_id','taken'])
        writer.writeheader()
        for row in occupancy_list:
            writer.writerow(row)
    print(f"Occupancy saved to CSV: {csv_path}")


def save_occupancy_json(occupancy_list, json_path):
    """Save occupancy info to JSON array."""
    with open(json_path, 'w') as f:
        json.dump(occupancy_list, f, indent=2)
    print(f"Occupancy saved to JSON: {json_path}")