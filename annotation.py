import cv2
import json
import numpy as np


def annotate_parking_spots(image_path, output_json='zones.json', lot_id=0):
    """
    Manually annotate parking spots by clicking 4 corners per spot.
    After 4 clicks, an on-screen prompt appears to type Spot ID (any length), confirmed by Enter.
    Press ESC during prompt to cancel this spot.
    Main keys:
      q: save & exit
      r: reset all zones
      Esc: cancel annotation (exit without saving)
    Saves JSON with structure:
      {
        "lot_id": <int>,
        "zones": [ {"spot_id": <str>, "coords": [[x,y],...]} ]
      }
    """
    image = cv2.imread(image_path)
    image_display = image.copy()
    parking_zones = []
    current_points = []
    cancelled = False

    # Fullscreen window
    win_name = 'Annotate Parking Spots'
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(win_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    def click_event(event, x, y, flags, param):
        nonlocal current_points, parking_zones, image_display
        if event == cv2.EVENT_LBUTTONDOWN:
            current_points.append((x, y))
            cv2.circle(image_display, (x, y), 5, (0, 0, 255), -1)

            if len(current_points) == 4:
                # Backup before drawing
                backup_img = image_display.copy()
                # Draw polygon
                pts = np.array(current_points, np.int32).reshape(-1, 1, 2)
                cv2.polylines(image_display, [pts], isClosed=True,
                              color=(0, 255, 0), thickness=2)

                # Prompt overlay for typing ID
                prompt = ''
                spot_cancel = False
                while True:
                    prompt_img = image_display.copy()
                    cv2.putText(prompt_img, 'Type Spot ID and press Enter (Esc to cancel):', (50, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
                    cv2.putText(prompt_img, prompt, (50, 100),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                    cv2.imshow(win_name, prompt_img)

                    key = cv2.waitKey(0)
                    if key == 13:  # Enter
                        break
                    elif key == 27:  # ESC
                        spot_cancel = True
                        break
                    elif key in (8, 127):  # Backspace/Delete
                        prompt = prompt[:-1]
                    elif 32 <= key <= 126:
                        prompt += chr(key)

                if spot_cancel:
                    image_display = backup_img.copy()
                    current_points.clear()
                    return

                spot_id = prompt if prompt else f"spot{len(parking_zones)+1}"
                # Label center
                cx = int(sum(p[0] for p in current_points) / 4)
                cy = int(sum(p[1] for p in current_points) / 4)
                cv2.putText(image_display, spot_id, (cx, cy),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

                parking_zones.append({
                    'spot_id': spot_id,
                    'coords': current_points.copy()
                })
                current_points.clear()

    cv2.setMouseCallback(win_name, click_event)

    print("Click 4 corners per spot.")
    print("Controls: q=save & exit, r=reset, Esc=cancel without saving.")
    while True:
        # Overlay Controls Text
        display_img = image_display.copy()
        cv2.putText(display_img, 'q=save  r=reset  Esc=cancel', (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        cv2.imshow(win_name, display_img)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):  # Save & exit
            break
        elif key == ord('r'):  # Reset
            image_display = image.copy()
            parking_zones.clear()
            current_points.clear()
            print("Reset all zones.")
        elif key == 27:  # ESC cancel all
            cancelled = True
            break

    cv2.destroyAllWindows()

    if cancelled:
        print("Annotation cancelled; no zones saved.")
        return None

    # Save JSON
    output = {'lot_id': lot_id, 'zones': parking_zones}
    with open(output_json, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"Saved {len(parking_zones)} zones for lot {lot_id} to {output_json}")
    return output