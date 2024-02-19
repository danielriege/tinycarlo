import cv2
import sys
import numpy as np
import json

from typing import List, Tuple, Dict, Any

from mapbuilder.layer_builder import LayerBuilder

layer_builders: List[LayerBuilder] = []
dragging: bool = False

colors: Dict[str, Tuple[int, int, int]] = {
    "red": (0, 0, 255),
    "green": (0, 255, 0),
    "blue": (255, 0, 0),
    "yellow": (0, 255, 255),
    "magenta": (255, 0, 255),
    "cyan": (255, 255, 0),
    "white": (255, 255, 255),
    "purple": (128, 0, 128),
    "orange": (0, 165, 255),
    "pink": (147, 20, 255),
    "brown": (42, 42, 165),
    "gray": (128, 128, 128)
}

# Load an image to use as background
if len(sys.argv) == 2:
    image: np.ndarray = cv2.imread(sys.argv[1])  # Replace "background.jpg" with your image file path
else:
    image: np.ndarray = np.zeros((512, 512, 3), np.uint8)  # Create a black image if no image file is provided

# Mouse callback function
def on_mouse(event, x, y, flags, params):
    global layer_builders, dragging
    lb = layer_builders[-1]
    if event == cv2.EVENT_LBUTTONUP:
        nearest_node_idx = lb.find_nearest_node(x, y)
        if nearest_node_idx is not None:
            # Add edge to the nearest node > Loop Closure
            lb.loop_closure(nearest_node_idx)
            lb.selected_node_idx = None
        else:
            lb.add_new_node(x, y)
    elif event == cv2.EVENT_RBUTTONDOWN:
        if lb.selected_node_idx is not None:
            nearest_node_idx = lb.find_nearest_node(x, y)
            if nearest_node_idx is not None and nearest_node_idx == lb.selected_node_idx:
                lb.selected_node_idx = None
                return
        lb.select_node(x, y)
        dragging = True
    elif event == cv2.EVENT_RBUTTONUP:
        dragging = False
    elif event == cv2.EVENT_MOUSEMOVE and dragging:
        lb.move_selected_node(x, y)

def print_manual() -> None:
    print("============== Map Builder Manual ==============")
    print("This tool is used to build a map consisting of lanelines and a lanepaths.")
    print("You can create multiple layers, each representing a different laneline or a lanepath, but only one lanepath is allowed.")
    print("")
    print("Press 'q' to finish map building and quit (map saved as map.json)")
    print("Press 'n' to create new layer")
    print("   You will be prompted to enter the layer name. If left blank, map building will finish.")
    print("   This will save the last layer which then cannot be changed anymore.")
    print("Press 'u' to undo last action in the current layer")
    print("Press 'r' to reset the current layer")
    print("")
    print("Layer Building:")
    print("   Left click to add a new node. Edge will be added to the selected node (marked in blue).")
    print("   Right click to select a node")
    print("   Right click and drag to move a node")
    print("Note: You can use the mouse wheel to zoom and navigate the map.")
    print("It is recommended to not make the distance between nodes bigger than roughly 4 car lengths.")
    print("================================================")

def finish_map_building() -> None:
    map_dict: Dict[str, Any] = {
        "width": image.shape[1],
        "height": image.shape[0],
        "lanelines": {},
        "lanepath": {}
        }
    for lb in layer_builders:
        layer_name, layer_dict = lb.build_layer_dict()
        if lb.is_lane_path:
            map_dict["lanepath"] = layer_dict
        else:
            map_dict["lanelines"][layer_name] = layer_dict
    print("Saving map to map.json ...")
    with open("map.json", "w") as f:
        json.dump(map_dict, f)

        

def main() -> None:
    global layer_builders, image, color_wheel, current_color_idx
    cv2.namedWindow("Map Builder", cv2.WINDOW_GUI_NORMAL)
    cv2.setMouseCallback("Map Builder", on_mouse)
    cv2.imshow("Map Builder", image)
    cv2.waitKey(1)
    print_manual()

    run_map_building: bool = True
    
    while run_map_building:
        is_path_line: bool = True if input("Is this the lanepath ? (y/n): ").lower() == "y" else False
        if not is_path_line:
            layer_name: str = input("Enter layer name (Leave blank to finish map building): ")
            if layer_name == "":
                # Build the final map
                run_map_building = False
                break
            layer_color = colors[input("Enter layer color (red, green, blue, yellow, magenta, cyan, white, purple, orange, pink, brown, gray): ").lower()]
        else:
            layer_name = "lanepath"
            layer_color = colors["gray"]
        print("Now building layer: " + layer_name + " ...")

        
        layer_builders.append(LayerBuilder(layer_name, layer_color, is_path_line, image))

        while True:
            rendered_image = layer_builders[-1].render_current_layer()
            cv2.imshow("Map Builder", rendered_image)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):  # Press 'q' to quit
                run_map_building = False
                break
            elif key == ord("n"):  # Press 'n' to create new layer
                print("Finished building layer: " + layer_name)
                image = layer_builders[-1].render_final()
                # Saving current progress
                finish_map_building()
                break
            elif key == ord("u"):  # Press 'u' to undo
                layer_builders[-1].undo()
            elif key == ord("r"):  # Press 'r' to reset the layer
                layer_builders[-1].reset()
    
    cv2.destroyAllWindows()
    finish_map_building()


if __name__ == "__main__":
    main()
