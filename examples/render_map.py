from tinycarlo.map import Map
from tinycarlo.renderer import Renderer
from tinycarlo.helper import getenv

import os
import sys
import cv2

map_config = {
    "json_path": sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "maps/knuffingen.json"),
    "pixel_per_meter": int(sys.argv[2]) if len(sys.argv) > 2 else 266
}
map = Map(map_config)
bg_color = (255,255,255) if getenv("WHITE") else None
renderer = Renderer(map, overview_pixel_per_meter=map_config["pixel_per_meter"], background_color=bg_color, line_thickness=2)
overview = renderer.render_overview()

cv2.namedWindow("Overview", cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_EXPANDED)
# Close the window with a key press
# https://github.com/opencv/opencv/issues/20822
cv2.imshow("Overview", overview)
cv2.waitKey(0)
cv2.destroyAllWindows()

