import os

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PHOTOS = os.path.join(ROOT, "photos")
OUT = os.path.join(ROOT, "results", "correspondence_search")
CALIB = os.path.join(ROOT, "results", "calibration", "calibration.json")
SCENES = ["yunnan_rock_1", "yunnan_rock_2", "yunnan_rock_1_lower_height", "yunnan_rock_2_lower_height"]
NORMS = {"rootsift": "L1_ROOT", "sift": "L2"}

MAX_NUM_FEATURES = 4000
MAX_IMAGE_SIZE = 3200
PEAK_THRESHOLD = 0.02 / 3
EDGE_THRESHOLD = 10.0
CACHE_VERSION = "port-2"

MAX_RATIO = 0.8
RATIOS = [0.6, 0.7, 0.8, 0.9, 1.0]
MAX_DISTANCE = 0.7

MAX_ERROR = 4.0
CONFIDENCE = 0.999
MIN_NUM_TRIALS = 100
MAX_NUM_TRIALS = 10000
MIN_INLIER_RATIO = 0.25
DYN_TRIALS_MULTIPLIER = 3.0
LOCAL_TRIALS = 10

MIN_NUM_INLIERS = 15
MAX_H_INLIER_RATIO = 0.8
ROTATION_TOL = 1e-3

LABELS = {"general": "good", "planar": "flat", "panoramic": "rotation-only", "rejected": "suspicious"}
USABLE = ("general", "planar", "panoramic")
