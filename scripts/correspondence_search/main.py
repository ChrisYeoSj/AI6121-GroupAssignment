import os
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import argparse
import config as C


def run(scenes, norm="rootsift", ratio=C.MAX_RATIO, max_distance=C.MAX_DISTANCE,
        min_inliers=C.MIN_NUM_INLIERS, tag=None, workers=None):
    from utils.pipeline import process_scene
    for s in scenes or C.SCENES:
        process_scene(s, norm, ratio, max_distance, min_inliers, tag, workers)


def experiments(execute, sweep_scene, workers=None):
    from utils.analysis import run_experiments, write_table
    if execute:
        run_experiments(sweep_scene, workers)
    print(write_table())


def graph(scenes, tag):
    from utils.analysis import draw
    for s in scenes or C.SCENES:
        draw(s, tag)


def check(scenes, tag):
    from load import load
    for s in scenes or C.SCENES:
        r = load(s, tag)
        print(s, len(r["image_names"]), "images", len(r["matches"]), "pairs", len(r["tracks"]), "tracks")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("run")
    p.add_argument("scenes", nargs="*")
    p.add_argument("--norm", choices=list(C.NORMS), default="rootsift")
    p.add_argument("--ratio", type=float, default=C.MAX_RATIO)
    p.add_argument("--max-distance", type=float, default=C.MAX_DISTANCE)
    p.add_argument("--min-inliers", type=int, default=C.MIN_NUM_INLIERS)
    p.add_argument("--tag")
    p.add_argument("--workers", type=int)
    p = sub.add_parser("experiments")
    p.add_argument("--run", action="store_true")
    p.add_argument("--sweep-scene", default="yunnan_rock_1")
    p.add_argument("--workers", type=int)
    for name in ("graph", "load"):
        p = sub.add_parser(name)
        p.add_argument("scenes", nargs="*")
        p.add_argument("--tag", default="rootsift")
    p = sub.add_parser("all")
    p.add_argument("--sweep-scene", default="yunnan_rock_1")
    p.add_argument("--workers", type=int)
    a = ap.parse_args()
    if a.cmd == "run":
        run(a.scenes, a.norm, a.ratio, a.max_distance, a.min_inliers, a.tag, a.workers)
    elif a.cmd == "experiments":
        experiments(a.run, a.sweep_scene, a.workers)
    elif a.cmd == "graph":
        graph(a.scenes, a.tag)
    elif a.cmd == "load":
        check(a.scenes, a.tag)
    else:
        run([], workers=a.workers)
        experiments(True, a.sweep_scene, a.workers)
        graph([], "rootsift")
        check([], "rootsift")


if __name__ == "__main__":
    main()
