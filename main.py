import argparse
from distutils.util import strtobool
import time
from datetime import datetime

from draw import animate_traj
from generate_reference import generate_reference
from traj_opt import traj_opt
from export import export_to_csv

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        "--display",
        help="toggle whether to display animation window",
        type=strtobool,
        default=1,
    )
    parser.add_argument("-n", "--name", help="experiment name", type=str, default=None)
    parser.add_argument(
        "-s", "--save", help="toggle whether to save motion", type=strtobool, default=0
    )
    parser.add_argument(
        "-e",
        "--export",
        help="toggle whether to export trajectory to csv",
        type=strtobool,
        default=0,
    )

    # parse and post processing
    args = parser.parse_args()
    args.display = bool(args.display)
    args.save = bool(args.save)
    if args.name is None:
        args.name = datetime.now().strftime("%m-%d-%Y_%H-%M-%S")

    # generate reference, optionally display/save animation
    X_ref, U_ref, dt, motion_options = generate_reference()
    if args.save:
        fname_ref = args.name + "-ref"
    else:
        fname_ref = None
    animate_traj(
        X_ref, U_ref, dt, fname_ref, display=args.display, motion_options=motion_options
    )

    # optionally export reference to csv
    if args.export:
        export_to_csv(X_ref, U_ref, dt, args.name + "-ref", motion_options)

    # solve trajectory optimization
    start_time = time.time()
    X_sol, U_sol = traj_opt(X_ref, U_ref, dt, motion_options)
    print("\nOptimization took {} minutes".format((time.time() - start_time) / 60.0))

    # optionally export trajectory to csv
    if args.export:
        export_to_csv(X_sol, U_sol, dt, args.name, motion_options)

    # optionally display/save solution animation
    if args.save:
        fname = args.name
    else:
        fname = None
    animate_traj(
        X_sol, U_sol, dt, fname, display=args.display, motion_options=motion_options
    )
