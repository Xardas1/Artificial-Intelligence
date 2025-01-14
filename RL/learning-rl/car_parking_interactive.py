from yarl.environments import make_environment
from yarl.presentation.car_parking import run_interactive
import math,argparse
from math import radians

parser = argparse.ArgumentParser(description="gra asteroids")
parser.add_argument("--scale","-s",type=float, default="50", help="scaling environmetn to display")
parser.add_argument("--verbose","-v", action='store_true', help="print more informations")
parser.add_argument("--fullscreen","-f", action='store_true', help="go fullscreen")
Args = parser.parse_args()

env = make_environment('car-parking',car='small',env='perpendicular')
run_interactive(env,initial_state=(3,7,radians(-90),0,0),scale=Args.scale,fullscreen=Args.fullscreen,verbose=Args.verbose)
