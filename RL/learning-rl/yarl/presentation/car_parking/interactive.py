import pygame
import numpy as np
from pygame.gfxdraw import filled_polygon,filled_circle
from ..pygame import got_quit_event,display_text
from datetime import datetime
from ...utils import Vector2,rotMove_radians
from math import pi,radians

def create_env_polygons(env_config, screen_size, scale):
    env_polygons = []
    for rect in env_config.obstacles:
        pts = np.array(rect)*scale
        env_polygons.append(pts)
    return env_polygons

def draw_env(screen,env_polygons,color):
    for poly in env_polygons:
        filled_polygon(screen, poly, color)

def draw_car(screen,car_state,geometry,color,scale,screen_size):
    scaled_pos = Vector2(car_state[0],car_state[1]) * scale
    pts = rotMove_radians(car_state[2], scaled_pos, geometry)
    filled_polygon(screen, pts, color)
    c = scaled_pos.asInt()
    filled_circle(screen,c.x,c.y,2,(255,255,255))

def display_info(screen,font,car_state,verbose):
    if not verbose: return
    text = [f'steering angle : {car_state[4]:.1f}',
            f'orientation    : {car_state[2]:.1f}']
    display_text(screen,font,(255,255,255),(0,0,0),text,(10,10))

def get_actions(events,car_config):
    keys = pygame.key.get_pressed()
    if keys[pygame.K_UP]:
        acceleration = -car_config.speed_increment
    elif keys[pygame.K_DOWN]:
        acceleration = car_config.speed_increment
    else:
        acceleration = 0

    sai = car_config.steering_angle_increment
    if keys[pygame.K_LEFT]:
        turn = sai
    elif keys[pygame.K_RIGHT]:
        turn = -sai
    else:
        turn = 0

    do_print = False
    for e in events:
        if e.type == pygame.KEYDOWN:
            if e.unicode == 'p':
                do_print = True
                break
        
    return acceleration,turn,do_print

def run_interactive(env, initial_state=None, scale=50, fullscreen=False,verbose=False):
    print('starting interactive car-parking simulation')
    car_config = env.get_info().car
    env_config = env.get_info().environment
    pygame.init()
    ew,eh = env_config.bounding_box.width(), env_config.bounding_box.height()
    screen_size = (ew*scale, eh*scale)
    if verbose:
        print(f'environment width x height [ {ew} x {eh} ] screen [{screen_size[0]} x {screen_size[1]}]')
    
    screen = pygame.display.set_mode(screen_size)
    if fullscreen:
        pygame.display.toggle_fullscreen()

    background_color=0,0,0
    car_color = 72,54,107
    env_color = 180,180,180
    w2,l2 = car_config.width * scale // 2, car_config.length * scale // 2
    if verbose:
        print(f'car size [width x height] [ {2*w2} {2*l2} ]')
    car_geometry = np.array([(-w2,-l2), (-w2,l2),(w2,l2),(w2,-l2)])
    env_polygons = create_env_polygons(env_config,screen_size,scale)
    font = pygame.font.SysFont("Arial", 15)
    
    car_state = list(initial_state) if initial_state is not None else env.random_state()
        
    t0 = t1 = datetime.now()
    while True:
        events = pygame.event.get()
        if got_quit_event(events):
            break
        screen.fill(background_color)
        actions = get_actions(events,car_config)
        car_state,end_state,_ = env.step(car_state, actions, (t1-t0).total_seconds())
        draw_env(screen,env_polygons,env_color)
        draw_car(screen,car_state,car_geometry,car_color if not end_state else (255,100,100),scale,screen_size)
        display_info(screen,font,car_state,verbose)
        pygame.display.flip()
        t0 = t1
        t1 = datetime.now()
    pygame.quit()