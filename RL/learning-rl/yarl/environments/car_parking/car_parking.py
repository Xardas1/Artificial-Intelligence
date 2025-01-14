from ..environment import environment
from ...utils import readConfig,clamp,Bbox,RBbox,rotMove_radians,Vector2

from math import pi,sin,cos,radians,tan
from pathlib import Path

class EnvInfo:
    def __init__(self, **kwargs):
        for k,v in kwargs.items():
            setattr(self,k,v)
        
        sai = self.car.steering_angle_increment = radians(self.car.steering_angle_increment)
        max_sa = self.car.max_steering_angle = radians(self.car.max_steering_angle)
        spi = self.car.speed_increment
        self.action_space = [
            ('accelerate' , (-spi,0,+spi)),
            ('turn' ,       (-sai,0,sai))
        ]
        self.environment.bounding_box = ebb = Bbox.make(self.environment.bounding_box)
        self.environment.obstacles_rbb = [RBbox(c) for c in self.environment.obstacles]
        
        self.state_space = [
            ('position_x'    , ebb.xrange()),
            ('position_y'    , ebb.yrange()),
            ('orientation'   , (0.,2*pi)),
            ('speed'         , (-self.car.max_speed,self.car.max_speed)),
            ('steering_angle', (-max_sa,max_sa))
        ]
        self.car.wheel_base = self.car.length - self.car.front_wheel_mount_point + self.car.rear_wheel_mount_point

class car_parking_env(environment):
    def __init__(self,**kwargs):
        folder = Path(__file__).resolve().parent
        car_name = kwargs.get('car','small')
        car_cfg=readConfig(folder / 'cars' / f'{car_name}.json')
        env_name = kwargs.get('env','perpendicular')
        env_cfg=readConfig(folder / 'env' / f'{env_name}.json')
        self.info = EnvInfo(car=car_cfg.car,\
                            environment=env_cfg.environment,\
                            target_state = env_cfg.target_state)
        self.info.pos_reward = kwargs.get("positive_reward", 10)
        self.info.neg_reward = kwargs.get("negative_reward",-10)

    def get_info(self):
        return self.info

    def random_state(self):
        raise NotImplementedError
    
    def _make_car_rbbox(self,state,c,s):
        w2,l2 = self.info.car.width / 2, self.info.car.length / 2
        return RBbox(rotMove_radians(state[2],[state[0],state[1]],[(-w2,-l2), (-w2,l2),(w2,l2),(w2,-l2)]))
    
    def step(self, state, action, dt=1.0):
        # state[0] : position_x
        # state[1] : position_y
        # state[2] : orientation
        # state[3] : speed
        # state[4] : steering_angle
        # action[0] : acceleration
        # action[1] : turn
        max_speed = self.info.car.max_speed
        max_steering_angle = self.info.car.max_steering_angle
        snew = [0.0] * 5
        accel = action[0] * dt
        turn = action[1] * dt
        snew[3] = clamp(state[3] + accel, -max_speed,max_speed)
        snew[4] = clamp(state[4] + turn, -max_steering_angle,max_steering_angle)
        c = cos(state[2])
        s = sin(state[2])
        speed = snew[3] * dt
        snew[0] = state[0] + speed * s
        snew[1] = state[1] - speed * c
        snew[2] = state[2] + speed * tan(snew[4]) / self.info.car.wheel_base
   
        car_rbbox = self._make_car_rbbox(snew,c,s)
        car_bbox = car_rbbox.Bbox()
        if not self.info.environment.bounding_box.enclose(car_bbox):
            return snew,True,self.info.neg_reward
        
        collision = False
        for o in self.info.environment.obstacles_rbb:
            if car_rbbox.overlap(o):
                collision = True
        if collision:
            return snew, True, self.info.neg_reward
        
        # compare with target state
        ts = self.info.target_state
        p1 = Vector2(snew[0],snew[1])
        p2 = Vector2(*ts.position)
        
        position_error = p1.dist(p2)
        orient_error = min( [radians(x) - abs(snew[2]) for x in ts.orientation] )
        assert( orient_error <= pi/2 )
        speed_error = abs(snew[3] - ts.speed)
        speed_error_r = speed_error / max_speed

        orient_error_r = orient_error / (pi/2)
        position_error_max = max(self.info.environment.bounding_box.width(), self.info.environment.bounding_box.height())
        position_error_r = position_error / position_error_max

        if speed_error_r <= ts.tolerance and orient_error_r <= ts.tolerance and position_error_r <= ts.tolerance:
            return snew, True, self.info.pos_reward
        
        return snew, False, None


        

            
        

