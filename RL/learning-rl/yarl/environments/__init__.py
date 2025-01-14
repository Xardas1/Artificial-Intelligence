from .car_parking import car_parking_env
    
def make_environment(name,**kwargs):
    match name:
        case 'car-parking':
            return car_parking_env(**kwargs)
        case _:
            raise NotImplementedError

