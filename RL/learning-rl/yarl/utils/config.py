import json
def readConfig(filename):
    class Config:
        pass
    def objectify(dict_):
        cfg = Config()
        for k,v in dict_.items():
            if type(v) is dict:
                setattr(cfg,k,objectify(v))
            else:
                setattr(cfg,k,v)    
        return cfg
    with open(filename) as jsonFile:
        data = json.load(jsonFile)
    return objectify(data)
