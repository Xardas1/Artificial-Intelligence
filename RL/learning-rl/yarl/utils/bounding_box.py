from dataclasses import dataclass
from .vector import Vector2
from copy import deepcopy
import numpy as np

def bbox_range(bbox,coord):  return bbox[0][coord],bbox[1][coord]
def bbox_xrange(bbox):  return bbox_range(bbox,0)
def bbox_yrange(bbox):  return bbox_range(bbox,1)

@dataclass
class Bbox:
    x_min : float
    y_min : float
    x_max : float
    y_max : float

    @classmethod
    def make(cls,other):
        if type(other) is list:
            if type(other[0]) is list:
                    #assumig [[left,top], [right,bottom]]
                    xr,yr = bbox_xrange(other), bbox_yrange(other)
                    return Bbox(xr[0],yr[0],xr[1],yr[1])
            else:
                #assumig [x_min,y_min,x_max,y_max]
                return Bbox(*other)
    
    def xrange(self):
        return self.x_min,self.x_max

    def width(self):
        return self.x_max - self.x_min
    
    def yrange(self):
        return self.y_min,self.y_max
    
    def height(self):
        return self.y_max - self.y_min

    def overlap(self, other):
        l,r = self,other if self.x_min < other.x_min else other,self
        if l.x_max > r.x_min: return False
        b,t = self,other if self.y_min < other.y_min else other,self
        if b.y_max > t.x_min: return False
        return True
    
    def enclose(self, other):
        return self.x_min <= other.x_min and self.y_max >= other.x_max\
            and self.y_min <= other.y_min and self.y_max >= other.y_max

class RBbox:
    def __init__(self, corners : list[Vector2]):
        self.corners = np.array(corners)

    def overlap(self, other):
        # Convert the rectangles into numpy arrays
        rect1 = self.corners
        rect2 = other.corners
    
        # Helper function to project rectangle onto an axis
        def project_rectangle(rect, axis):
            projections = rect @ axis  # Dot product of all corners with the axis
            return np.min(projections), np.max(projections)

        # Helper function to check if two projections overlap
        def projections_overlap(proj1, proj2):
            return not (proj1[1] < proj2[0] or proj2[1] < proj1[0])

        # Compute edges of the rectangles
        edges = np.vstack([
            rect1[1] - rect1[0],  # Edge 1 of rect1
            rect1[3] - rect1[0],  # Edge 2 of rect1
            rect2[1] - rect2[0],  # Edge 1 of rect2
            rect2[3] - rect2[0],  # Edge 2 of rect2
        ])

        # Compute perpendicular axes
        axes = np.array([[-edge[1], edge[0]] for edge in edges])

        # Check for overlap on each axis
        for axis in axes:
            axis = axis / np.linalg.norm(axis)  # Normalize the axis
            proj1 = project_rectangle(rect1, axis)
            proj2 = project_rectangle(rect2, axis)
            if not projections_overlap(proj1, proj2):
                return False  # No overlap on this axis

        return True  # Overlaps on all axes
    
    def Bbox(self):
         #self.corners = np.array(corners)
         x = self.corners[:,0]
         y = self.corners[:,1]
         return Bbox(min(x),min(y),max(x),max(y))
 