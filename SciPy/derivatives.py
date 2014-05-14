import numpy as np
import matplotlib.pyplot as plt
def twoPtForwardDiff(x,y):
    '''Takes an array of x and an array of y and differentiates y with respect to x, 
    using forward difference. The last point is manually assigned a derivative.
    '''
    dydx = np.zeros(y.shape,float)
    dydx[0:-1] = np.diff(y)/np.diff(x)
    dydx[-1] = (y[-1] - y[-2])/(x[-1] - x[-2])
    
    return dydx

def twoPtCenteredDiff(x,y):
    '''Takes an array of x and and array of y and differentiates y with respect to x, 
        using Centered Difference. This means that it takes into account the point on the left and on the right
        of any given point, then computes the derivative. The First and last points are manually given 
        derivatives
    '''
    dydx = np.zeros(y.shape,float)
    dydx[1:-1] = (y[2:] - y[:-2]) /(x[2:] - x[:-2])
    dydx[0] = (y[1] - y[0])/(x[1]-x[0])
    dydx[-1] = (y[-1]-y[-2])/(x[-1]-x[-2])
    return dydx

def fourPtCenteredDiff(x,y):
    '''
    This is the FourPt Centered Differentiation Method. It adjusts for the fact that the four points are truncated at
    beginning and the end of the series. It starts at x=2, then computes the derivative using two points to the left
    two points to the right, continuing until x = -3
    '''
    dydx = np.zeros(y.shape,float)
    h = x[1]-x[0]
    dydx[2:-2] = (y[:-4] - 8*y[1:-3] + 8*y[3:-1] -y[4:]) / (12*h)
    dydx[0] = (y[1]-y[0])/(x[1]-x[0])
    dydx[1] = (y[2]-y[1])/(x[2]-x[1])
    dydx[-1] = (y[-1]-y[-2])/(x[-1]-y[-2])
    dydx[-2] = (y[-2]-y[-3])/(x[-2]-x[-3])
    return dydx

    