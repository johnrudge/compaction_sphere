import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from scipy.integrate import odeint

def sphere_plot(ax, a):
    circ = plt.Circle((0.0,0.0), radius = a, facecolor='w',edgecolor = 'k', linewidth=1)
    ax.add_patch(circ)

def make_coord_grid(maxxy, n, ndim=3):
    x, y = np.meshgrid(np.linspace(-maxxy,maxxy,n),np.linspace(-maxxy,maxxy,n))
    xf, yf = x.ravel(), y.ravel()    
    if ndim==3:
        z = np.zeros_like(x)
        zf = z.ravel()
    if ndim==3:
        xyz = np.concatenate([xf[:,None],yf[:,None],zf[:,None]],axis = 1)
    else:
        xyz = np.concatenate([xf[:,None],yf[:,None]],axis = 1)
    return xyz

def scalar_plot(ax, a, field_function, maxxy, n, ndim=3):
    xyz = make_coord_grid(a*maxxy, n, ndim)
    interior = (np.sum(xyz*xyz,axis=1) < a*a)

    field = field_function(xyz)
    maxfield = max(field[~interior])
    minfield = min(field[~interior])
    
    x = xyz[:,0]
    y = xyz[:,1]
    x.shape = (n,n)
    y.shape = (n,n)
    field.shape = (n,n)
    im = plt.contourf(x,y,field,np.linspace(minfield,maxfield,24),cmap=plt.cm.seismic)
    sphere_plot(ax, a)
    ax.axes.xaxis.set_ticklabels([])
    ax.axes.yaxis.set_ticklabels([])
    return im

def vector_plot(ax, a, field_function, maxxy, n, ndim=3):
    xyz = make_coord_grid(a*maxxy,n, ndim)

    interior = (np.sum(xyz*xyz,axis=1) < a*a).nonzero()

    field = field_function(xyz)
    field = np.ma.array(field)
    field[interior,:] = np.ma.masked 
    
    plt.quiver(xyz[...,0], xyz[...,1], field[...,0], field[...,1],pivot='mid')
    plt.xlim((-maxxy*a,maxxy*a))
    plt.ylim((-maxxy*a,maxxy*a))
    sphere_plot(ax, a)
    ax.axes.xaxis.set_ticklabels([])
    ax.axes.yaxis.set_ticklabels([])

def streamtracer(a, vfun, gradvfun, maxxy, ndim=3):    
    def odefun(x,t):
        return vfun(x[None,:])

    def odebackwardfun(x,t):
        return -vfun(x[None,:])

    def Dfun(x,t):
        return gradvfun(x[None,:])
    
    def Dbackwardfun(x,t):
        return -gradvfun(x[None,:])
    
    def traceline(initial, tmax, n, linestyle):
        trange = np.linspace(0.0,tmax,n)
        y1 = odeint(odefun,np.array(initial),trange, Dfun=Dfun)
        y2 = odeint(odebackwardfun,np.array(initial),trange,Dfun=Dbackwardfun)

        plt.plot(y1[:,0],y1[:,1],linestyle)
        plt.plot(y2[:,0],y2[:,1],linestyle)

    def tracelines(initials, tmax, n, linestyle):
        for i in range(initials.shape[0]):
            traceline(initials[i,:], tmax, n, linestyle)

    yvals = a* np.linspace(-maxxy,maxxy,17)
    initials = np.zeros((len(yvals),ndim))
    initials[:,1] = yvals
    initials[:,0] = -maxxy*a
    tracelines(initials, 20*maxxy, 1000, 'k-')
    initials[:,0] = maxxy*a
    tracelines(initials, 20*maxxy, 1000, 'k-')
 
def phi_plot(ax, a, field_function, phi0, maxxy, n, ndim=3, colormax=0):
    xyz = make_coord_grid(a*maxxy,n, ndim)

    interior = (np.sum(xyz*xyz,axis=1) < 0.9*a*a)
    exterior = ~interior
    extvals = exterior.nonzero()[0]

    xyz_ext = xyz[extvals,:]

    phiint = field_function(xyz_ext)
    phiv = np.zeros_like(xyz[:,0].ravel())
    phiv[exterior] = phiint

    realexterior = (np.sum(xyz*xyz,axis=1) >= 0.99*a*a)
    maxphi = max(phiv[realexterior])
    minphi = min(phiv[realexterior])
    rangeplus = maxphi-phi0
    rangeminus = phi0 -minphi
    maxrange = max([rangeplus,rangeminus])

    phiv = np.ma.array(phiv)
    phiv[interior] = np.ma.masked 

    x = xyz[:,0]
    y = xyz[:,1]
    x.shape = (n,n)
    y.shape = (n,n)        
    phiv.shape = (n,n)
    
    if colormax==0:
        colorbar_ticks = np.linspace(phi0-maxrange,phi0+maxrange,24)
    else:
        colorbar_ticks = np.linspace(phi0-colormax,phi0+colormax,24)
        print("Colour max:", colormax, "Max range:", maxrange)
    
    im = plt.contourf(x,y,phiv,colorbar_ticks,cmap=plt.cm.seismic)
    sphere_plot(ax, a)

    plt.colorbar(use_gridspec=True)
    
    ax.axes.xaxis.set_ticklabels([])
    ax.axes.yaxis.set_ticklabels([])

    plt.tight_layout()
      
def plot_error_ellipse(mu, cov, ax=None, **kwargs):
    """
    Plot the error ellipse at a point given it's covariance matrix

    Parameters
    ----------
    mu : array (2,)
        The center of the ellipse

    cov : array (2,2)
        The covariance matrix for the point

    ax : matplotlib.Axes, optional
        The axis to overplot on

    **kwargs : dict
        These keywords are passed to matplotlib.patches.Ellipse

    """
    # some sane defaults
    facecolor = kwargs.pop('facecolor', 'none')
    edgecolor = kwargs.pop('edgecolor', 'k')

    x, y = mu
    U,S,V = np.linalg.svd(cov)
    theta = np.degrees(np.arctan2(U[1,0], U[0,0]))
    ellipsePlot = Ellipse(xy=[x, y],
            width = 2*np.sqrt(S[0]),
            height= 2*np.sqrt(S[1]),
            angle=theta,
            facecolor=facecolor, edgecolor=edgecolor, **kwargs)

    if ax is None:
        ax = plt.gca()
    ax.add_patch(ellipsePlot)
    
def strain_marker_plot(ax, a, flow, maxxy, n, scalefactor=0.05, ndim=3):
    sphere_plot(ax, a)
    
    xyz = make_coord_grid(a*maxxy,n, ndim)

    interior = (np.sum(xyz*xyz,axis=1) < 0.9*a*a)
    exterior = ~interior
    extvals = exterior.nonzero()[0]

    xyz_ext = xyz[extvals,:]

    Gint = flow.strain_marker(xyz_ext)
    Gv = np.zeros((xyz.shape[0],ndim,ndim))
    Gv[exterior] = Gint


    Gv = np.ma.array(Gv)
    Gv[interior] = np.ma.masked 

    x = xyz[:,0]
    y = xyz[:,1]

    
    for i in range(xyz_ext.shape[0]):
       B = flow.strain_marker(xyz_ext[i,:])
       B2 = B[0:2,0:2]
       x2 = xyz_ext[i, 0:2]
       plot_error_ellipse(x2,((a*scalefactor)**2)*B2) 
    
    plt.xlim((min(xyz[:,0]),max(xyz[:,0])))
    plt.ylim((min(xyz[:,1]),max(xyz[:,1])))
    ax.set_aspect('equal')
    
    ax.axes.xaxis.set_ticklabels([])
    ax.axes.yaxis.set_ticklabels([])

    plt.tight_layout()