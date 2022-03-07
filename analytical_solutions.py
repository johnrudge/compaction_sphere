# Analytical solutions for compaction around a sphere
import numpy as np
from scipy.integrate import odeint

# Levi-Civita symbol (is this not built-in to numpy anywhere?)
eijk = np.zeros((3, 3, 3))
eijk[0, 1, 2] = eijk[1, 2, 0] = eijk[2, 0, 1] = 1
eijk[0, 2, 1] = eijk[2, 1, 0] = eijk[1, 0, 2] = -1

# Modified spherical Bessel functions of the second kind
def k0(r):
    return np.exp(-r)/r

def k1(r):
    return (np.exp(-r)/(r**2))*(r+1)

def k2(r):
    return (np.exp(-r)/(r**3))*((r**2)+3*r+3)

def k3(r):
    return (np.exp(-r)/(r**4))*((r**3)+6*(r**2)+15*r+15)

def k4(r):
    return (np.exp(-r)/(r**5))*((r**4)+10*(r**3)+45*(r**2)+105*r+105)

def k0p(r):
    return -(np.exp(-r)/(r**2))*(r+1)

def k1p(r):
    return -(np.exp(-r)/(r**3))*((r**2)+2*r+2)

def k2p(r):
    return -(np.exp(-r)/(r**4))*((r**3)+4*(r**2)+9*r+9)

class Flow:
    """ A general flow class providing some common methods """
    def grad_velocity(self,x):
        e = self.strain_tensor(x)
        vort = self.vorticity_tensor(x)
        grad_v = e + vort
        return grad_v

    def zero_scalar(self,x):
        x = self.coord_reshape(x)
        z = np.zeros(x.shape[0])
        return np.squeeze(z)
    
    def zero_vector(self,x):
        x = self.coord_reshape(x)
        z = np.zeros((x.shape[0],x.shape[1]))
        return np.squeeze(z)

    def zero_tensor(self,x):
        x = self.coord_reshape(x)
        z = np.zeros((x.shape[0],x.shape[1],x.shape[1]))
        return np.squeeze(z)
    
    def porosity(self, x):
        def single_coord_calc(x):
            ndim = len(x)
            def odefun(z,t):
                x = z[None,:-1]
                v = self.velocity(x).ravel()
                C = self.compaction_rate(x).ravel()
                return np.concatenate([-v, C])

            def Dfun(z,t):
                x = z[None,:-1]
                dvdx = self.grad_velocity(x)
                dCdx = self.grad_compaction(x)
                jac = np.zeros((ndim+1,ndim+1))
                jac[0:ndim,0:ndim] = -dvdx
                jac[ndim,0:ndim] = dCdx
                return jac
                    
            initial = np.concatenate([x, [0.0]])
            trange = np.linspace(0.0, self.t, 2)
            y = odeint(odefun,initial,trange,Dfun=Dfun)
            H = y[-1,-1]
            ph = 1.0 - (1.0-self.phi0)*np.exp(-H)
            return ph
        if x.ndim==1:
            return single_coord_calc(x)
        else:
            return np.array([single_coord_calc(x[i,:]) for i in range(x.shape[0])])
        
    def strain_marker(self,x):
        """ The left Cauchy-Green tensor B = F F^T """
        def single_coord_calc(x):
            ndim = len(x)
            
            def v_backward_odefun(x,t):
                return -self.velocity(x).ravel()

            def odefun(z,t):
                x = z[None,0:ndim]
                F = z[None,ndim:]
                F= F.reshape((ndim,ndim))
                v = self.velocity(x).ravel()
                dvdx = self.grad_velocity(x)
                
                dxdt = v
                dFdt = np.dot(dvdx, F).ravel()
                
                return np.concatenate([dxdt, dFdt])
                        
            # integrate backwards in time to find initial point
            trange = np.linspace(0.0, self.t, 2)
            y = odeint(v_backward_odefun, x, trange)
            
            # initial conditions for integration
            x0 = y[-1,:]
            F0 = np.eye(ndim).ravel()
            initial = np.concatenate([x0, F0])
            
            # do main integration for deformation tensor
            y = odeint(odefun,initial,trange)

            F = y[-1,ndim:]
            F = F.reshape((ndim,ndim))
            
            B = np.dot(F,np.transpose(F))
            
            return B
        
        if x.ndim==1:
            return single_coord_calc(x)
        else:
            return np.array([single_coord_calc(x[i,:]) for i in range(x.shape[0])])  
    
    def coord_reshape(self, x):
        if x.ndim==1:
            return x[None,:]
        else:
            return x

    def vorticity_tensor(self,x):
        # Can calculate vorticity tensor from vorticity vector
        vort = self.coord_reshape(self.vorticity(x))
        vort_tensor = -0.5*np.einsum('kij,sk->sij',eijk,vort)
        return np.squeeze(vort_tensor)
    
    def strain_invariant(self,x):
        x = self.coord_reshape(x)
        e = self.strain_tensor(x)
        I = np.tile(np.eye(3),(x.shape[0],1,1))
        c = self.compaction_rate(x)
            
        cI = np.einsum('ijk,i ->ijk', I, c)
        eprime = e - (1.0/3.0)*cI
        
        invariant = np.sqrt(0.5*np.einsum('ijk,ijk->i', eprime, eprime))
        return np.squeeze(invariant)


class Translation(Flow):
    """ Solution for the translating sphere """
    def __init__(self, V, a, B):
        self.ndim = 3
        self.V = V
        self.a = a
        self.B = B
        denom = B*k0(a)-a*a*k1p(a)
        
        self.C = - 3.0 * (a**3)* k1p(a) / (4.0*denom)
        self.D = (a**3) *(4.0* B * k2(a) - a*a*k1p(a))/(4.0*denom)
        self.F = - 3.0*B / denom 

    def common(self,x):
        x = self.coord_reshape(x)
        r = np.sqrt(np.sum(x*x,axis=-1))
        Vdotx = np.sum(self.V[None,:]*x,axis=-1)
        return x, r, Vdotx

    def common_tensor(self,x):
        x = self.coord_reshape(x)
        I = np.tile(np.eye(3),(x.shape[0],1,1))
        x_x = np.einsum('ij,ik->ijk',x,x)
        x_V = np.einsum('ij,k',x,self.V)
        V_x = np.einsum('ik,j',x,self.V)
        return I, x_x, x_V, V_x

    def compaction_rate(self, x):
        x, r, Vdotx = self.common(x)
        comp = - self.F *k1(r) * Vdotx / r
        return np.squeeze(comp)

    def velocity(self, x):
        x, r, Vdotx = self.common(x)
        coeff_V = -(self.C/r) - (self.D/(r**3)) - (self.F * k1(r)/r)
        coeff_x = ((-self.C/(r**3)) + (3.0*self.D/(r**5)) + (self.F *k2(r)/(r**2)))*Vdotx
        
        v = coeff_V[...,None] * self.V[None,...] + coeff_x[...,None] * x
        return np.squeeze(v)
         
    def pressure(self, x):
        x, r, Vdotx = self.common(x)
        pressure = (-(2*self.B*self.C/(r**3))-self.F*k1(r)/r)*Vdotx
        return np.squeeze(pressure)
    
    def darcy_flux(self,x):
        x, r, Vdotx = self.common(x)
        coeff_V = 2*(self.B*self.C/(r**3))+ (self.F * k1(r)/r)
        coeff_x = ((-6*self.B*self.C/(r**5)) - (self.F *k2(r)/(r**2)))*Vdotx
        
        q = coeff_V[...,None] * self.V[None,...] + coeff_x[...,None] * x
        return np.squeeze(q)

    def grad_compaction(self,x):
        x, r, Vdotx = self.common(x)
        coeff_V = (-self.F * k1(r)/r)
        coeff_x = ((self.F *k2(r)/(r**2)))*Vdotx
        
        grad_c = coeff_V[...,None] * self.V[None,...] + coeff_x[...,None] * x
        return np.squeeze(grad_c)

    def strain_tensor(self,x):
        x, r, Vdotx = self.common(x)
        I, x_x, x_V, V_x = self.common_tensor(x)
        coeff_I = (-self.C/(r**3) + 3*self.D/(r**5) + self.F*k2(r)/(r**2))*Vdotx
        coeff_VxpxV = (3*self.D/(r**5) + self.F*k2(r)/(r**2))
        coeff_xx = (3*self.C/(r**5) - 15 *self.D/(r**7) - self.F *k3(r)/(r**3))*Vdotx
        e = coeff_I[...,None,None] * I + coeff_VxpxV[...,None,None] * (V_x + x_V) + coeff_xx[...,None,None]*x_x
        return np.squeeze(e)

    def vorticity(self,x):
        x, r, Vdotx = self.common(x)
        coeff = 2.0*self.C/(r**3)
        vort = coeff[...,None]*np.cross(x,self.V[None,:])
        return np.squeeze(vort)

    def stress_tensor(self,x):
        x, r, Vdotx = self.common(x)
        I, x_x, x_V, V_x = self.common_tensor(x)
        coeff_I = (self.C/(r**3) + self.F*k1(r)/(r))*Vdotx
        sig = 2.0*self.B*(coeff_I[...,None,None] * I + self.strain_tensor(x))
        return np.squeeze(sig)
        
class Strain(Flow):
    """ Solution for the strained sphere """
    def __init__(self, E, a, B):
        self.ndim = 3
        self.E = E
        self.a = a
        self.B = B
        denom = 6.0*B*k1(a)-a*a*k2p(a)
        
        self.C = 5.0 * (a**5)* k2p(a) / (6.0*denom)
        self.D = -((a**5)/6.0) - (5.0*(a**4)*B*k2(a)/denom)
        self.F = 15.0*a*B / denom 

    def common(self,x):
        x = self.coord_reshape(x)
        r = np.sqrt(np.sum(x*x,axis=-1))
        Ex = np.tensordot(x,self.E,([-1],[0]))
        xEx = np.sum(x*Ex,axis=-1)
        return x, r, Ex, xEx

    def common_tensor(self,x):
        x, r, Ex, xEx = self.common(x)
        I = np.tile(np.eye(3),(x.shape[0],1,1))
        x_x = np.einsum('ij,ik->ijk',x,x)
        x_Ex = np.einsum('ij,ik->ijk',x,Ex)
        Ex_x = np.einsum('ik,ij->ijk',x,Ex)
        return I, x_x, x_Ex, Ex_x

    def compaction_rate(self, x):
        x, r, Ex, xEx = self.common(x)
        comp = self.F *k2(r) * xEx / (r**2)
        return np.squeeze(comp)

    def pressure(self, x):
        x, r, Ex, xEx = self.common(x)
        pressure = ((6.0*self.B*self.C/(r**5))+(self.F *k2(r)/(r**2))) * xEx 
        return np.squeeze(pressure)

    def velocity(self, x):
        x, r, Ex, xEx = self.common(x)
        
        coeff_Ex = (6.0*self.D/(r**5)) + 2.0* (self.F * k2(r)/(r*r))
        coeff_x = ((3.0*self.C/(r**5)) - (15.0*self.D/(r**7)) - (self.F *k3(r)/(r**3)))*xEx
        
        v = coeff_Ex[...,None] * Ex + coeff_x[...,None] * x
        return np.squeeze(v)
    
    def darcy_flux(self, x):
        x, r, Ex, xEx = self.common(x)
        
        coeff_Ex = (-12.0*self.B*self.C/(r**5)) - 2.0* (self.F * k2(r)/(r*r))
        coeff_x = ((30.0*self.B*self.C/(r**7)) + (self.F *k3(r)/(r**3)))*xEx
        
        q = coeff_Ex[...,None] * Ex + coeff_x[...,None] * x
        return np.squeeze(q)

    def grad_compaction(self, x):
        x, r, Ex, xEx = self.common(x)
        
        coeff_Ex =  2.0* (self.F * k2(r)/(r*r))
        coeff_x = -(self.F *k3(r)/(r**3))*xEx
        
        grad_c = coeff_Ex[...,None] * Ex + coeff_x[...,None] * x
        return np.squeeze(grad_c)
    
    def strain_tensor(self,x):
        x, r, Ex, xEx = self.common(x)
        I, x_x, x_Ex, Ex_x = self.common_tensor(x)
        coeff_E = (6*self.D/(r**5) + 2*k2(r)*self.F/(r**2))
        coeff_I = (3*self.C/(r**5) - 15*self.D/(r**7) - self.F*k3(r)/(r**3))*xEx
        coeff_ExpxE = (3*self.C/(r**5)-30*self.D/(r**7) - 2*self.F*k3(r)/(r**3))
        coeff_xx = (-15*self.C/(r**7) + 105 *self.D/(r**9) + self.F *k4(r)/(r**4))*xEx
        e = coeff_E[...,None,None]*self.E[None,...] + coeff_I[...,None,None] * I + coeff_ExpxE[...,None,None] * (Ex_x + x_Ex) + coeff_xx[...,None,None]*x_x
        return np.squeeze(e)

    def vorticity(self,x):
        x, r, Ex, xEx = self.common(x)
        coeff = -6.0*self.C/(r**5)
        vort = coeff[...,None]*np.cross(x,Ex)
        return np.squeeze(vort)   

    def stress_tensor(self,x):
        x, r, Ex, xEx = self.common(x)
        I, x_x, x_Ex, Ex_x = self.common_tensor(x)
        coeff_I = -(3*self.C/(r**5) + self.F*k2(r)/(r**2))*xEx
        sig = 2.0 * self.B *(coeff_I[...,None,None] * I + self.strain_tensor(x))
        return np.squeeze(sig)


class Torsion(Flow):
    """ Solution for the torsioned sphere """
    def __init__(self, TH, a, B):
        self.ndim = 3
        self.TH = TH
        self.a = a
        self.B = B

    def common(self,x):
        x = self.coord_reshape(x)
        r = np.sqrt(np.sum(x*x,axis=-1))
        THx = np.tensordot(x,self.TH,([-1],[0]))
        xTHx = np.sum(x*THx,axis=-1)
        return x, r, THx, xTHx

    def compaction_rate(self, x):
        return self.zero_scalar(x)

    def pressure(self, x):
        return self.zero_scalar(x)

    def velocity(self, x):
        x, r, THx, xTHx = self.common(x)
        
        coeff = (1.0/3.0)*((self.a/r)**5)
        x_cross_THx = np.cross(x,THx)
        
        v = coeff[...,None]*x_cross_THx
        return np.squeeze(v)
    
    def darcy_flux(self, x):
        return self.zero_vector(x)

    def grad_compaction(self, x):
        return self.zero_vector(x)
    
    def vorticity(self,x):
        x, r, THx, xTHx = self.common(x)
        coeff_x = -((5.0 * (self.a**5))/(3.0*(r**7)))*xTHx
        coeff_THx =  (2.0* (self.a**5))/(3.0*(r**5))
        
        vort = coeff_x[...,None] * x + coeff_THx[...,None] * THx
        return np.squeeze(vort)
    
    def strain_tensor(self,x):
        x, r, THx, xTHx = self.common(x)
        t1a = np.einsum('sk,ikl,lj->sij',x,eijk,self.TH)
        t1b = np.einsum('sk,jkl,li->sij',x,eijk,self.TH)
        t2a = np.einsum('sk,sl,ikl,sj->sij',x,THx,eijk,x)
        t2b = np.einsum('sk,sl,jkl,si->sij',x,THx,eijk,x)
        coeff_1 = (self.a**5)/(6.0*(r**5))
        coeff_2 = -(5.0*(self.a**5))/(6.0*(r**7))
        strain = coeff_1[...,None,None]*(t1a+t1b) + coeff_2[...,None,None]*(t2a+t2b)
        return np.squeeze(strain)
        

class FarField(Flow):
    """ The far field flow (linear)"""
    def __init__(self, Vinf, gradVinf, B):
        self.ndim = 3
        self.B = B
        self.Vinf = Vinf
        self.gradVinf = gradVinf
        self.E = 0.5 * (gradVinf + gradVinf.T)
        self.O = 0.5 * (gradVinf - gradVinf.T)
        self.Omega = 0.5*np.einsum('kij,ji->k',eijk,gradVinf)

    def velocity(self, x):
        if self.Vinf is not None:
            v = self.Vinf[None,:] + np.tensordot(x,self.gradVinf,([-1],[-1]))
        else:
            v = np.tensordot(x,self.gradVinf,([-1],[-1]))
        return np.squeeze(v)
    
    def compaction_rate(self, x):
        return self.zero_scalar(x)
    
    def pressure(self, x):
        return self.zero_scalar(x)
    
    def darcy_flux(self, x):
        return self.zero_vector(x)
    
    def grad_compaction(self,x):
        return self.zero_vector(x)
        
    def strain_tensor(self,x):
        x = self.coord_reshape(x)
        e = np.tile(self.E,(x.shape[0],1,1))
        return np.squeeze(e)

    def vorticity(self,x):
        x = self.coord_reshape(x)
        vort = np.tile(2.0*self.Omega,(x.shape[0],1,1))
        return np.squeeze(vort)

    def stress_tensor(self,x):
        return 2.0*self.B*self.strain_tensor(x)


class FarFieldQ(Flow):
    """ The far field flow (quadratic) """
    def __init__(self, THinf, B):
        self.ndim = 3
        self.B = B
        self.THinf = THinf

    def velocity(self, x):
        x = self.coord_reshape(x)
        THx = np.tensordot(x,self.THinf,([-1],[0]))
        x_cross_THx = np.cross(x,THx)
        v = -(1.0/3.0) * x_cross_THx
        return np.squeeze(v)
    
    def compaction_rate(self, x):
        return self.zero_scalar(x)
    
    def pressure(self, x):
        return self.zero_scalar(x)
    
    def darcy_flux(self, x):
        return self.zero_vector(x)
    
    def grad_compaction(self,x):
        return self.zero_vector(x)
    
    def vorticity(self,x):
        x = self.coord_reshape(x)
        THx = np.tensordot(x,self.THinf,([-1],[0]))
        vort = THx
        return np.squeeze(vort)

    def strain_tensor(self,x):
        x = self.coord_reshape(x)
        ea = -(1.0/6.0)*np.einsum('sk,kli,lj->sij',x,eijk,self.THinf)
        eb = -(1.0/6.0)*np.einsum('sk,klj,li->sij',x,eijk,self.THinf)
        return np.squeeze(ea+eb)

    def stress_tensor(self,x):
        return 2.0*self.B*self.strain_tensor(x)

    
class LinearFlow(Flow):
    """ Linear flow past a freely rotating sphere """
    def __init__(self, Vinf, gradVinf, a, B):
        self.ndim = 3
        E = 0.5 * (gradVinf + gradVinf.T)
        self.trans = Translation(Vinf, a, B)
        self.strain = Strain(E, a, B)
        self.far = FarField(Vinf, gradVinf, B)

    def __getattr__(self, name):
        """ All methods are done by linear superposition """
        def method(*args):
            tmethod = getattr(self.trans,name)
            smethod = getattr(self.strain,name)
            fmethod = getattr(self.far,name)
            return tmethod(*args) + smethod(*args) + fmethod(*args)
        return method


class QuadraticFlow(Flow):
    """ Quadratic flow past a freely rotating and translating sphere """
    def __init__(self, Vinf, gradVinf, THinf, a, B):
        self.ndim = 3
        E = 0.5 * (gradVinf + gradVinf.T)
        self.strain = Strain(E, a, B)
        self.torsion = Torsion(THinf, a, B)
        self.farL = FarField(Vinf, gradVinf, B)
        self.farQ = FarFieldQ(THinf, B)

    def __getattr__(self, name):
        """ All methods are done by linear superposition """
        def method(*args):
            smethod = getattr(self.strain,name)
            tormethod = getattr(self.torsion,name)
            fLmethod = getattr(self.farL,name)
            fQmethod = getattr(self.farQ,name)
            return smethod(*args) + fLmethod(*args) + fQmethod(*args)+ tormethod(*args)
        return method

# 2D Solutions (Cylinder)
from scipy.special import kn as Kn

class Strain2D(Flow):
    """ Solution for the strained cylinder """
    def __init__(self, E, a, B):
        self.E = E
        self.a = a
        self.B = B
        self.ndim = 2
        Knp2a = -0.5 * (Kn(1,a)+Kn(3,a))  # K'_2(a)
        denom = 4.0*B*Kn(1,a)-a*a*Knp2a
        
        self.C = -(a**4)*Knp2a/denom
        self.D = ((a**4)/4.0) + (4.0*(a**3)*B*Kn(2,a)/denom)
        self.F = 8.0*a*B / denom 

    def common(self,x):
        x = self.coord_reshape(x)
        r = np.sqrt(np.sum(x*x,axis=-1))
        Ex = np.tensordot(x,self.E,([-1],[0]))
        xEx = np.sum(x*Ex,axis=-1)
        return x, r, Ex, xEx

    def common_tensor(self,x):
        x, r, Ex, xEx = self.common(x)
        I = np.tile(np.eye(2),(x.shape[0],1,1))
        x_x = np.einsum('ij,ik->ijk',x,x)
        x_Ex = np.einsum('ij,ik->ijk',x,Ex)
        Ex_x = np.einsum('ik,ij->ijk',x,Ex)
        return I, x_x, x_Ex, Ex_x

    def compaction_rate(self, x):
        x, r, Ex, xEx = self.common(x)
        comp = self.F *Kn(2,r) * xEx / (r**2)
        return np.squeeze(comp)

    def pressure(self, x):
        x, r, Ex, xEx = self.common(x)
        pressure = ((-4.0*self.B*self.C/(r**4))+(self.F *Kn(2,r)/(r**2))) * xEx 
        return np.squeeze(pressure)

    def velocity(self, x):
        x, r, Ex, xEx = self.common(x)
        
        coeff_Ex = (-4.0*self.D/(r**4)) + 2.0* (self.F * Kn(2,r)/(r**2))
        coeff_x = (-(2.0*self.C/(r**4)) + (8.0*self.D/(r**6)) - (self.F *Kn(3,r)/(r**3)))*xEx
        
        v = coeff_Ex[...,None] * Ex + coeff_x[...,None] * x
        return np.squeeze(v)
    
    def darcy_flux(self, x):
        x, r, Ex, xEx = self.common(x)
        
        coeff_Ex = (8.0*self.B*self.C/(r**4)) - 2.0* (self.F * Kn(2,r)/(r*r))
        coeff_x = (-(16.0*self.B*self.C/(r**6)) + (self.F *Kn(3,r)/(r**3)))*xEx
        
        q = coeff_Ex[...,None] * Ex + coeff_x[...,None] * x
        return np.squeeze(q)

    def grad_compaction(self, x):
        x, r, Ex, xEx = self.common(x)
        
        coeff_Ex =  2.0* (self.F * Kn(2,r)/(r*r))
        coeff_x = -(self.F *Kn(3,r)/(r**3))*xEx
        
        grad_c = coeff_Ex[...,None] * Ex + coeff_x[...,None] * x
        return np.squeeze(grad_c)
    
    def strain_tensor(self,x):
        x, r, Ex, xEx = self.common(x)
        I, x_x, x_Ex, Ex_x = self.common_tensor(x)
        coeff_E = (-4*self.D/(r**4) + 2*Kn(2,r)*self.F/(r**2))
        coeff_I = (-2*self.C/(r**4) + 8*self.D/(r**6) - self.F*Kn(3,r)/(r**3))*xEx
        coeff_ExpxE = (-2*self.C/(r**4)+16*self.D/(r**6) - 2*self.F*Kn(3,r)/(r**3))
        coeff_xx = (8*self.C/(r**6) - 48 *self.D/(r**8) + self.F *Kn(4,r)/(r**4))*xEx
        e = coeff_E[...,None,None]*self.E[None,...] + coeff_I[...,None,None] * I + coeff_ExpxE[...,None,None] * (Ex_x + x_Ex) + coeff_xx[...,None,None]*x_x
        return np.squeeze(e) 

    def vorticity(self,x):
        x, r, Ex, xEx = self.common(x)
        coeff = 4.0*self.C/(r**4)
        vort = coeff[...,None]*np.cross(x,Ex)
        return np.squeeze(vort)   

    def stress_tensor(self,x):
        x, r, Ex, xEx = self.common(x)
        I, x_x, x_Ex, Ex_x = self.common_tensor(x)
        coeff_I = (2*self.C/(r**4) - self.F*Kn(2,r)/(r**2))*xEx
        sig = 2.0 * self.B *(coeff_I[...,None,None] * I + self.strain_tensor(x))
        return np.squeeze(sig)

class LinearFlow2D(Flow):
    """ Linear flow past a freely rotating cylinder """
    def __init__(self, gradVinf, a, B):
        self.ndim = 2
        E = 0.5 * (gradVinf + gradVinf.T)
        self.strain = Strain2D(E, a, B)
        self.far = FarField(None, gradVinf, B)

    def __getattr__(self, name):
        """ All methods are done by linear superposition """
        def method(*args):
            smethod = getattr(self.strain,name)
            fmethod = getattr(self.far,name)
            return smethod(*args) + fmethod(*args)
        return method
