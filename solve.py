from hippylib import *
import dolfin as dl
import numpy as np
import matplotlib.pyplot as plt
import sys

class AcousticWaveEquation:
    def __init__(self, n_elements, polynomial_order):
        self.n_elements = n_elements
        self.polynomial_order = polynomial_order
        self.f = dl.Constant(0.0)
        self.u_init = None
        self.bcs = None
            
        self.mesh = dl.UnitSquareMesh(n_elements, n_elements)
        self.Vh = dl.FunctionSpace(self.mesh, "Lagrange", self.polynomial_order)
        self.u = dl.TrialFunction(self.Vh)
        self.vhat = dl.TestFunction(self.Vh)
    
        # dolfin functions for storing timesteps
        self.u_tp1, self.u_t, self.u_tm1 = dl.Function(self.Vh), dl.Function(self.Vh), dl.Function(self.Vh)
        self.m = dl.Function(self.Vh)
        
    def setWaveSpeed(self, m_expression):
        self.m.assign(dl.interpolate(m_expression, self.Vh))
        
    def setInitialCondition(self, u_init_expression):        
        self.u_init = dl.interpolate(u_init_expression, self.Vh)

    def updateWeakForm(self, dt):
        self.lhs = (self.u * self.vhat / dt**2 + dl.inner(dl.grad(self.m**2 * self.vhat), dl.grad(self.u))) * dl.dx 
        self.rhs = ( 2 * self.u_t / dt**2 - self.u_tm1 / dt**2 + self.f) * self.vhat * dl.dx
        
    def setForcing(self, forcing):
        # forcing needs to be set after functions have been created
        self.f = forcing

    def writeWaveSpeed(self, run_name):
        paraview_file = dl.File(f'paraview/wave_speed_{run_name}_.pvd')
        paraview_file << self.m, 0

    def writeStatesToNumpy(self, run_name, numpy_states):
        filename = f'numpy/states_{run_name}.npy'
        np.save(filename, numpy_states)
        
    def solve(self, dt, final_time, run_name=''):
        # initial condition needs set first
        if self.u_init is None:
            raise ValueError("setInitialCondition must be called first\n   setInitialCondition(u_init_expression)")

        print(f'Solving with {self.Vh.dim()} degrees of freedom')
        
        # weak forms need to be updated for new dt and possible new wave speed
        self.updateWeakForm(dt)
        
        # for first timestep, set tm1 to t
        self.u_t.assign(self.u_init)
        self.u_tm1.assign(self.u_t)
        print(run_name)
        paraview_file = dl.File(f"paraview/solution_{run_name}_.pvd")
        
        T = 0
        i = 0
        n_steps = int(final_time / dt)
        
        numpy_states = np.zeros((n_steps+1, self.u_t.vector().get_local().shape[0]))
        numpy_states[0, :] = self.u_t.vector().get_local()

        for n in range(n_steps):
            A, b = dl.assemble_system(self.lhs, self.rhs, bcs=self.bcs)
            dl.solve(A, self.u_tp1.vector(), b)
            
            # push back timesteps
            self.u_tm1.assign(self.u_t)
            self.u_t.assign(self.u_tp1)

            T = (n+1)*dt

            # pretty printing status
            if (n % (n_steps//100) == 0):
                print(f"{n / n_steps * 100:.0f}% done")
                sys.stdout.write("\033[F")
            
            self.u_tp1.rename("u", "u")
            paraview_file << self.u_tp1, T

            numpy_states[n+1, :] = self.u_t.vector().get_local()

        self.writeWaveSpeed(run_name)
        self.writeStatesToNumpy(run_name, numpy_states)
        print("100% done")
        return self.u_tp1

class RectangularAnomaly:
    def __init__(self, x_lims, y_lims, speed_perturbation):
        self.x_lims = self.checkLimits(x_lims)
        self.y_lims = self.checkLimits(y_lims)
        self.speed = speed_perturbation

    def checkLimits(self, lims):
        lims = self.checkLimitsOrder(lims)
        lims = self.checkLimitsBounds(lims)
        return lims

    def checkLimitsOrder(self, lims):
        # makes sure upper and lower bounds make sense
        if lims[1] < lims[0]:
            temp = lims[1]
            lims[1] = lims[0]
            lims[0] = temp
        return lims

    def checkLimitsBounds(self, lims):
        # assume lims is already ordered and that at least
        # one of lims is within the range [0, 1]
        if lims[0] < 0.0:
            lims[0] = 0.0
        if lims[1] > 1.0:
            lims[1] = 1.0
        return lims
    
    def buildExpressionString(self):
        return f'+({self.speed})*(x[0]>{self.x_lims[0]} && x[0]<{self.x_lims[1]} && x[1]>{self.y_lims[0]} && x[1]<{self.y_lims[1]})'
    
class WaveSpeedWithAnomalies:
    def __init__(self, background):
        self.background = background
        self.anomalies = []
        
    def addAnomaly(self, x_lims, y_lims, speed_perturbation):
         self.anomalies.append(RectangularAnomaly(x_lims, y_lims, speed_perturbation))

    def buildExpression(self):
        return self.multiAnomalyExpressionBuilder()
        
    def multiAnomalyExpressionBuilder(self):
        anomaly_string = f'{self.background}'
        for anomaly in self.anomalies:
            anomaly_string += anomaly.buildExpressionString()
        return dl.Expression(anomaly_string, degree=5)

def anomalyBoundariesFromCWH(center, width, height):
    # cehcking whether lims are in the domain is done in
    # RectangularAnomaly class during lim checking
    x_lims = [center - width, center + width]
    y_lims = [center - height, center + height]
    return x_lims, y_lims
    
def scenariosToRun(problem, n_draws):
    # define boundary an initial conditions
    def boundary(x, on_boundary):
        return on_boundary
    
    dirichlet = dl.DirichletBC(problem.Vh, dl.Constant(0.0), boundary)
    possible_bcs = [{ 'name' : 'neumann', 'val' : None},]
    #{ 'name' : 'dirichlet', 'val' : dirichlet}]
    scenarios = []
    for bcs in possible_bcs:
        for n in range(n_draws):
            # wave speed definition with anomalies
            background = 1.5
            wave_speed = WaveSpeedWithAnomalies(background)
            n_anomalies = np.random.randint(1, 5)
            for _ in range(n_anomalies):
                cwh_coords = list(np.random.uniform(0.0, 1.0, 3))
                speed_perturbation = np.random.uniform(0.0, 10.0)
                wave_speed.addAnomaly(*anomalyBoundariesFromCWH(*cwh_coords), 10.0)
                
            problem.setWaveSpeed(wave_speed.buildExpression())
            scenarios.append({
                'bcs' : bcs['val'],
                'wave_speed' : wave_speed,
                'run_name' : bcs['name'] + '_' + str(n),
                'n_anomalies' : n_anomalies,
            })
    return scenarios

def runScenario(problem, scenario, dt, final_time):
    problem.bcs = scenario['bcs']
    problem.setWaveSpeed(scenario['wave_speed'].buildExpression())
    solution = problem.solve(dt, final_time, scenario['run_name'])
    return solution
        
if __name__ == '__main__':    
    # define mesh
    n_elements = 16
    polynomial_order = 3

    problem = AcousticWaveEquation(n_elements, polynomial_order)

    mean = 0.0
    alpha = 0.4
    u_init_expression = dl.Expression("mean+alpha*sin(2*pi*x[0])*sin(2*pi*x[0])*cos(2*pi*x[1])*cos(2*pi*x[1])", degree=11, mean=mean, alpha=alpha)
    problem.setInitialCondition(u_init_expression)
    problem.setForcing(2 * dl.sin(np.pi * problem.u_t))
       
    t_final = 1.0
    dt = 0.001

    n_scenarios = 20
    scenarios = scenariosToRun(problem, 20)
    for ind, scenario in enumerate(scenarios, 1):
        print(f'running scenario {ind} / {len(scenarios)}\n   {scenario}\n')
        solution = runScenario(problem, scenario, dt, t_final)
        
    # define weak form of adjoint problem



