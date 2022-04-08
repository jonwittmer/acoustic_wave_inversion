# Copyright (c) 2016-2018, The University of Texas at Austin 
# & University of California--Merced.
# Copyright (c) 2019-2020, The University of Texas at Austin 
# University of California--Merced, Washington University in St. Louis.
#
# All Rights reserved.
# See file COPYRIGHT for details.
#
# This file is part of the hIPPYlib library. For more information and source code
# availability see https://hippylib.github.io.
#
# hIPPYlib is free software; you can redistribute it and/or modify it under the
# terms of the GNU General Public License (as published by the Free
# Software Foundation) version 2.0 dated June 1991.

import dolfin as dl
import ufl
import numpy as np
import matplotlib.pyplot as plt
import argparse

import sys
import os
sys.path.append( os.environ.get('HIPPYLIB_BASE_DIR', "../../") )
from hippylib import *

class SpaceTimePointwiseStateObservation(Misfit):
    def __init__(self, Vh,
                 observation_times,
                 targets,
                 d = None,
                 noise_variance=None):
        
        self.Vh = Vh
        self.observation_times = observation_times
        
        self.B = assemblePointwiseObservation(self.Vh, targets)
        self.ntargets = targets
        
        if d is None:
            self.d = TimeDependentVector(observation_times)
            self.d.initialize(self.B, 0)
        else:
            self.d = d
            
        self.noise_variance = noise_variance
        
        ## TEMP Vars
        self.u_snapshot = dl.Vector()
        self.Bu_snapshot = dl.Vector()
        self.d_snapshot  = dl.Vector()
        self.B.init_vector(self.u_snapshot, 1)
        self.B.init_vector(self.Bu_snapshot, 0)
        self.B.init_vector(self.d_snapshot, 0)
        
    def observe(self, x, obs):        
        obs.zero()
        
        for t in self.observation_times:
            x[STATE].retrieve(self.u_snapshot, t)
            self.B.mult(self.u_snapshot, self.Bu_snapshot)
            obs.store(self.Bu_snapshot, t)
            
    def cost(self, x):
        c = 0
        for t in self.observation_times:
            x[STATE].retrieve(self.u_snapshot, t)
            self.B.mult(self.u_snapshot, self.Bu_snapshot)
            self.d.retrieve(self.d_snapshot, t)
            self.Bu_snapshot.axpy(-1., self.d_snapshot)
            c += self.Bu_snapshot.inner(self.Bu_snapshot)
            
        return c/(2.*self.noise_variance)
    
    def grad(self, i, x, out):
        out.zero()
        if i == STATE:
            for t in self.observation_times:
                x[STATE].retrieve(self.u_snapshot, t)
                self.B.mult(self.u_snapshot, self.Bu_snapshot)
                self.d.retrieve(self.d_snapshot, t)
                self.Bu_snapshot.axpy(-1., self.d_snapshot)
                self.Bu_snapshot *= 1./self.noise_variance
                self.B.transpmult(self.Bu_snapshot, self.u_snapshot) 
                out.store(self.u_snapshot, t)           
        else:
            pass
            
    def setLinearizationPoint(self, x, gauss_newton_approx=False):
        pass
    
    def apply_ij(self, i,j, direction, out):
        out.zero()
        if i == STATE and j == STATE:
            for t in self.observation_times:
                direction.retrieve(self.u_snapshot, t)
                self.B.mult(self.u_snapshot, self.Bu_snapshot)
                self.Bu_snapshot *= 1./self.noise_variance
                self.B.transpmult(self.Bu_snapshot, self.u_snapshot) 
                out.store(self.u_snapshot, t)
        else:
            pass    
        

class TimeDependentAcousticWave:
    def __init__(self, mesh, Vh, prior, misfit, simulation_times):
        self.mesh = mesh
        self.Vh = Vh[0]
        self.prior = prior
        self.misfit = misfit
        
        # Assume constant timestepping
        self.simulation_times = simulation_times
        self.dt = simulation_times[1] - simulation_times[0]
        
        dt_expr = dl.Constant(dt)

        self.m = dl.Function(self.Vh)
        self.m.vector() = self.prior.mean
        self.buildWeakForms()
                
        self.solver = PETScLUSolver( self.mesh.mpi_comm() )
        self.solvert = PETScLUSolver( self.mesh.mpi_comm() ) 
                        
        # Part of model public API
        self.gauss_newton_approx = False

    def buildWeakForms(self):
        u = dl.TrialFunction(self.Vh)
        v = dl.TestFunction(self.Vh)
        self.forward_lhs = (u * v / self.dt**2 + dl.inner(dl.grad(self.m**2 * v), dl.grad(u))) * dl.dx 
        self.adjoint_lhs = (u * v / self.dt**2 + dl.inner(dl.grad(self.m**2 * u), dl.grad(v))) * dl.dx 

    def updateForwardRhs(self, t, u):
        u_t = dl.Function(self.Vh)
        u_tm1 = dl.Function(self.Vh)
        u_t.vector() = u.retrieve(t)
        if t == 0:
            u_tm1.vector() = u_t
        else:
            u_tm1.vector() = u.retrieve(t - self.dt)
        vhat = dl.TestFunction(Vh)
        rhs_expression = (2 * u_t / self.dt**2 - u_tm1 / self.dt**2 + dl.sin(u_t)) * vhat * dl.dx
        return dl.assemble(rhs_expression)

    def updateAdjointRhs(self, t, u, v):
        v_t = dl.Function(self.Vh)
        v_tp1 = dl.Function(self.Vh)
        u_t = dl.Function(self.Vh)
        v_t.vector() = v.retrieve(t)
        if t == v.times[-1]:
            v_tp1.vector() = v_t
        else:
            v_tp1.vector() = v.retrieve(t + self.dt)
        u_t.vector() = u.retrieve(t)
        uhat = dl.TestFunction(Vh)
        rhs_expression = (2 * v_t / self.dt**2 - v_tp1 / self.dt**2 + dl.cos(u_t) * v_t) * uhat * dl.dx
        return dl.assemble(rhs_expression)
        
    def generate_vector(self, component = "ALL"):
        if component == "ALL":
            u = TimeDependentVector(self.simulation_times)
            u.initialize(self.forward_lhs, 0)
            m = dl.Vector()
            self.prior.init_vector(m, 0)
            p = TimeDependentVector(self.simulation_times)
            p.initialize(self.forward_lhs, 0)
            return [u, m, p]
        elif component == STATE:
            u = TimeDependentVector(self.simulation_times)
            u.initialize(self.forward_lhs, 0)
            return u
        elif component == PARAMETER:
            m = dl.Vector()
            self.prior.init_vector(m, 0)
            return m
        elif component == ADJOINT:
            p = TimeDependentVector(self.simulation_times)
            p.initialize(self.forward_lhs, 0)
            return p
        else:
            raise
    
    def init_parameter(self, m):
        self.prior.init_vector(m, 0)        
          
    def cost(self, x):
        Rdx = dl.Vector()
        self.prior.init_vector(Rdx,0)
        dx = x[PARAMETER] - self.prior.mean
        self.prior.R.mult(dx, Rdx)
        reg = .5*Rdx.inner(dx)
        
        misfit = self.misfit.cost(x)
                
        return [reg+misfit, reg, misfit]
    
    def solveFwd(self, out, x):
        self.buildWeakForms()
        self.solver.set_operator(dl.as_backend_type(self.forward_lhs))
        u = dl.Vector()
        self.forward_lhs.init_vector(u, 0)
        out.zero()
        for t in self.simulation_times[1::]:
            rhs = self.updateForwardRhs(t, x[STATE])
            self.solver.solve(u, rhs)
            out.store(u, t)
            uold = u
    
    def solveAdj(self, out, x):
        grad_state = TimeDependentVector(self.simulation_times)
        grad_state.initialize(self.M, 0)
        self.misfit.grad(STATE, x, grad_state)
        
        out.zero()
        
        pold = dl.Vector()
        self.M.init_vector(pold, 0)
            
        p = dl.Vector()
        self.M.init_vector(p, 0)
        
        rhs = dl.Vector()
        self.M.init_vector(rhs, 0)
        
        grad_state_snap = dl.Vector()
        self.M.init_vector(grad_state_snap, 0)

        for t in self.simulation_times[::-1]:
            self.Mt_stab.mult(pold,rhs)
            grad_state.retrieve(grad_state_snap, t)
            rhs.axpy(-1., grad_state_snap)
            self.solvert.solve(p, rhs)
            pold = p
            out.store(p, t)            
            
    def evalGradientParameter(self,x, mg, misfit_only=False):
        self.prior.init_vector(mg,1)
        if misfit_only == False:
            dm = x[PARAMETER] - self.prior.mean
            self.prior.R.mult(dm, mg)
        else:
            mg.zero()
        
        p0 = dl.Vector()
        self.M.init_vector(p0,0)
        x[ADJOINT].retrieve(p0, self.simulation_times[1])
        
        mg.axpy(-1., self.Mt_stab*p0)
        
        g = dl.Vector()
        self.M.init_vector(g, 1)
        
        self.prior.Msolver.solve(g, mg)
        
        grad_norm = g.inner(mg)
        
        return grad_norm
        
    
    def setPointForHessianEvaluations(self, x, gauss_newton_approx=False):
        """
        Specify the point x = [u,a,p] at which the Hessian operator (or the Gauss-Newton approximation)
        need to be evaluated.
        
        Nothing to do since the problem is linear
        """
        self.gauss_newton_approx = gauss_newton_approx
        return

        
    def solveFwdIncremental(self, sol, rhs):
        sol.zero()
        uold = dl.Vector()
        u = dl.Vector()
        Muold = dl.Vector()
        myrhs = dl.Vector()
        self.M.init_vector(uold, 0)
        self.M.init_vector(u, 0)
        self.M.init_vector(Muold, 0)
        self.M.init_vector(myrhs, 0)

        for t in self.simulation_times[1::]:
            self.M_stab.mult(uold, Muold)
            rhs.retrieve(myrhs, t)
            myrhs.axpy(1., Muold)
            self.solver.solve(u, myrhs)
            sol.store(u,t)
            uold = u


        
    def solveAdjIncremental(self, sol, rhs):
        sol.zero()
        pold = dl.Vector()
        p = dl.Vector()
        Mpold = dl.Vector()
        myrhs = dl.Vector()
        self.M.init_vector(pold, 0)
        self.M.init_vector(p, 0)
        self.M.init_vector(Mpold, 0)
        self.M.init_vector(myrhs, 0)

        for t in self.simulation_times[::-1]:
            self.Mt_stab.mult(pold,Mpold)
            rhs.retrieve(myrhs, t)
            Mpold.axpy(1., myrhs)
            self.solvert.solve(p, Mpold)
            pold = p
            sol.store(p, t)  
            
    
    def applyC(self, dm, out):
        out.zero()
        myout = dl.Vector()
        self.M.init_vector(myout, 0)
        self.M_stab.mult(dm,myout)
        myout *= -1.
        t = self.simulation_times[1]
        out.store(myout,t)
        
        myout.zero()
        for t in self.simulation_times[2:]:
            out.store(myout,t)
    
    def applyCt(self, dp, out):
        t = self.simulation_times[1]
        dp0 = dl.Vector()
        self.M.init_vector(dp0,0)
        dp.retrieve(dp0, t)
        dp0 *= -1.
        self.Mt_stab.mult(dp0, out)

    
    def applyWuu(self, du, out):
        out.zero()
        self.misfit.apply_ij(STATE, STATE, du, out)

    
    def applyWum(self, dm, out):
        out.zero()

    
    def applyWmu(self, du, out):
        out.zero()
    
    def applyR(self, dm, out):
        self.prior.R.mult(dm,out)
    
    def applyWmm(self, dm, out):
        out.zero()
        
    def exportState(self, x, filename, varname):
        out_file = dl.XDMFFile(self.Vh[STATE].mesh().mpi_comm(), filename)
        out_file.parameters["functions_share_mesh"] = True
        out_file.parameters["rewrite_function_mesh"] = False
        ufunc = dl.Function(self.Vh[STATE], name=varname)
        t = self.simulation_times[0]
        out_file.write(vector2Function(x[PARAMETER], self.Vh[STATE], name=varname),t)
        for t in self.simulation_times[1:]:
            x[STATE].retrieve(ufunc.vector(), t)
            out_file.write(ufunc, t)    

        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Model Acoustic Wave Propagation')
    parser.add_argument('--nref',
                        default=0,
                        type=int,
                        help="Number of uniform mesh refinements")
    args = parser.parse_args()
    try:
        dl.set_log_active(False)
    except:
        pass
    np.random.seed(1)
    sep = "\n"+"#"*80+"\n"

    base_n_elements = 25
    n_elements = base_n_elements * (args.nref + 1)
    mesh = dl.UnitSquareMesh(n_elements, n_elements)

    rank = dl.MPI.rank(mesh.mpi_comm())
    nproc = dl.MPI.size(mesh.mpi_comm())
        
    if rank == 0:
        print( sep, "Set up the mesh and finite element spaces.\n","Compute wind velocity", sep )
    Vh = dl.FunctionSpace(mesh, "Lagrange", 2)
    if rank == 0:
        print("Number of dofs: {0}".format(Vh.dim()))
    
    if rank == 0:
        print(sep, "Set up Prior Information and model", sep)

    ic_mean = 0.0
    ic_alpha = 0.4
    ic_expression = dl.Expression("mean+alpha*sin(2*pi*x[0])*sin(2*pi*x[0])*cos(2*pi*x[1])*cos(2*pi*x[1])", degree=5, mean=ic_mean, alpha=ic_alpha)
    true_initial_condition = dl.interpolate(ic_expression, Vh).vector()

    gamma = 1.
    delta = 8.
    prior = BiLaplacianPrior(Vh, gamma, delta, robin_bc=True)
    prior.mean = dl.interpolate(dl.Constant(1.0), Vh).vector()
    if rank == 0:
        print("Prior regularization: (delta - gamma*Laplacian)^order: delta={0}, gamma={1}, order={2}".format(delta, gamma, 2))

    # for now, assume that the simulation and observations happen at the same times.
    # We will have a few "seismic stations" throughout the domain registering observations at each timestep
    t_init = 0.
    t_final = 1.
    dt = .001
    simulation_times = np.arange(t_init, t_final+.5*dt, dt)
    observation_times = simulations_times.copy()
    
    targets = np.array([[0.1, 0.1], [0.9, 0.9], [0.1, 0.9], [0.9, 0.1], [0.5, 0.5]])
    if rank == 0:
        print ("Number of observation points: {0}".format(targets.shape[0]))
    misfit = SpaceTimePointwiseStateObservation(Vh, observation_times, targets)
    
    problem = TimeDependentAcousticWave(mesh, [Vh,Vh,Vh], prior, misfit, simulation_times)
    
    if rank == 0:
        print( sep, "Generate synthetic observation", sep )
    rel_noise = 0.01
    utrue = problem.generate_vector(STATE)
    x = [utrue, true_initial_condition, None]
    problem.solveFwd(x[STATE], x)
    misfit.observe(x, misfit.d)
    MAX = misfit.d.norm("linf", "linf")
    noise_std_dev = rel_noise * MAX
    parRandom.normal_perturb(noise_std_dev,misfit.d)
    misfit.noise_variance = noise_std_dev*noise_std_dev
    
    
    if rank == 0:
        print( sep, "Test the gradient and the Hessian of the model", sep )
    m0 = true_initial_condition.copy()
    modelVerify(problem, m0, is_quadratic = True, misfit_only = True,  verbose = (rank == 0))

    if rank == 0:
        print( sep, "Compute the reduced gradient and hessian", sep)
    [u,m,p] = problem.generate_vector()
    problem.solveFwd(u, [u,m,p])
    problem.solveAdj(p, [u,m,p])
    mg = problem.generate_vector(PARAMETER)
    grad_norm = problem.evalGradientParameter([u,m,p], mg)
    
    if rank == 0:    
        print( "(g,g) = ", grad_norm )
    
    if rank == 0:
        print( sep, "Compute the low rank Gaussian Approximation of the posterior", sep  )
    
    H = ReducedHessian(problem, misfit_only=True) 
    k = 80
    p = 20
    if rank == 0:
        print( "Double Pass Algorithm. Requested eigenvectors: {0}; Oversampling {1}.".format(k,p) )
    
    Omega = MultiVector(x[PARAMETER], k+p)
    parRandom.normal(1., Omega)

    d, U = doublePassG(H, prior.R, prior.Rsolver, Omega, k, s=1, check=False)
    posterior = GaussianLRPosterior(prior, d, U)
    
    if True:
        P = posterior.Hlr
    else:
        P = prior.Rsolver
    
    if rank == 0:
        print( sep, "Find the MAP point", sep)
    
    H.misfit_only = False
        
    solver = CGSolverSteihaug()
    solver.set_operator(H)
    solver.set_preconditioner( P )
    solver.parameters["print_level"] = 1
    solver.parameters["rel_tolerance"] = 1e-6
    if rank != 0:
        solver.parameters["print_level"] = -1
    solver.solve(m, -mg)
    problem.solveFwd(u, [u,m,p])
 
    total_cost, reg_cost, misfit_cost = problem.cost([u,m,p])
    if rank == 0:
        print( "Total cost {0:5g}; Reg Cost {1:5g}; Misfit {2:5g}".format(total_cost, reg_cost, misfit_cost) )
    
    posterior.mean = m

    compute_trace = False
    if compute_trace:
        post_tr, prior_tr, corr_tr = posterior.trace(method="Randomized", r=200)
        if rank == 0:
            print( "Posterior trace {0:5g}; Prior trace {1:5g}; Correction trace {2:5g}".format(post_tr, prior_tr, corr_tr) )
    post_pw_variance, pr_pw_variance, corr_pw_variance = posterior.pointwise_variance(method="Randomized", r=200)
    
    if rank == 0:
        print( sep, "Save results", sep  )
    problem.exportState([u,m,p], "results/conc.xdmf", "concentration")
    problem.exportState([utrue,true_initial_condition,p], "results/true_conc.xdmf", "concentration")

    with dl.XDMFFile(mesh.mpi_comm(), "results/pointwise_variance.xdmf") as fid:
        fid.parameters["functions_share_mesh"] = True
        fid.parameters["rewrite_function_mesh"] = False
    
        fid.write(vector2Function(post_pw_variance, Vh, name="Posterior"), 0)
        fid.write(vector2Function(pr_pw_variance, Vh, name="Prior"), 0)
        fid.write(vector2Function(corr_pw_variance, Vh, name="Correction"), 0)
    
    U.export(Vh, "results/evect.xdmf", varname = "gen_evect", normalize = True)
    if rank == 0:
        np.savetxt("results/eigevalues.dat", d)
    
    fid_prmean  = dl.XDMFFile(mesh.mpi_comm(), "results/pr_mean.xdmf")
    fid_prmean.write(vector2Function(prior.mean, Vh, name="prior mean"))
    
    if rank == 0:
        print( sep, "Generate samples from Prior and Posterior", sep)

    nsamples = 50
    noise = dl.Vector()
    posterior.init_vector(noise,"noise")
    s_prior = dl.Function(Vh, name="sample_prior")
    s_post = dl.Function(Vh, name="sample_post")
    with dl.XDMFFile(mesh.mpi_comm(), "results/samples.xdmf") as fid:
        fid.parameters["functions_share_mesh"] = True
        fid.parameters["rewrite_function_mesh"] = False
        for i in range(nsamples):
            parRandom.normal(1., noise)
            posterior.sample(noise, s_prior.vector(), s_post.vector())
            fid.write(s_prior, i)
            fid.write(s_post, i)
    
    if rank == 0:
        print( sep, "Visualize results", sep )
        plt.figure()
        plt.plot(range(0,k), d, 'b*', range(0,k), np.ones(k), '-r')
        plt.yscale('log')
        plt.show()

    
