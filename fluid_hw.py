import numpy as np
import taichi as ti
import taichi.math as tm

ti.init(arch=ti.cuda, default_fp=ti.f32, fast_math=True, debug=False)

res = 256
fps = 60
dt = 1.0 / fps
force_radius = res / 2.0
force_strength = 1e4
dye_decay = 1 - 1 / (fps * 2)
eps = 1e-4
# Change solving_method to "mgpcg" after you finish Task 3.
solving_method = "jacobi" # "jacobi" or "mgpcg"

class TexPair:
    def __init__(self, cur, nxt):
        self.cur = cur
        self.nxt = nxt

    def swap(self):
        self.cur, self.nxt = self.nxt, self.cur

class MouseData:
    def __init__(self):
        self.prev_mouse = None
        self.prev_color = None

    def __call__(self, gui):
        # [0:2]: normalized delta direction
        # [2:4]: current mouse xy
        # [4:7]: color
        mouse_data = np.zeros(8, dtype=np.float32)
        if gui.is_pressed(ti.GUI.LMB):
            mxy = np.array(gui.get_cursor_pos(), dtype=np.float32) * res
            if self.prev_mouse is None:
                self.prev_mouse = mxy
                # Set lower bound to 0.3 to prevent too dark colors
                self.prev_color = (np.random.rand(3) * 0.7) + 0.3
            else:
                mdir = mxy - self.prev_mouse
                mdir = mdir / (np.linalg.norm(mdir) + eps)
                mouse_data[0], mouse_data[1] = mdir[0], mdir[1]
                mouse_data[2], mouse_data[3] = mxy[0], mxy[1]
                mouse_data[4:7] = self.prev_color
                self.prev_mouse = mxy
        else:
            self.prev_mouse = None
            self.prev_color = None
        return mouse_data

    def reset(self):
        self.prev_mouse = None
        self.prev_color = None

@ti.data_oriented
class Solver:
    def solve(self, rhs, x_pair, num_iter=-1):
        # This is the base class for the solvers.
        raise NotImplementedError
    
    @ti.kernel
    def dot(self, a: ti.template(), b: ti.template()) -> float:
        ret = 0.0
        for I in ti.grouped(a):
            ret += a[I] * b[I]
        return ret
    
    @ti.kernel
    def residue(self, b: ti.template(), x: ti.template(), r: ti.template()):
        for I in ti.grouped(b):
            ret = 0.0
            cnt = 0
            for i, j in ti.static([[-1, 0], [1, 0], [0, -1], [0, 1]]):
                J = I + [i, j]
                if all(J >= 0 and J < x.shape):
                    ret += x[J]
                    cnt += 1
            r[I] = b[I] + ret - cnt * x[I] 
    
    @ti.kernel
    def recenter(self, x0: ti.template(), x: ti.template()):
        ret = 0.0
        for I in ti.grouped(x0):
            ret += x0[I]
        ti.sync()
        for I in ti.grouped(x):
            x[I] = x[I] - ret / (x.shape[0] * x.shape[1])

class PoissonJacobiSolver(Solver):
    def __init__(self, resolution):
        self.resolution = resolution

    @ti.kernel
    def iteration(self, b: ti.template(), x0: ti.template(), x: ti.template()):
        pass
        ################################################################################
        # Task 2.2 (2 points)
        ################################################################################
        # 
        # Solve the Poisson equation by the Jacobi iterative method. The Poisson
        # equation is discretized as Ax = b, where A is the discretized Laplacian
        # matrix, b is the right-hand-side velocity divergence, and x is the pressure
        # field to be solved.
        # 
        # Recall for the interior grid points, the stencil for A is:
        #       -1
        #    -1  4  -1
        #       -1
        #
        # And it is different at boundaries. For example, at the top-left corner, it is:
        #  2 -1
        # -1
        #
        # Let A = D + U, where D and U are the diagonal and off-diagonal parts of A
        # respectively. The Jacobi iteration is performed as x = D^-1 (b - U x0).
        #
        # In this function, we instead implement the following fixed-point iteration:
        # x = (b - U x0 + (4I - D) x0) / 4, where I is the identity matrix.
        #
        # This is essentially the Jacobi iteration with a flexible modification at the
        # boundaries: we treat boundary cells the same as interior ones, but replace any 
        # invalid neighbor access with the value at the current cell.
        #
        # This function has the following arguments:
        # 1. The right-hand-side field "b";
        # 2. The old solution field "x0";
        # 3. The output solution field "x".
        #
        # Fill in the field "x" accordingly.
        #
        ################################################################################

        
    
    def solve(self, rhs, x_pair, num_iter=10000):
        tol = max(force_strength, self.dot(rhs, rhs)) / res
        for i in range(num_iter):
            self.iteration(rhs, x_pair.cur, x_pair.nxt)
            x_pair.swap()
            if i % 100 == 99:
                self.recenter(x_pair.cur, x_pair.nxt)
                x_pair.swap()
                self.residue(rhs, x_pair.cur, x_pair.nxt)
                if self.dot(x_pair.nxt, x_pair.nxt) < tol:
                    return

class PoissonMultiGridSolver(Solver):
    def __init__(self, resolution, bottom_resolution, smoothing_iter=4, bottom_solving_iter=4):
        self.resolution_list = []
        while resolution > bottom_resolution:
            self.resolution_list.append(resolution)
            resolution = resolution // 2
        self.num_levels = len(self.resolution_list)
        self.smoothing_iter = smoothing_iter
        self.bottom_solving_iter = bottom_solving_iter
        self.x = {res: TexPair(ti.field(dtype=float, shape=(res, res)), ti.field(dtype=float, shape=(res, res))) for res in self.resolution_list}
        self.r = {res: ti.field(dtype=float, shape=(res, res)) for res in self.resolution_list}
        self.b = {res: ti.field(dtype=float, shape=(res, res)) for res in self.resolution_list}
        self.smoother_list = {res: PoissonJacobiSolver(res) for res in self.resolution_list}

    def smooth(self, resolution, num_iter):
        smoother = self.smoother_list[resolution]
        ################################################################################
        # Task 3.1 (2 points)
        ################################################################################
        #
        # Implement the smoother of the multigrid method. A smoother is usually an
        # iterative solver. In this assignment, we use the Jacobi iterative method as the
        # smoother, which you should have implemented in Task 2.2.
        #
        # Call the "iteration" method of the "smoother" by "num_iter" times.
        #
        ################################################################################


    @ti.kernel
    def prolongate(self, x0: ti.template(), x: ti.template()):
        for I in ti.grouped(x):
            J = I // 2
        ################################################################################
        # Task 3.2 (2 points)
        ################################################################################
        #
        # Implement the prolongation operator in the multigrid method. We prolongate the
        # coarse solution to the fine resolution, and then add it to the solution to 
        # correct the low-frequency error.
        # ---------------------------+---------------------------+
        # |                          |                           |
        # |         (h)              |           (h)             |
        # |       X                  |         X                 |
        # |         (i, j+1)         |           (i+1, j+1)      |
        # |                          |                           |
        # |                       (2h)                           |
        # |-------------------  X              ------------------|
        # |                       (i//2, j//2)                   |
        # |                          |                           |
        # |         (h)              |           (h)             |
        # |       X                  |         X                 |
        # |         (i, j)           |           (i+1, j)        |
        # |                          |                           |
        # ---------------------------+---------------------------+ 
        # In this assignment, we adopt the following simple prolongation strategy:
        # the (i, j) entry of the prolongated vector is x_{i/2, j/2}.
        # 
        # This function has the following arguments:
        # 1. The coarse solution field "x0";
        # 2. The fine solution field "x".
        #
        # Update the solution "x" accordingly.
        #
        ################################################################################

    @ti.kernel
    def restrict(self, r: ti.template(), b: ti.template()):
        pass
        ################################################################################
        # Task 3.3 (2 points)
        ################################################################################
        #
        # The restriction operator in the multigrid method. The right-hand-side "b" of
        # the linear equation in the coarser grid is the residual "r" of the fine grid.
        #
        # In this assignment, the restriction operator is the transpose of the
        # prolongation operator.
        #
        # This function has the following arguments:
        # 1. The fine residual field "r".
        # 2. The coarse right-hand-side field "b";
        #
        # Fill in the field "b" accordingly.
        #
        ################################################################################

    def v_cycle(self, resolution):
        self.x[resolution].cur.fill(0)
        self.x[resolution].nxt.fill(0)
        self.smooth(resolution, self.smoothing_iter)
        if resolution > self.resolution_list[-1]:
            self.r[resolution].fill(0)
            self.residue(self.b[resolution], self.x[resolution].cur, self.r[resolution])
            self.restrict(self.r[resolution], self.b[resolution // 2])
            self.v_cycle(resolution // 2)
            self.prolongate(self.x[resolution // 2].cur, self.x[resolution].cur)
        else:
            self.smooth(resolution, self.bottom_solving_iter)
        self.smooth(resolution, self.smoothing_iter)

    def solve(self, rhs, x_pair, num_iter=100):
        self.b[res].copy_from(rhs)
        for i in range(num_iter):
            self.v_cycle(res)
        x_pair.cur.copy_from(self.x[res].nxt)

class PoissonMultiGridPreconditionedConjugateGradientSolver(Solver):
    def __init__(self):
        self.preconditioner = PoissonMultiGridSolver(res, 4)
        self.x = ti.field(dtype=float, shape=(res, res))
        self.r = ti.field(dtype=float, shape=(res, res))
        self.z = TexPair(ti.field(dtype=float, shape=(res, res)), ti.field(dtype=float, shape=(res, res)))
        self.p = ti.field(dtype=float, shape=(res, res))
        self.Ap = ti.field(dtype=float, shape=(res, res))
    
    @ti.kernel
    def saxpy_xr(self, x: ti.template(), r: ti.template(), p: ti.template(), Ap: ti.template(), alpha: float):
        for I in ti.grouped(p):
            x[I] = x[I] + alpha * p[I]
            r[I] = r[I] - alpha * Ap[I]

    @ti.kernel
    def saxpy_p(self, z: ti.template(), p: ti.template(), beta: float):
        for I in ti.grouped(p):
            p[I] = z[I] + beta * p[I]

    @ti.kernel
    def Ax(self, x: ti.template(), Ax: ti.template()):
        for I in ti.grouped(x):
            ret = 0.0
            cnt = 0
            for i, j in ti.static([[-1, 0], [1, 0], [0, -1], [0, 1]]):
                J = I + [i, j]
                if all(J >= 0 and J < x.shape):
                    ret += x[J]
                    cnt += 1
            Ax[I] = cnt * x[I] - ret
    
    def solve(self, rhs, x_pair, num_iter=-1):
        tol = max(force_strength, self.dot(rhs, rhs)) / res
        self.x.fill(0)
        self.z.cur.fill(0)
        self.r.copy_from(rhs)
        self.preconditioner.solve(self.r, self.z, 1)
        self.p.copy_from(self.z.cur)
        rMr = self.dot(self.z.cur, self.r)

        while True:
            self.Ax(self.p, self.Ap)
            self.saxpy_xr(self.x, self.r, self.p, self.Ap, rMr / (self.dot(self.p, self.Ap) + eps))
            if self.dot(self.r, self.r) < tol:
                break
            self.preconditioner.solve(self.r, self.z, 1)
            new_rMr = self.dot(self.z.cur, self.r)
            self.saxpy_p(self.z.cur, self.p, new_rMr / (rMr + eps))
            rMr = new_rMr

        x_pair.cur.copy_from(self.x)

@ti.data_oriented
class Simulator:
    def __init__(self):
        self.pressure = TexPair(ti.field(dtype=float, shape=(res, res)), ti.field(dtype=float, shape=(res, res)))
        self.u = TexPair(ti.field(dtype=float, shape=(res + 1, res)), ti.field(dtype=float, shape=(res + 1, res)))
        self.v = TexPair(ti.field(dtype=float, shape=(res, res + 1)), ti.field(dtype=float, shape=(res, res + 1)))
        self.dye_buffer = TexPair(ti.Vector.field(3, dtype=float, shape=(res, res)), ti.Vector.field(3, dtype=float, shape=(res, res)))
        self.velocity_divs = ti.field(dtype=float, shape=(res, res))
        self.velocity_curls = ti.field(dtype=float, shape=(res, res))
        self.mouse_data = MouseData()
        if solving_method == "jacobi":
            self.solver = PoissonJacobiSolver(res)
        elif solving_method == "mgpcg":
            self.solver = PoissonMultiGridPreconditionedConjugateGradientSolver()
        self.visual_mode = "d"
    
    def reset(self):
        for pairs in [self.pressure, self.u, self.v, self.dye_buffer]:
            pairs.cur.fill(0)
            pairs.nxt.fill(0)
        self.velocity_divs.fill(0)
        self.velocity_curls.fill(0)
        self.mouse_data.reset()
    
    @ti.func
    def sample(self, f, pos, displacement):
        rel_pos = ti.max(eps, ti.min(pos - displacement, tm.vec2(f.shape) - 1 - eps))
        ret = f[0, 0]
        ################################################################################
        # Task 1.1 (3 points)
        ################################################################################
        #
        # Implement bilinear interpolation to sample the field value "f" at a given 
        # position "pos". The "rel_pos" is the position relative to the field's origin.
        # For instance, in a staggered MAC grid, the x-velocity field `u` is staggered
        # in the y-direction by 0.5. That means "u[0, 0]" corresponds to a physical
        # location of (0, 0.5), so in this case, "displacement" is (0, 0.5).
        #
        # This function has the following arguments:
        # 1. A Taichi field "f" (possibly a scalar field or a vector field);
        # 2. The queried position "pos", a 2-dimentional vector;
        # 3. The displacement of the grid origin "displacement". 
        #
        # Note: when "f" is a scalar field, the return value "ret" should be a scalar;
        # similarly, when "f" is a vector field, "ret" should be a vector. It should be
        # not hard to support the both case at the same time. If you find it difficult,
        # you may split this function into "sample" and "sample_dye" (the "dye" field
        # is a 3D-vector field) and replace the "sample" in the "advect_dye" function
        # with your "sample_dye".
        #
        # Replace "ret = f[0, 0]" with the correct computation of the sampled value.
        #
        ################################################################################
        return ret

    @ti.func
    def backtrace(self, u, v, pos) -> tm.vec2:
        ret = pos
        ################################################################################
        # Task 1.2 (3 points)
        ################################################################################
        #
        # Use the 2nd-order Runge-Kutta method to backtrace the particle's previous
        # position based on the current position "pos". First estimate the velocity at
        # time dt / 2 as "v_mid", and get the backtraced 2D position based on "v_mid".
        #
        # This function has the following arguments:
        # 1. x-directional velocity field "u" in shape (res + 1, res);
        # 2. y-directional velocity field "v" in shape (res, res + 1);
        # 3. physical location "pos", a 2D vector.
        #
        # Replace "ret = pos" with the correct computation of the backtraced position.
        #
        ################################################################################
        return ret

    @ti.kernel
    def advect_dye(self, u0: ti.template(), v0: ti.template(), q0: ti.template(), q: ti.template()):
        for I in ti.grouped(q):
            q[I] = self.sample(q0, self.backtrace(u0, v0, I + tm.vec2(0.5, 0.5)), tm.vec2(0.5, 0.5)) * dye_decay
    
    @ti.kernel
    def advect_velocity(self, u0: ti.template(), v0: ti.template(), u: ti.template(), v: ti.template()):
        pass
        ################################################################################
        # Task 1.3 (2 points)
        ################################################################################
        #
        # Advect the velocity field using semi-Lagrangian. Given the old velocity field
        # "u0" and "v0", you need to fill in the advected velocity field "u" and "v" by
        # sampling the old velocity value at the backtraced position.
        #
        # This function has the following arguments:
        # 1. the previous x-directional velocity field "u0" in shape (res + 1, res);
        # 2. the previous y-directional velocity field "v0" in shape (res, res + 1);
        # 3. the output x-directional velocity field "u" in shape (res + 1, res);
        # 4. the output y-directional velocity field "v" in shape (res, res + 1);
        #
        # Fill in "u" (1 point) and "v" (1 point) accordingly.
        #
        # Note: the velocity field follows Dirichlet boundary conditions. The 
        # x-directional velocity is zero at the left and right boundaries, while the 
        # y-directional velocity is zero at the bottom and top boundaries. For example, 
        # u[0, i] should always be zero. 
        ################################################################################


    @ti.kernel
    def project(self, u: ti.template(), v: ti.template(), p: ti.template()):
        pass
        ################################################################################
        # Task 2.3 (2 points)
        ################################################################################
        #
        # Project the velocity field to be divergence-free. After solving the pressure
        # from the Poisson equation, we adjust the velocity with the gradient of the
        # pressure to zero out the divergence. In the staggered MAC-grid setting, the
        # directional gradient of pressure located at the velocity node can be computed
        # without interpolation. Therefore, you may avoid calling the "sample" function.
        # 
        # This function has the following arguments:
        # 1. the x-directional velocity field "u" in shape (res + 1, res);
        # 2. the y-directional velocity field "v" in shape (res, res + 1);
        # 3. the pressure field "p" in shape (res, res).
        #
        # Update the field "u" (1 point) and "v" (1 point) accordingly.
        # 
        ################################################################################
    
    @ti.func
    def momentum(self, mdir, dx, dy, dc):
        g_dir = -ti.Vector([0, 9.8]) * 3e2
        d2 = dx * dx + dy * dy
        factor = ti.exp(-d2 / force_radius)
        a = dc.norm()
        return (mdir * force_strength * factor + g_dir * a / (1 + a)) * dt
    
    @ti.kernel
    def apply_impulse(self, u: ti.template(), v: ti.template(), dyef: ti.template(), imp_data: ti.types.ndarray()):
        for i, j in dyef:
            omx, omy = imp_data[2], imp_data[3]
            mdir = ti.Vector([imp_data[0], imp_data[1]])
            dx, dy = (i + 0.5 - omx), (j + 0.5 - omy)
            ux, uy = i + 1.0, j + 0.5
            vx, vy = i + 0.5, j + 1.0
            u_dx, u_dy = dx + 0.5, dy
            v_dx, v_dy = dx, dy + 0.5
            d2 = dx * dx + dy * dy

            if i < res - 1:
                u[i + 1, j] += self.momentum(mdir, u_dx, u_dy, self.sample(dyef, tm.vec2(ux, uy), tm.vec2(0.5, 0.5)))[0]
            if j < res - 1:
                v[i, j + 1] += self.momentum(mdir, v_dx, v_dy, self.sample(dyef, tm.vec2(vx, vy), tm.vec2(0.5, 0.5)))[1]

            if mdir.norm() > 0.5:
                dyef[i, j] += ti.exp(-d2 * (4 / (res / 15) ** 2)) * ti.Vector([imp_data[4], imp_data[5], imp_data[6]])

    @ti.kernel
    def divergence(self, u: ti.template(), v: ti.template(), div: ti.template()):
        pass
        ################################################################################
        # Task 2.1 (2 points)
        ################################################################################
        # 
        # Compute the divergence of the velocity field. To zero our the divergence in
        # the projection step, we need to solve a Poisson equation whose right-hand-side
        # is the divergence of the velocity field.
        #
        # This function has the following arguments:
        # 1. the previous x-directional velocity field "u" in shape (res + 1, res);
        # 2. the previous y-directional velocity field "v" in shape (res, res + 1);
        # 3. the output divergence field "div" in shape (res, res);
        #
        ################################################################################
    
    @ti.kernel
    def vorticity(self, u: ti.template(), v: ti.template(), curl: ti.template()):
        for i, j in curl:
            x_dy = self.sample(u, tm.vec2(i + 1, j + 0.5), tm.vec2(0, 0.5)) - self.sample(u, tm.vec2(i, j + 0.5), tm.vec2(0, 0.5))
            y_dx = self.sample(v, tm.vec2(i + 0.5, j + 1), tm.vec2(0.5, 0)) - self.sample(v, tm.vec2(i + 0.5, j), tm.vec2(0.5, 0))
            curl[i, j] = x_dy - y_dx

    def step(self, gui):
        self.advect_dye(self.u.cur, self.v.cur, self.dye_buffer.cur, self.dye_buffer.nxt)
        self.advect_velocity(self.u.cur, self.v.cur, self.u.nxt, self.v.nxt)
        self.u.swap()
        self.v.swap()
        self.dye_buffer.swap()
        self.apply_impulse(self.u.cur, self.v.cur, self.dye_buffer.cur, self.mouse_data(gui))
        self.divergence(self.u.cur, self.v.cur, self.velocity_divs)
        self.vorticity(self.u.cur, self.v.cur, self.velocity_curls)
        self.solver.solve(self.velocity_divs, self.pressure)
        self.project(self.u.cur, self.v.cur, self.pressure.cur)
        if self.visual_mode == "d": # visualize dye (default)
            gui.set_image(self.dye_buffer.cur)
        elif self.visual_mode == "u": # visualize velocity x
            gui.set_image(self.u.cur.to_numpy()[:res, :res] * 0.01 + 0.5)
        elif self.visual_mode == "v": # visualize velocity y
            gui.set_image(self.v.cur.to_numpy()[:res, :res] * 0.01 + 0.5)
        elif self.visual_mode == "c": # visualize curl
            gui.set_image(self.velocity_curls.to_numpy() * 0.03 + 0.5)


if __name__ == "__main__":
    paused = False

    # Turn fast_gui off if you want to visualize velocities.
    gui = ti.GUI("Stable Fluid", (res, res), fast_gui=True)
    gui.fps_limit = fps
    sim = Simulator()
    sim.reset()

    while gui.running:
        if gui.get_event(ti.GUI.PRESS):
            e = gui.event
            if e.key == ti.GUI.ESCAPE:
                break
            elif e.key == "r":
                paused = False
                sim.reset()
            elif e.key in "uvdc":
                sim.visual_mode = e.key
            elif e.key == "p":
                paused = not paused

        if not paused:
            sim.step(gui)
        gui.show()
