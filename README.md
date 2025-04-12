# Physics-Based Simulation Homework: Multigrid Stable Fluids

This homework is developed by Kangbo Lyu (Tsinghua University).

## Installation Guide
- Use git to clone the codebase:
```
git clone --recursive https://github.com/taodu-eecs/phys_sim_multigrid_stable_fluids.git
```
- Install [Taichi Lang](https://github.com/taichi-dev/taichi) on your computer:
```
pip install --upgrade taichi
```
- Run the homework with the following command:
```
python fluid_hw.py
```

## Programming Tasks
This programming assignment invites you to implement a few key functions in a real-time stable fluid program. The fluid domain is relatively simple: it is a 2D square full of fluid surrounded by free-slip, no-separation boundaries at the four static (zero velocity) edges of the square. In particular, you will implement two Poisson solvers: one based on Jacobi iterations and one based on multigrids. The code skeleton runs the script on GPUs (line 5 in `fluid_hw.py`):
```
ti.init(arch=ti.cuda, default_fp=ti.f32, fast_math=True, debug=False)
```
If you do not have access to a GPU, feel free to replace `ti.cuda` with `ti.cpu`. The choice of CPU/GPU architecture won't affect our grading of your homework.

### Task 1 (8 points)
Implement the semi-Lagrangian advection algorithm split in three functions:
- Task 1.1 (3 points): Read the instructions in `sample` to implement bilinear interpolation in a scalar/vector field with the (clamped) coordinates `rel_pos`.
- Task 1.2 (3 points): Implement the Runge-Kutta 2 (RK2) time integration in `backtrace`. Note that this function computes the position *backward* in time.
- Task 1.3 (2 points): Use the two functions above to implement semi-Lagrangian advection in `advect_velocity`.

### Task 2 (6 points)
Next, we invite you to implement the Poisson solver in stable fluids split in three functions:
- Task 2.1 (2 points): Implement the `divergence` function to compute the divergence of a given velocity field. We will need the divergence value in the right-hand side of the Poisson equation.
- Task 2.2 (2 points): Read the instructions in `iteration` and implement a modified Jacobi iteration in this function. It is also helpful to read the `solve` function that calls `iteration`.
- Task 2.3 (2 points): Use the two functions above as building blocks to finish the `project` function.

Running `fluid_hw.py` after finishing Tasks 1 and 2 should show an interactive stable fluid demo in a squared domain, thanks to the high-performance computational power offered by Taichi Lang. Read the GUI code in `fluid_hw.py` to figure out how to interact with this fluid demo. The frame rate of your demo depends on your hardware platform and CPU/GPU architecture.

The Poisson solver in stable fluids typically consumes a substantial portion of the whole computational time. In the next task, you will implement a few functions in `PoissonMultiGridSolver` class and compare the performance of the multi-grid preconditioned conjugate gradient (MGPCG) solver with the previous Jacobi solver.

### Task 3 (6 points)
We highly encourage you to read the full `PoissonMultiGridSolver` class carefully before you start working on the three tasks below.
- Task 3.1 (2 points): Implement `smooth`, the smoothing operator used in the V-cycle scheme. It may be helpful to skim over functions calling `smooth` and get a sense of its inputs and outputs before your start coding;
- Task 3.2 (2 points): Implement `prolongate`, the prolongation/interpolation operator.
- Task 3.3 (2 points): Implement `restrict`, the restriction operator.
- Task 3.4 (0 points): Try to run `fluid_hw.py` with Jacobi and MGPCG Poisson solvers, respectively, and compare their frame rates on your computer. Feel free to adjust the solver's tolerance and the fluid domain's resolution and repeat the comparison. What is your feeling about the computational speed of these two solvers?

Implementing a correct multigrid can be tricky. If you struggle with Task 3, please feel free to swing by our office hours for help.

## Submission Guide
Submit your `fluid_hw.py` file to [Web Learning](https://learn.tsinghua.edu.cn/). **It is your responsibility to double check** that your submission file is correct. Email your `fluid_hw.py` to taodu@tsinghua.edu.cn or our Teaching Assistant if you miss the deadline but still have late days left. We do not accept submissions in any other formats.

## Acknowledgments
The code skeleton is from the [stable_fluid.py](https://github.com/taichi-dev/taichi/blob/master/python/taichi/examples/simulation/stable_fluid.py) example in [Taichi Lang](https://github.com/taichi-dev/taichi/tree/master). Feel free to read their implementation before you start working on this homework, but you should not **copy and paste** code from it. In fact, we have made substantial changes in `fluid_hw.py` so that directly copying the implementation in `stable_fluid.py` may not lead to a correct solution to our homework.
