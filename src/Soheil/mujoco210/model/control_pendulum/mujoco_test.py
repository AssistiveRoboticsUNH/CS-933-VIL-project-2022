"""
Shows how to toss a capsule to a container.
"""
from mujoco_py import load_model_from_path, MjSim, MjViewer
import os
import numpy as np

# Load the model and environment from its xml file
# model = load_model_from_path("/home/soheil/Sync/unh/courses/hri/project/src/Soheil/mujoco210/model/control_pendulum/pendulum.xml")
model = load_model_from_path("/home/soheil/Sync/unh/courses/hri/project/src/" +
                             "Soheil/mujoco210/model/threejoint.xml")
sim = MjSim(model)

# the time for each episode of the simulation
sim_horizon = 4000

# initialize the simulation visualization
viewer = MjViewer(sim)

# get initial state of simulation
sim_state = sim.get_state()
sim_state.qpos[0] = sim_state.qpos[0] + 1000 * np.random.rand()

# repeat indefinitely
while True:
    # set simulation to initial state
    sim.set_state(sim_state)

    # for the entire simulation horizon
    for i in range(sim_horizon):

        # trigger the lever within the 0 to 150 time period
        # if i < 150:
        #     sim.data.ctrl[:] = 0.0
        # else:
        #     sim.data.ctrl[:] = -1.0
        # import ipdb; ipdb.set_trace()
        states = sim.get_state()
        sim.data.ctrl[0] = -10 * (states.qpos[0] - 0) - 1 * states.qvel[0]
        # move one time step forward in simulation
        sim.step()
        viewer.render()

    if os.getenv('TESTING') is not None:
        break
