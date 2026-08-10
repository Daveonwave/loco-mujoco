import numpy as np
from loco_mujoco.task_factories import RLFactory

# Create the environment. The factory will find "AgibotX2" because 
# you registered it inside the internal library structure.
env = RLFactory.make("AgibotX2")

action_dim = env.info.action_space.shape[0]
print(f"AgibotX2 loaded successfully with {action_dim} actuators!")

env.reset()
env.render()

i, absorbing = 0, False

# Main simulation loop
while True:
    if i == 1000 or absorbing:
        env.reset()
        i = 0
    
    # Send random noise to the actuators
    action = np.random.randn(action_dim)
    
    # Step the simulation
    obs, reward, absorbing, done, info = env.step(action)
    env.render()
    
    i += 1