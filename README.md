## What is tinycarlo?

tinycarlo is a 2D simulation written in python as an [OpenAI Gym](https://gym.openai.com/). It can be used to test algorithms for autonomous driving with a given perception system. Reinforcement Learning Algorithms can also be trained with this simulation since the simulation already has an integrated reward system. Currently, only kinematics are used and dynamics are not taken into account, therefore the car has no slip. As input, cameras are simulated. These cameras are in birds-eye-view. 

To give you an idea for what it can be used, here are 3 ideas:

#### Segmented road markings

We assume the road marking detection is already given and it segments these markings. The color describes different classes like dashed or solid lines. 

An example track can be found [here](/example_tracks/segmented/).

#### Formula Student

In formula student competitions, color coded cones are used to define the track. Using colored circles on the track image, these tracks can be simulated as well. Further processing is needed if your driverless system is not using images.

An example track can be found [here](/example_tracks/formula_student/).

#### Carolo-Cup

At the Carolo-Cup you have white road markings on a black surface. So the perception can also be tested with this simulation. To use this sim for your system, you can create a [custom track](#Custom-Tracks).



## Installation

To get started, you'll need to have Python 3.9+ installed. You can use pip to install.

```
pip install 
```

Please note that `gym` is already a dependency, so no need to install that too if not already done. 



## Usage

This simulation can be used like every other [OpenAI Gym](https://gym.openai.com/). Here is a very basic example

```python
import gym
import tinycarlo

env = gym.make("tinycarlo-v0", environment="./example_tracks/formula_student")

observation = env.reset()
while True:
    env.render()
    action = env.action_space.sample()
    observation, reward, done, info = env.step(action)
    if done:
        observation = env.reset()
        break
env.close()
```

To configure the simulation, take a look on [Custom Tracks](#Custom-Tracks). 

#### Action Space

The action space is a single floating point number with a range of `[-1.0, 1.0]`. This the normalized steering angle for the car.

#### Observation Space

The observation space is a dictionary, depending on how many cameras are configured. The value of every key-value pair is the camera image in a uint8 BGR format. 

## Custom Tracks

To create a custom track, create a new folder with a `track.yaml` file. The path of the folder is then provided in the `gym.make("tinycarlo-v0", environment="./path/to/your/folder")` call. 

For the structure of the `track.yaml` file, please use on of the example tracks: [example 1](/example_tracks/segmented/track.yaml) or [example 2](/example_tracks/formula_student/track.yaml). Below you can find a detailed descriptions of the parameters:

#### sim

| Key                | Type  | Default | Description                                                  |
| ------------------ | ----- | ------- | ------------------------------------------------------------ |
| fps                | int   | 30      | Defines the step size of the simulation                      |
| render_realtime    | bool  | True    | Does only affect the visualization. When using reinforcment learning it can be useful to no render everything in realtime to reduce training time. Step size of simulation is unaffected. |
| step_limit         | int   | None    | If the simulation has reached this limit, the done flag is automatically set. Useful in reinforcement learning |
| overview_downscale | float | 1       | This factor is used to divide the track image in the visualization screen. A value greater than 1 resizes the visualization to make the rendering faster since the full resolution is normally overkill for an overview. |

#### reward_design

| Key               | Type | Default | Description                                                  |
| ----------------- | ---- | ------- | ------------------------------------------------------------ |
| color_obstacles   | list | None    | List of obstacles. If the car hits one of these obstacles, a given reward is returned. An obstacle is defined by pixel colors. See [more](#color_obstacle) |
| cross_track_error | dict | None    | Information if a crosst track error should be used and where it is defined. See [more](#cross_track_error) |

#### color_obstacle

| Key    | Type          | Description                                                  |
| ------ | ------------- | ------------------------------------------------------------ |
| color  | [int,int,int] | Defines color of an obstacle in BGR format. Please use only 0 or 255, since everything will be binarized. |
| reward | int or `done` | If an int is provided, this will be returned as an reward upon collision. If `done` as a String is set, the done flag on the step() function will be set, resetting the simulation run. |

#### cross_track_error

| Key              | Type            | Default       | Description                                                  |
| ---------------- | --------------- | ------------- | ------------------------------------------------------------ |
| use_cte          | bool            | False         | Whether to use cross track error or no. If it is True, the next parameter needs to be set! |
| trajectory_color | [int, int, int] | [255,255,255] | Defines the color of the ground truth trajectory used for the cte calculation. This does not need to be a fully connected line in the track image. Multiple colored pixels are fine! Please use only 0 or 255 in BGR format. |

#### car

| Key                 | Type         | Default | Description                                                  |
| ------------------- | ------------ | ------- | ------------------------------------------------------------ |
| wheelbase           | int          | 160     | in pixels                                                    |
| track_width         | int          | 100     | in pixels                                                    |
| velocity            | float or int | 500     | pixels per second                                            |
| max_steering_change | int          | None    | limits how fast the car can change it's steering angle. This depends on human performance or servo. if none, the steering angle can change completely between to sim steps. |

#### camera

| Key        | Type       | Description                                                  |
| ---------- | ---------- | ------------------------------------------------------------ |
| id         | String     | ID of a camera. Only used as a key for the observation space. |
| resolution | [int, int] | [y,x] resolution of camera. This does not affect the viewing space on track. If this is the same as the roi, np resize will be used. This should be equal or less than the roi to avoid weird upsampling. |
| roi        | [int, int] | [y,x] region of interest of camera. This defines the viewing space of the camera on the track image. |
| position   | [int, int] | [y,x] position of the camera relative to the middle of the front axle. |

#### track

| Key    | Type                          | Description                                                  |
| ------ | ----------------------------- | ------------------------------------------------------------ |
| image  | String                        | relative path to the track image based on the track.yaml file. |
| spawns | list of [float, float, float] | list of spawn points. A spawn point is defined by x,y,alpha. x and y position is relative to the track image, therefore in pixels, and alpha is the direction the car is heading. Alpha is in radians and the 0 point is north (top of the track image). |

