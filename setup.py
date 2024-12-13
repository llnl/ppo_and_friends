from setuptools import setup, find_packages

version  = "1.0"
packages = find_packages()

package_dirs = {}
for pkg in packages:
    name               = "ppo_and_friends." + pkg
    directory          = pkg.replace(".", "/")
    package_dirs[name] = directory

package_dirs["ppo_and_friends"] = "."

dependencies = [
    'gymnasium',
    'pillow',
    'matplotlib',
    'plotly',
    'pandas',
    'kaleido',
    'opencv-python',
    'pygame',
    'box2d-py',
    'numpy<1.24',
    'dill',
    'mpi4py',
    'moviepy',
    'pyyaml',
    'torch>=1.10.2,<2.0',
    'pymunk',
    'packaging',
]

setup(name             = "ppo_and_friends",
      version          = version,
      description      = "Proximal Policy Optimization And Friends",
      author           = "Alister Maguire",
      license          = "MIT",
      package_dir      = package_dirs,
      packages         = list(package_dirs.keys()),
      package_data     = {"" : ["environments/abmarl/envs/maze.txt",
                                "environments/abmarl/envs/large_maze.txt"]},
      install_requires = dependencies,

      entry_points     = {
          'console_scripts' : ['ppoaf=ppo_and_friends.ppoaf_cli:cli']
      },

      extras_require = {
          'gym': ['gym==0.21.0', 'rware==1.0.3'],
          'gymnasium': ['gymnasium[mujoco]', 'gymnasium[atari]', 'autorom[accept-rom-license]', 'gym==0.23.0'],
          'minigrid': ['gymnasium', 'minigrid', 'gym==0.23.0'],
          'abmarl': ['abmarl', 'gym==0.23.0'],
          'pettingzoo': ['pettingzoo==1.23', 'gym==0.23.0'],
      },

      zip_safe         = False)
