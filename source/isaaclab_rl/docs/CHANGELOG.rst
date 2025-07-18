Changelog
---------

0.1.8 (2025-06-29)
~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Support SB3 VecEnv wrapper to configure with composite observation spaces properly so that the cnn creation pipelines
  natively supported by sb3 can be automatically triggered


0.1.7 (2025-06-30)
~~~~~~~~~~~~~~~~~~

Fixed
^^^^^

* Call :meth:`eval` during :meth:`forward`` RSL-RL OnnxPolicyExporter


0.1.6 (2025-06-26)
~~~~~~~~~~~~~~~~~~

Fixed
^^^^^

* Relaxed upper range pin for protobuf python dependency for more permissive installation.


0.1.5 (2025-04-11)
~~~~~~~~~~~~~~~~~~

Changed
^^^^^^^
* Optimized Stable-Baselines3 wrapper ``Sb3VecEnvWrapper`` (now 4x faster) by using Numpy buffers and only logging episode and truncation information by default.
* Upgraded minimum SB3 version to 2.6.0 and added optional dependencies for progress bar

0.1.4 (2025-04-10)
~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added configurations for distillation implementation in RSL-RL.
* Added configuration for recurrent actor-critic in RSL-RL.


0.1.3 (2025-03-31)
~~~~~~~~~~~~~~~~~~

Fixed
^^^^^

* Fixed the location of :meth:`isaaclab_rl.rsl_rl.RslRlOnPolicyRunnerCfg._modify_action_space`
  to be called only after retrieving the dimensions of the environment, preventing errors
  related to accessing uninitialized attributes.


0.1.2 (2025-03-28)
~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added symmetry and curiosity-based exploration configurations for RSL-RL wrapper.


0.1.1 (2025-03-10)
~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added a parameter to clip the actions in the action space inside the RSL-RL wrapper.
  This parameter is set to None by default, which is the same as not clipping the actions.
* Added attribute :attr:`isaaclab_rl.rsl_rl.RslRlOnPolicyRunnerCfg.clip_actions` to set
  the clipping range for the actions in the RSL-RL on-policy runner.


0.1.0 (2024-12-27)
~~~~~~~~~~~~~~~~~~

Added
^^^^^

Initial version of the extension.
This extension is split off from ``isaaclab_tasks`` to include the wrapper scripts for the supported RL libraries.

Supported RL libraries are:

* RL Games
* RSL RL
* SKRL
* Stable Baselines3
