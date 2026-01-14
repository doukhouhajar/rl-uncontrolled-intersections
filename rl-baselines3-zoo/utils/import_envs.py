try:
    import pybullet_envs  # pytype: disable=import-error
except ImportError:
    pybullet_envs = None

try:
    import highway_env  # pytype: disable=import-error
except ImportError:
    highway_env = None

try:
    import neck_rl  # pytype: disable=import-error
except ImportError:
    neck_rl = None

try:
    import mocca_envs  # pytype: disable=import-error
except ImportError:
    mocca_envs = None

try:
    import custom_envs  # pytype: disable=import-error
except ImportError:
    custom_envs = None

try:
    import gym_donkeycar  # pytype: disable=import-error
except ImportError:
    gym_donkeycar = None

try:
    import rl_racing.envs  # pytype: disable=import-error
except ImportError:
    rl_racing = None

try:
    import gym_space_engineers  # pytype: disable=import-error
except ImportError:
    gym_space_engineers = None

try:
    import panda_gym  # pytype: disable=import-error
except ImportError:
    panda_gym = None

# -----------------------------
# Manual Donkey registration (Gymnasium compatible)
# -----------------------------

if gym_donkeycar is not None:
    try:
        import gym
        import gymnasium
        from gymnasium.envs.registration import register

        DONKEY_ENVS = [
            "donkey-mountain-track-v0",
            "donkey-generated-track-v0",
        ]

        for env_id in DONKEY_ENVS:
            spec = gym.spec(env_id)

            # register versioned
            try:
                gymnasium.spec(env_id)
            except Exception:
                register(
                    id=env_id,
                    entry_point=spec.entry_point,
                    kwargs=spec.kwargs,
                    max_episode_steps=spec.max_episode_steps,
                )

            # register base name
            base_id = env_id.replace("-v0", "")
            try:
                gymnasium.spec(base_id)
            except Exception:
                register(
                    id=base_id,
                    entry_point=spec.entry_point,
                    kwargs=spec.kwargs,
                    max_episode_steps=spec.max_episode_steps,
                )

        print("[import_envs] Donkey envs registered for Gymnasium")

    except Exception as e:
        print("[import_envs] Registration failed:", e)
