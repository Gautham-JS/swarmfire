"""
redis_rendered_env.py

Subclasses your existing SingleAgentEnv and overrides only the render-path
methods (render, reset, close) to publish to Redis instead of drawing
matplotlib figures / writing video in-process. Nothing about reward
calculation, stepping, or revisit tracking is touched - all of that is
inherited as-is.

Because render() is overridden here, SingleAgentEnv._step()'s call to
`self.render()` resolves to *this* class's version via normal polymorphism -
no changes needed in the base class for that to work.

Usage:
    from your_module import SingleAgentEnv   # wherever the original lives
    from redis_rendered_env import RedisRenderedEnv

    env = RedisRenderedEnv(
        world_size=(512, 512),
        video_conf=video_conf,
        redis_port=8090,
        ...  # all your usual SingleAgentEnv kwargs
    )
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')

from clients.RedisClient import RedisRenderPublisher
from envs.WildfireSingleAgentEnv import SingleAgentEnv


class RedisRenderedEnv(SingleAgentEnv):
    def __init__(
        self,
        *args,
        enable_redis_render: bool = True,
        redis_host: str = "localhost",
        redis_port: int = 8090,
        redis_db: int = 0,
        redis_channel_prefix: str = "render",
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        # Publisher degrades to a no-op internally if redis is unreachable,
        # so this is safe to construct unconditionally.
        self._render_pub = RedisRenderPublisher(
            host=redis_host,
            port=redis_port,
            db=redis_db,
            channel_prefix=redis_channel_prefix,
        ) if enable_redis_render else None

        # Note: self._fig / self._axes / self.out from the base class are
        # never touched by this subclass (render() is fully overridden), so
        # no matplotlib window or cv2.VideoWriter is ever created in this
        # process - the base class's own cleanup code for those in reset()/
        # close() just stays a no-op since they're never populated.

    # ------------------------------------------------------------------
    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)

        if self._render_pub is not None:
            self._render_pub.publish_episode_start(
                env_id=self.env_id,
                episode_count=self._episode_count,
                world_size=self.world_size,
                vp_size=self.vp_size,
                render_mode=self.render_mode,
                video_config=self.video_config,
            )

        return obs, info

    # ------------------------------------------------------------------
    def render(self):
        """
        Overrides the base class's matplotlib rendering entirely. Gathers
        everything render_subscriber.py needs for one frame and publishes
        it to redis - no figure is ever created in this process.

        Note: unlike the base implementation, this always returns None -
        rgb_array mode no longer hands back a synchronous numpy frame, since
        drawing now happens out-of-process. Pull frames from the mp4 that
        render_subscriber.py writes if you need them after the fact.
        """
        if self.render_mode not in ["human", "rgb_array"]:
            return

        if self._render_pub is None:
            return

        last_pos = self._positions_history[-1] if len(self._positions_history) > 0 else self.start_poss

        gt_map = None
        if self.is_gt_visible:
            gt_map = self.agent_instance.get_GT_map()

        accumulated_scene = self.view_acc.get_scene()

        latest_obs_chw = (
            self._observation_history[-1]["viewport"]
            if len(self._observation_history) > 0
            else np.zeros((self._n_obs_channels, 84, 84), dtype=np.float32)
        )

        latest_obs_chw[0, :, :] = np.zeros((84, 84), dtype=np.float32)

        global_recency_map = self.agent_instance.get_recency_map()

        self._render_pub.publish_frame(
            env_id=self.env_id,
            episode_count=self._episode_count,
            step_count=self._step_count,
            world_size=self.world_size,
            vp_size=self.vp_size,
            render_mode=self.render_mode,
            video_config={
                "is_enabled": self.video_config.is_enabled,
                "base_path": self.video_config.base_path,
                "fps": self.video_config.fps,
                "sample_interval": self.video_config.sample_interval,
            } if self.video_config is not None else None,
            is_gt_visible=self.is_gt_visible,
            gt_map=gt_map,
            accumulated_scene=accumulated_scene,
            latest_obs_chw=latest_obs_chw,
            global_recency_map=global_recency_map,
            last_pos=last_pos,
            positions_history=list(self._positions_history),
            reward_history=list(self._reward_history),
            reward_components={k: list(v) for k, v in self._reward_components.items()},
        )
        return None

    # ------------------------------------------------------------------
    def close(self):
        if self._render_pub is not None:
            self._render_pub.publish_close(self.env_id)
            self._render_pub.close()

        if not self.is_ue5_mode and self.agent_instance is not None:
            self.agent_instance.close()