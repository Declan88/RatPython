import pygame
import glm


class SoundManager:
    """Positional sound emitters: distance-based volume attenuation plus
    simple left/right stereo panning based on the emitter's position
    relative to the listener's facing direction, recomputed every frame
    in update(). This isn't true 3D/HRTF spatialization - pygame's mixer
    can't do that - but it's genuinely what most games mean by "3D
    sound" in practice.

    Distance falloff uses the same curve as Scene's point lights
    (add_point_light) for a consistent feel between light and sound
    falloff across the same max_distance/radius-style parameter.

    Not tied to any particular Scene - create one instance (e.g. in
    app.py, alongside NetworkManager) and call update() once per frame
    with the Camera as the listener.
    """

    def __init__(self):
        self.emitters = []

    def add_sound(
        self,
        sound_path,
        position,
        volume=1.0,
        min_distance=1.0,
        max_distance=20.0,
        loop=True,
    ):
        """loop=True starts it as a looping ambient sound immediately
        (e.g. a hum, a fire crackling); loop=False plays it once and
        the emitter is automatically dropped once it finishes."""
        if not pygame.mixer.get_init():
            pygame.mixer.init()
            pygame.mixer.set_num_channels(32)

        sound = pygame.mixer.Sound(sound_path)
        sound.set_volume(volume)
        channel = sound.play(loops=-1 if loop else 0)

        if channel is None:
            print(f"[SoundManager] No free mixer channel for sound: {sound_path}")

        emitter = {
            "sound": sound,
            "channel": channel,
            "position": glm.vec3(position),
            "volume": float(volume),
            "min_distance": float(min_distance),
            "max_distance": float(max_distance),
            "loop": bool(loop),
        }
        self.emitters.append(emitter)
        return emitter

    def update(self, camera):
        """Call once per frame, passing the Camera as the listener."""
        # print(
        #     f"[SoundManager] update() called, {len(self.emitters)} emitter(s)"
        # )  # TEMP DEBUG
        if not self.emitters:
            return

        right_vec = glm.normalize(glm.cross(camera.front, camera.up))
        listener_pos = camera.position

        still_active = []
        for emitter in self.emitters:
            channel = emitter["channel"]
            if channel is None or not channel.get_busy():
                # One-shot finished, or never got a free channel - drop
                # it rather than continuing to hold/update a stale
                # Channel object (pygame can reuse a finished channel
                # for an unrelated sound, and calling .set_volume() on
                # it afterward would affect that different sound).
                continue

            to_emitter = emitter["position"] - listener_pos
            dist = glm.length(to_emitter)
            direction = to_emitter / max(dist, 0.0001)

            radius = max(emitter["max_distance"], 0.01)
            effective_dist = max(dist, emitter["min_distance"])
            falloff = max(0.0, min(1.0, 1.0 - (effective_dist / radius) ** 4))
            atten = falloff * falloff

            pan = max(-1.0, min(1.0, glm.dot(direction, right_vec)))
            base_volume = emitter["volume"] * atten
            left_volume = base_volume * (1.0 - max(0.0, pan))
            right_volume = base_volume * (1.0 + min(0.0, pan))

            # print(
            #     f"[SoundManager] dist={dist:.2f} atten={atten:.3f} L={left_volume:.3f} R={right_volume:.3f}"
            # )  # TEMP DEBUG - remove once confirmed working

            channel.set_volume(left_volume, right_volume)
            readback = channel.get_volume()
            # print(f"[SoundManager] readback after set_volume: {readback}")  # TEMP DEBUG
            still_active.append(emitter)

        self.emitters = still_active

    def destroy(self):
        for emitter in self.emitters:
            try:
                if emitter["channel"] is not None:
                    emitter["channel"].stop()
            except Exception:
                pass
        self.emitters.clear()
