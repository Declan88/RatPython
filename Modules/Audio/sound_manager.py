import numpy as np
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

    INVERSE_EXPONENT = 0.7
    # (distance in metres up to which this applies, low-pass cutoff in Hz or
    # None for the untouched sound), nearest first.
    MUFFLE_TIERS = ((20.0, None), (40.0, 9000.0), (70.0, 6000.0), (110.0, 3500.0),
                    (160.0, 2200.0), (float("inf"), 1400.0))

    def __init__(self):
        self.emitters = []
        self._listener = None   # (position, right vector) from the last update()
        self._muffled = {}      # (path, cutoff Hz) -> low-passed sample array

    @classmethod
    def _muffle_cutoff(cls, dist):
        return next(cutoff for limit, cutoff in cls.MUFFLE_TIERS if dist <= limit)

    def _load_sound(self, path, cutoff):
        sound = pygame.mixer.Sound(path)
        if cutoff is None:
            return sound
        key = (path, cutoff)
        samples = self._muffled.get(key)
        if samples is None:
            samples = self._muffled[key] = self._lowpass(pygame.sndarray.array(sound), cutoff)
        return pygame.sndarray.make_sound(samples)

    @staticmethod
    def _lowpass(samples, cutoff):
        """Zero-phase low-pass of a (frames, channels) integer sample array:
        a smooth 4th-order roll-off applied in the frequency domain."""
        rate = pygame.mixer.get_init()[0]
        spectrum = np.fft.rfft(samples.astype(np.float64), axis=0)
        freqs = np.fft.rfftfreq(samples.shape[0], 1.0 / rate)
        spectrum *= (1.0 / (1.0 + (freqs / cutoff) ** 4))[:, None]
        out = np.fft.irfft(spectrum, n=samples.shape[0], axis=0)
        info = np.iinfo(samples.dtype)
        return np.clip(out, info.min, info.max).astype(samples.dtype)

    def add_sound(
        self,
        sound_path,
        position,
        volume=1.0,
        min_distance=1.0,
        max_distance=20.0,
        loop=True,
        universal=False,
        follow=None,
        channel=None,
        falloff="legacy",
        muffle=False,
    ):
        """falloff="inverse" swaps the flat point-light-style curve for a
        realistic-for-a-game one: full volume inside min_distance, then
        amplitude (min_distance / distance) ** INVERSE_EXPONENT - roughly -4 dB
        per doubling of distance - fading out over the last quarter of
        max_distance.

        muffle=True (a one-shot only) plays a low-passed copy of the sound
        picked by how far away it starts (see MUFFLE_TIERS): air soaks up
        the highs first, so a distant gunshot loses its crack and is left as a
        dull thump. The mixer can't filter live, so each cutoff is rendered
        once per file and cached.

        loop=True starts it as a looping ambient sound immediately
        (e.g. a hum, a fire crackling); loop=False plays it once and
        the emitter is automatically dropped once it finishes.

        universal=True skips distance attenuation and stereo panning
        entirely - the sound plays at a flat `volume` in both ears
        regardless of the listener's position/facing (e.g. music, UI
        sounds, a global ambience bed). position is still required but
        ignored in this mode.

        follow: optional callable returning the emitter's CURRENT world
        position, polled every update() while it plays - a sound that's
        attached to something that moves (a gunshot ringing out of a moving
        player) instead of staying where it started. `position` is where it
        begins.

        channel: the channel an earlier call returned (its emitter's "channel")
        to play on again instead of whichever is free: whatever is still
        playing on it is cut off, its emitter dropped, and this sound takes
        its place. For a sound that repeats faster than it ends (a gunshot) -
        the old one is killed and the new one always plays, rather than
        needing a free channel from the pool each time, which with the mixer
        busy could be none."""
        if not pygame.mixer.get_init():
            pygame.mixer.init()
            pygame.mixer.set_num_channels(32)

        cutoff = None
        if muffle and not universal and self._listener is not None:
            cutoff = self._muffle_cutoff(glm.length(glm.vec3(position) - self._listener[0]))
        sound = self._load_sound(sound_path, cutoff)
        sound.set_volume(volume)
        if channel is not None:
            self.emitters = [e for e in self.emitters if e["channel"] is not channel]
            channel.play(sound, loops=-1 if loop else 0)
        else:
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
            "universal": bool(universal),
            "follow": follow,
            "falloff": falloff,
        }
        # Attenuate NOW, from where the listener was at the last update(): a
        # new sound otherwise plays at full volume, dead centre, until the next
        # update() runs - for a gunshot that's its loudest moment (the attack)
        # coming out un-attenuated, e.g. a far-away player's shot arriving over
        # the network after this frame's update() had already run.
        if channel is not None and self._listener is not None:
            self._apply(emitter, *self._listener)
        self.emitters.append(emitter)
        return emitter

    @staticmethod
    def _apply(emitter, listener_pos, right_vec):
        """Sets the emitter's channel volume (distance falloff + L/R pan)."""
        if emitter["follow"] is not None:
            emitter["position"] = glm.vec3(emitter["follow"]())

        if emitter["universal"]:
            # No distance attenuation or panning - flat volume in
            # both ears regardless of listener position/facing.
            left_volume = right_volume = emitter["volume"]
        else:
            to_emitter = emitter["position"] - listener_pos
            dist = glm.length(to_emitter)
            direction = to_emitter / max(dist, 0.0001)

            radius = max(emitter["max_distance"], 0.01)
            effective_dist = max(dist, emitter["min_distance"])
            if emitter["falloff"] == "inverse":
                atten = (emitter["min_distance"] / effective_dist) ** SoundManager.INVERSE_EXPONENT
                atten *= max(0.0, min(1.0, (radius - dist) / (0.25 * radius)))
            else:
                falloff = max(0.0, min(1.0, 1.0 - (effective_dist / radius) ** 4))
                atten = falloff * falloff

            pan = max(-1.0, min(1.0, glm.dot(direction, right_vec)))
            base_volume = emitter["volume"] * atten
            left_volume = base_volume * (1.0 - max(0.0, pan))
            right_volume = base_volume * (1.0 + min(0.0, pan))

        emitter["channel"].set_volume(left_volume, right_volume)

    def update(self, camera):
        """Call once per frame, passing the Camera as the listener."""
        right_vec = glm.normalize(glm.cross(camera.front, camera.up))
        listener_pos = glm.vec3(camera.position)
        self._listener = (listener_pos, right_vec)
        if not self.emitters:
            return

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

            self._apply(emitter, listener_pos, right_vec)
            still_active.append(emitter)

        self.emitters = still_active

    def pause_all(self):
        """Pauses every currently-playing emitter's own mixer channel,
        WITHOUT stopping/clearing them (see resume_all) - for a Scene
        that's been switched away from but still exists (app.py keeps
        every Scene constructed for the lifetime of the app, not just
        the active one - see its own scene-switching code), so its
        emitters (including a looping ambient sound started once in
        __init__, never re-triggered) are still there to resume from
        exactly where they left off if switched back to, rather than
        needing to restart from the beginning or never play again.
        Without this, a Scene's own looping sounds keep playing
        completely independently of whether that Scene is the one
        actually being rendered/updated - pygame mixer channels have no
        concept of "which scene is active" on their own, and this
        project's own per-frame SoundManager.update() (which is what
        keeps a still-active emitter's volume/panning current) simply
        isn't called for an inactive Scene at all, which stops it being
        updated but was never enough to stop it being HEARD."""
        for emitter in self.emitters:
            channel = emitter["channel"]
            if channel is not None:
                channel.pause()

    def resume_all(self):
        """Undoes pause_all - see that method's own docstring."""
        for emitter in self.emitters:
            channel = emitter["channel"]
            if channel is not None:
                channel.unpause()

    def destroy(self):
        for emitter in self.emitters:
            try:
                if emitter["channel"] is not None:
                    emitter["channel"].stop()
            except Exception:
                pass
        self.emitters.clear()
