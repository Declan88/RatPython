import glm
import pygame

class Camera:
    def __init__(self, position=(0.0, 0.0, 3.0), fov=45.0, aspect=800/600):
        self.position = glm.vec3(position)
        self.pitch = 0.0
        self.yaw = -90.0
        self.front = glm.vec3(0.0, 0.0, -1.0)
        self.up = glm.vec3(0.0, 1.0, 0.0)
        self.fov = fov
        self.aspect = aspect
        self.speed = 7
        self.sensitivity = 0.1

    def get_view_matrix(self):
        return glm.lookAt(self.position, self.position + self.front, self.up)

    def get_projection_matrix(self):
        return glm.perspective(glm.radians(self.fov), self.aspect, 0.1, 100.0)

    def get_matrix(self):
        return self.get_projection_matrix() * self.get_view_matrix()

    def process_keyboard(self, keys, dt):
        velocity = self.speed * dt
        if keys[pygame.K_w]:
            self.position += self.front * velocity
        if keys[pygame.K_s]:
            self.position -= self.front * velocity
        if keys[pygame.K_a]:
            self.position -= glm.normalize(glm.cross(self.front, self.up)) * velocity
        if keys[pygame.K_d]:
            self.position += glm.normalize(glm.cross(self.front, self.up)) * velocity

    def process_mouse(self, xoffset, yoffset):
        xoffset *= self.sensitivity
        yoffset *= self.sensitivity

        self.yaw += xoffset
        self.pitch -= yoffset  # Reversed since screen coordinates start top-left

        # Clamp pitch to prevent camera flipping upside down
        if self.pitch > 89.0:
            self.pitch = 89.0
        if self.pitch < -89.0:
            self.pitch = -89.0

        # Calculate new front vector
        front = glm.vec3()
        front.x = glm.cos(glm.radians(self.yaw)) * glm.cos(glm.radians(self.pitch))
        front.y = glm.sin(glm.radians(self.pitch))
        front.z = glm.sin(glm.radians(self.yaw)) * glm.cos(glm.radians(self.pitch))
        self.front = glm.normalize(front)