import pygame
import time
import math
from utils import scale_image, blit_rotate_center

GRASS = scale_image(pygame.image.load("C:/Users/Administrator/Desktop/AI assignment/Testing 1,2/imgs/imgs/grass.jpg"), 2.5)
TRACK = scale_image(pygame.image.load("C:/Users/Administrator/Desktop/AI assignment/Testing 1,2/imgs/imgs/track.png"), 0.9)
TRACK_BORDER = scale_image(pygame.image.load("C:/Users/Administrator/Desktop/AI assignment/Testing 1,2/imgs/imgs/track-border.png"), 0.9)
RED_CAR = scale_image(pygame.image.load("C:/Users/Administrator/Desktop/AI assignment/Testing 1,2/imgs/imgs/red-car.png"), 0.55)
GREEN_CAR = scale_image(pygame.image.load("C:/Users/Administrator/Desktop/AI assignment/Testing 1,2/imgs/imgs/green-car.png"), 0.55)

print("Images loaded:")
print(GRASS, TRACK, TRACK_BORDER, RED_CAR, GREEN_CAR)

WIDTH, HEIGHT = TRACK.get_width(), TRACK.get_height()
WIN = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Racing Game!")

FPS = 60

TRACK_BORDER_MASK = pygame.mask.from_surface(TRACK_BORDER)

class AbstractCar:
    def __init__(self, max_vel, rotation_vel):
        self.img = self.IMG
        self.max_vel = max_vel
        self.vel = 0
        self.rotation_vel = rotation_vel
        self.angle = 0
        self.x, self.y = self.START_POS
        self.acceleration = 0.1

    def rotate(self, left=False, right=False):
        if left:
            self.angle += self.rotation_vel
        elif right:
            self.angle -= self.rotation_vel

    def draw(self, win):
        blit_rotate_center(win, self.img, (self.x, self.y), self.angle)

    def move_forward(self):
        self.vel = min(self.vel + self.acceleration, self.max_vel)
        print(f"Moving forward: velocity = {self.vel}")  # Debugging line
        self.move()

    def move(self):
        radians = math.radians(self.angle)
        vertical = math.cos(radians) * self.vel
        horizontal = math.sin(radians) * self.vel

        self.y -= vertical
        self.x -= horizontal
        print(f"Updated position: x = {self.x}, y = {self.y}")  # Debugging line

    def reduce_speed(self):
        self.vel = max(self.vel - self.acceleration / 2, 0)
        self.move()

class PlayerCar(AbstractCar):
    IMG = RED_CAR
    START_POS = (180, 200)

    def __init__(self, max_vel, rotation_vel):
        super().__init__(max_vel, rotation_vel)
        print(f"Initial position: x = {self.x}, y = {self.y}")  # Debugging line

def draw(win, images, player_car):
    for img, pos in images:
        win.blit(img, pos)
    player_car.draw(win)
    pygame.display.update()

    # Check for collision
    if TRACK_BORDER_MASK.overlap(pygame.mask.from_surface(player_car.img), (int(player_car.x), int(player_car.y))):
        print("Collision Detected!")
        player_car.vel = 0  # Stop the car or handle collision appropriately

run = True
clock = pygame.time.Clock()
images = [(GRASS, (0, 0)), (TRACK, (0, 0)), (TRACK_BORDER, (0, 0))]
player_car = PlayerCar(4, 4)

while run:
    clock.tick(FPS)
    draw(WIN, images, player_car)
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            run = False
            break

    keys = pygame.key.get_pressed()
    moved = False
    if keys[pygame.K_a]:
        player_car.rotate(left=True)
        print("A key pressed")  # Debugging line
    if keys[pygame.K_d]:
        player_car.rotate(right=True)
        print("D key pressed")  # Debugging line
    if keys[pygame.K_w]:
        print("W key pressed")  # Debugging line
        moved = True
        player_car.move_forward()

    if not moved:
        player_car.reduce_speed()
    print(f"Velocity: {player_car.vel}")  # Debugging line

pygame.quit()



