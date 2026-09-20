import pygame
import time

pygame.mixer.pre_init(44100, -16, 2, 2048)
pygame.mixer.init()
print('mixer', pygame.mixer.get_init())
snd = pygame.mixer.Sound('voices/21.mp3')
print('loaded')
snd.play()
time.sleep(2)
print('played')
