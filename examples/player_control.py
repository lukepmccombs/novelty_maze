import pygame as pg
from novelty_maze import MazeEnvironment, HARD_MAZE


if __name__ == "__main__":
    env = MazeEnvironment(HARD_MAZE)

    screen = pg.display.set_mode((300, 300))
    clock = pg.time.Clock()

    fb = 0
    lr = 0
    while True:

        for event in pg.event.get():
            if event.type == pg.QUIT:
                exit(0)
            
            if event.type == pg.KEYDOWN or event.type == pg.KEYUP:
                direction = 1 - 2 * (event.type == pg.KEYUP)
                if event.key == pg.K_UP:
                    fb += direction
                if event.key == pg.K_DOWN:
                    fb -= direction
                if event.key == pg.K_RIGHT:
                    lr += direction
                if event.key == pg.K_LEFT:
                    lr -= direction
        
        env.update(((fb+1)/2, (lr+1)/2))
        env.draw(screen)
        pg.display.flip()
        clock.tick(10)