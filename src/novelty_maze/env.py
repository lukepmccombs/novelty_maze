from .point_utils import *

from itertools import islice, repeat, pairwise
import os
import numpy as np


_root_mazes = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mazes")
MEDIUM_MAZE = os.path.join(_root_mazes, "medium.txt")
HARD_MAZE = os.path.join(_root_mazes, "hard.txt")

def _number_stream(fp):
    with open(fp, "r") as n_file:
        for n in n_file.read().split():
            yield int(n)

def _extract_point(ns):
    arr = np.array(list(islice(ns, 2)), dtype=np.float64)
    if len(arr) == 2:
        return arr


class Character:

    # From NoveltySearch
    ranger_angles = (-90, -45, 0, 45, 90, -180)
    radar_angles = (-45, 45, 135, 225, 315, 405) # wrap around values on both sides, but they correspond to the same
    ranger_range = 100

    max_speed = 3
    max_ang_vel = 3

    def __init__(self, start_location, start_heading):
        self._last_location = start_location.copy()
        self.location = start_location.copy()
        self.heading = start_heading

        self.speed = 0
        self.ang_vel = 0

        self.rangers = np.full(len(self.ranger_angles), self.ranger_range, dtype=np.float64)
        self.radar = np.zeros(len(self.radar_angles) - 2)
    
    def reset_observations(self):
        self.rangers.fill(self.ranger_range)
        self.radar.fill(0)
    
    def get_observation(self):
        return np.concatenate([
                [1],
                self.rangers / self.ranger_range,
                self.radar
            ])

    def apply_actions(self, action):
        speed_act, ang_act = action
        self.speed = np.clip(
                self.speed + np.clip(speed_act, 0, 1) - 0.5,
                -self.max_speed, self.max_speed
            )
        self.ang_vel = np.clip(
                 self.ang_vel + np.clip(ang_act, 0, 1) - 0.5,
                -self.max_ang_vel, self.max_ang_vel
            )
    
    def update(self):
        vel = rotate_vector(np.array([self.speed, 0]), self.heading)
        self._last_location = self.location.copy()
        self.location += vel
        self.heading += self.ang_vel
        return self.location

    def reset_location(self):
        self.location = self._last_location

    def get_ranger_lines(self):
        dest_points = []
        for angle in self.ranger_angles:
            off = rotate_vector([self.ranger_range, 0], angle + self.heading)
            dest_points.append(self.location + off)
        return list(zip(repeat(self.location), dest_points))

    def update_radar(self, target_heading):
        target_heading %= 360
        for i, (left, right) in enumerate(pairwise(self.radar_angles)):
            if target_heading >= left and target_heading < right:
                self.radar[i % len(self.radar)] = 1
                break


class MazeEnvironment:

    # From NoveltySearch
    end_radius = 5
    collide_radius = 8

    def __init__(self, fp):

        ns = _number_stream(fp)
        _ = next(ns) # Unneeded num_lines value
        
        self.start_location = _extract_point(ns)
        self.start_heading = next(ns)

        self.hero = Character(self.start_location, self.start_heading)
        self.end = _extract_point(ns)

        self.lines = []
        while True:
            line = (_extract_point(ns), _extract_point(ns))
            if line[0] is None:
                break
            self.lines.append(line)

        self.reach_goal = False

        self.hero.reset_observations()
        self.update_rangefinders()
        self.update_radar()
    
    def reset(self):
        self.hero = Character(self.start_location, self.start_heading)
        self.reach_goal = False

    def distance_to_target(self):
        dist = point_distance(self.hero.location, self.end)
        if np.isnan(dist):
            return 500
        return dist

    def check_goal(self):
        dist = self.distance_to_target()
        self.reach_goal |= dist < self.end_radius
        return self.reach_goal

    def update(self, action):
        self.hero.apply_actions(action)

        loc = self.hero.update()
        collided = self.collide_lines(loc)
        if collided:
            self.hero.reset_location()
        
        self.hero.reset_observations()
        self.update_rangefinders()
        self.update_radar()

        return self.distance_to_target(), collided, self.check_goal()
    
    def collide_lines(self, point):
        return np.any(multiple_check_line_circle_collide(self.lines, point, self.collide_radius))
    
    def update_rangefinders(self):
        ranger_lines = self.hero.get_ranger_lines()
        valid_arr, points_arr = multiple_intersection(ranger_lines, self.lines)

        for i, (valid_row, points_row) in enumerate(zip(valid_arr, points_arr)):
            if np.any(valid_row):
                self.hero.rangers[i] = np.min(multiple_point_distance(
                        self.hero.location, points_row[valid_row]
                    ))
    
    def update_radar(self):
        end_heading = angle(self.end - self.hero.location) - self.hero.heading
        self.hero.update_radar(end_heading)

    def draw(self, screen):
        import pygame as pg

        screen.fill("white")
        pg.draw.circle(screen, "green", self.end, self.end_radius)

        for line in self.lines:
            pg.draw.line(screen, "black", *line)

        for dist, (point_a, point_b) in zip(self.hero.rangers, self.hero.get_ranger_lines()):
            off = point_b - point_a
            off *= dist / np.linalg.norm(off)
            pg.draw.line(screen, "blue", point_a, point_a + off)
        
        bound_rect = pg.Rect(0, 0, self.collide_radius * 4, self.collide_radius * 4)
        bound_rect.center = self.hero.location
        for i, (left, right) in enumerate(pairwise(self.hero.radar_angles)):
            if self.hero.radar[i % len(self.hero.radar)]:
                base = np.array([self.collide_radius * 2, 0])
                pg.draw.line(
                        screen, "green",
                        self.hero.location + rotate_vector(base, left + self.hero.heading),
                        self.hero.location + rotate_vector(base, right + self.hero.heading),
                        2
                    )
            else:
                left = np.deg2rad(left + self.hero.heading + 15)
                right = np.deg2rad(right + self.hero.heading - 15)
                pg.draw.arc(screen, "dark green", bound_rect, -right, -left, 1)


        pg.draw.circle(screen, "red", self.hero.location, self.collide_radius)
        off = rotate_vector(np.array([self.collide_radius, 0]), self.hero.heading)
        pg.draw.line(screen, "orange", self.hero.location, self.hero.location + off)
