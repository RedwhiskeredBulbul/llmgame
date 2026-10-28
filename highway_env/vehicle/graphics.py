from __future__ import annotations

import itertools
from typing import TYPE_CHECKING

import numpy as np
import pygame

from highway_env.utils import Vector
from highway_env.vehicle.behavior import IDMVehicle, LinearVehicle
from highway_env.vehicle.controller import MDPVehicle
from highway_env.vehicle.dynamics import BicycleVehicle
from highway_env.vehicle.kinematics import Vehicle


if TYPE_CHECKING:
    from highway_env.road.graphics import WorldSurface


class VehicleGraphics:
    RED = (255, 100, 100)
    GREEN = (50, 200, 0)
    BLUE = (100, 200, 255)
    YELLOW = (200, 200, 0)
    BLACK = (60, 60, 60)
    PURPLE = (200, 0, 150)
    DEFAULT_COLOR = YELLOW
    EGO_COLOR = GREEN

    @classmethod
    def display(
        cls,
        vehicle: Vehicle,
        surface: WorldSurface,
        transparent: bool = False,
        offscreen: bool = False,
        label: bool = False,
        draw_roof: bool = False,
    ) -> None:
        """
        Display a vehicle on a pygame surface.

        The vehicle is represented as a colored rotated rectangle.

        :param vehicle: the vehicle to be drawn
        :param surface: the surface to draw the vehicle on
        :param transparent: whether the vehicle should be drawn slightly transparent
        :param offscreen: whether the rendering should be done offscreen or not
        :param label: whether a text label should be rendered
        """
        if not surface.is_visible(vehicle.position):
            return

        v = vehicle
        tire_length, tire_width = 1, 0.3
        headlight_length, headlight_width = 0.72, 0.6
        roof_length, roof_width = 2.0, 1.5

        # Vehicle rectangle
        length = v.LENGTH + 2 * tire_length
        vehicle_surface = pygame.Surface(
            (surface.pix(length), surface.pix(length)), flags=pygame.SRCALPHA
        )  # per-pixel alpha
        rect = (
            surface.pix(tire_length),
            surface.pix(length / 2 - v.WIDTH / 2),
            surface.pix(v.LENGTH),
            surface.pix(v.WIDTH),
        )
        rect_headlight_left = (
            surface.pix(tire_length + v.LENGTH - headlight_length),
            surface.pix(length / 2 - (1.4 * v.WIDTH) / 3),
            surface.pix(headlight_length),
            surface.pix(headlight_width),
        )
        rect_headlight_right = (
            surface.pix(tire_length + v.LENGTH - headlight_length),
            surface.pix(length / 2 + (0.6 * v.WIDTH) / 5),
            surface.pix(headlight_length),
            surface.pix(headlight_width),
        )
        color = cls.get_color(v, transparent)
        pygame.draw.rect(vehicle_surface, color, rect, 0)
        pygame.draw.rect(vehicle_surface, cls.lighten(color), rect_headlight_left, 0)
        pygame.draw.rect(vehicle_surface, cls.lighten(color), rect_headlight_right, 0)
        if draw_roof:
            rect_roof = (
                surface.pix(v.LENGTH / 2 - tire_length / 2),
                surface.pix(0.999 * length / 2 - 0.38625 * v.WIDTH),
                surface.pix(roof_length),
                surface.pix(roof_width),
            )
            pygame.draw.rect(vehicle_surface, cls.darken(color), rect_roof, 0)
        pygame.draw.rect(vehicle_surface, cls.BLACK, rect, 1)

        # Tires
        if type(vehicle) in [Vehicle, BicycleVehicle]:
            tire_positions = [
                [surface.pix(tire_length), surface.pix(length / 2 - v.WIDTH / 2)],
                [surface.pix(tire_length), surface.pix(length / 2 + v.WIDTH / 2)],
                [
                    surface.pix(length - tire_length),
                    surface.pix(length / 2 - v.WIDTH / 2),
                ],
                [
                    surface.pix(length - tire_length),
                    surface.pix(length / 2 + v.WIDTH / 2),
                ],
            ]
            tire_angles = [0, 0, v.action["steering"], v.action["steering"]]
            for tire_position, tire_angle in zip(tire_positions, tire_angles):
                tire_surface = pygame.Surface(
                    (surface.pix(tire_length), surface.pix(tire_length)),
                    pygame.SRCALPHA,
                )
                rect = (
                    0,
                    surface.pix(tire_length / 2 - tire_width / 2),
                    surface.pix(tire_length),
                    surface.pix(tire_width),
                )
                pygame.draw.rect(tire_surface, cls.BLACK, rect, 0)
                cls.blit_rotate(
                    vehicle_surface,
                    tire_surface,
                    tire_position,
                    np.rad2deg(-tire_angle),
                )

        # Centered rotation
        h = v.heading if abs(v.heading) > 2 * np.pi / 180 else 0
        position = [*surface.pos2pix(v.position[0], v.position[1])]
        if not offscreen:
            # convert_alpha throws errors in offscreen mode
            # see https://stackoverflow.com/a/19057853
            vehicle_surface = pygame.Surface.convert_alpha(vehicle_surface)
        cls.blit_rotate(surface, vehicle_surface, position, np.rad2deg(-h))

        # Label
        if label:
            font = pygame.font.Font(None, 15)
            text = f"#{id(v) % 1000}"
            text = font.render(text, 1, (10, 10, 10), (255, 255, 255))
            surface.blit(text, position)

    @staticmethod
    def blit_rotate(
        surf: pygame.SurfaceType,
        image: pygame.SurfaceType,
        pos: Vector,
        angle: float,
        origin_pos: Vector = None,
        show_rect: bool = False,
    ) -> None:
        """Many thanks to https://stackoverflow.com/a/54714144."""
        # calculate the axis aligned bounding box of the rotated image
        w, h = image.get_size()
        box = [pygame.math.Vector2(p) for p in [(0, 0), (w, 0), (w, -h), (0, -h)]]
        box_rotate = [p.rotate(angle) for p in box]
        min_box = (
            min(box_rotate, key=lambda p: p[0])[0],
            min(box_rotate, key=lambda p: p[1])[1],
        )
        max_box = (
            max(box_rotate, key=lambda p: p[0])[0],
            max(box_rotate, key=lambda p: p[1])[1],
        )

        # calculate the translation of the pivot
        if origin_pos is None:
            origin_pos = w / 2, h / 2
        pivot = pygame.math.Vector2(origin_pos[0], -origin_pos[1])
        pivot_rotate = pivot.rotate(angle)
        pivot_move = pivot_rotate - pivot

        # calculate the upper left origin of the rotated image
        origin = (
            pos[0] - origin_pos[0] + min_box[0] - pivot_move[0],
            pos[1] - origin_pos[1] - max_box[1] + pivot_move[1],
        )
        # get a rotated image
        rotated_image = pygame.transform.rotate(image, angle)
        # rotate and blit the image
        surf.blit(rotated_image, origin)
        # draw rectangle around the image
        if show_rect:
            pygame.draw.rect(surf, (255, 0, 0), (*origin, *rotated_image.get_size()), 2)

    @classmethod
    def display_trajectory(
        cls, states: list[Vehicle], surface: WorldSurface, offscreen: bool = False
    ) -> None:
        """
        Display the whole trajectory of a vehicle on a pygame surface.

        :param states: the list of vehicle states within the trajectory to be displayed
        :param surface: the surface to draw the vehicle future states on
        :param offscreen: whether the rendering should be done offscreen or not
        """
        for vehicle in states:
            cls.display(vehicle, surface, transparent=True, offscreen=offscreen)

    @classmethod
    def display_trajectory_points(
        cls, states: np.array, surface: WorldSurface, offscreen: bool = False, color: tuple = None
    ) -> None:
        """
        Display the whole trajectory of a vehicle on a pygame surface.

        :param states: the list of vehicle states within the trajectory to be displayed
        :param surface: the surface to draw the vehicle future states on
        :param offscreen: whether the rendering should be done offscreen or not
        """
        # 空数据保护
        if states is None or states.size == 0:
            return

        # pos2pix 只接受单个坐标值，需要遍历每个状态点（世界坐标 -> 像素坐标）
        positions = [surface.pos2pix(float(states[i, 0]), float(states[i, 1]))
                     for i in range(states.shape[0])]

        n = len(positions)
        if n == 0:
            return

        # 渐变参数
        start_rgb = color if color is not None else cls.YELLOW  # 起始颜色（可换为任意颜色）
        end_rgb = color if color is not None else cls.YELLOW     # 终止颜色
        alpha_start, alpha_end = 60, 255  # 透明度渐变范围

        radius = 2
        diameter = radius * 2

        for i, pos in enumerate(positions):
            # 归一化进度 t ∈ [0,1]
            t = (i / (n - 1)) if n > 1 else 1.0

            # 颜色线性插值（RGB）
            r = int(start_rgb[0] + (end_rgb[0] - start_rgb[0]) * t)
            g = int(start_rgb[1] + (end_rgb[1] - start_rgb[1]) * t)
            b = int(start_rgb[2] + (end_rgb[2] - start_rgb[2]) * t)

            # Alpha 渐变
            # alpha = int(alpha_start + (alpha_end - alpha_start) * t)
            alpha = int(alpha_end - (alpha_end - alpha_start) * t)

            # 在带 alpha 的临时 Surface 上绘制，再 Blit 到目标 Surface 以生效透明度
            circle_surf = pygame.Surface((diameter, diameter), pygame.SRCALPHA)
            pygame.draw.circle(circle_surf, (r, g, b, alpha), (radius, radius), radius)

            # 将圆心对齐到像素位置
            x, y = int(round(pos[0])), int(round(pos[1]))
            surface.blit(circle_surf, (x - radius, y - radius))

    @classmethod
    def display_corridor(
        cls, corridor: list, surface: WorldSurface, offscreen: bool = False
    ) -> None:
        """
        Display the safe corridor of a vehicle on a pygame surface.

        :param corridor: the list of points defining the corridor
        :param surface: the surface to draw the corridor on
        :param offscreen: whether the rendering should be done offscreen or not
        """
        # 绘制安全走廊：同一种绿色，随时间步 alpha 递减（未来更“淡”），矩形内部填充 + 细轮廓
        # x_lower_f, x_upper_f, y_lower_f, y_upper_f, x_lower_r, x_upper_r, y_lower_r, y_upper_r = corridor
        # num_points = len(x_lower_f)

        lt_f, rt_f, lb_f, rb_f, lt_r, rt_r, lb_r, rb_r = corridor
        num_points = len(lt_f)

        if num_points == 0:
            return

        base_green = np.array([60, 200, 80])   # 统一绿色（可与 cls.GREEN 区分开）
        outline_color = (40, 140, 55)          # 轮廓使用稍深绿色
        alpha_start, alpha_end = 20, 1       # 起始更实，远端更淡
        outline_width = 1  # 细轮廓宽度

        for i in range(num_points):
            t = (i / (num_points - 1)) if num_points > 1 else 1.0
            alpha = int(alpha_start + (alpha_end - alpha_start) * t)  # 线性递减
            fill_rgb = base_green.astype(int)
            fill_color_rgba = (int(fill_rgb[0]), int(fill_rgb[1]), int(fill_rgb[2]), alpha)
            border_color = outline_color

            # 绘制矩形：前后盘分别绘制
            # lt_f, rt_f, lb_f, rb_f为前盘四个点坐标，lb_f为左下，rt_f为右上


            # 前盘：使用四个角点(左上lt_f, 右上rt_f, 右下rb_f, 左下lb_f)绘制任意方向矩形
            def _pix(p):
                return surface.pos2pix(float(p[0]), float(p[1]))

            try:
                front_pts_world = [lt_f[i], rt_f[i], rb_f[i], lb_f[i]]
            except Exception:
                front_pts_world = []
            if front_pts_world:
                front_pts_pix = [_pix(p) for p in front_pts_world]
                xs_f = [p[0] for p in front_pts_pix]
                ys_f = [p[1] for p in front_pts_pix]
                min_x_f, max_x_f = int(round(min(xs_f))), int(round(max(xs_f)))
                min_y_f, max_y_f = int(round(min(ys_f))), int(round(max(ys_f)))
                w_f = max_x_f - min_x_f
                h_f = max_y_f - min_y_f
                if w_f > 0 and h_f > 0:
                    # 局部Surface提高透明度呈现效果
                    local_surf_f = pygame.Surface((w_f, h_f), pygame.SRCALPHA)
                    # 重新偏移到局部Surface坐标系
                    offset_front = [(int(round(x - min_x_f)), int(round(y - min_y_f))) for x, y in front_pts_pix]
                    pygame.draw.polygon(local_surf_f, fill_color_rgba, offset_front, 0)
                    pygame.draw.polygon(local_surf_f, (*border_color, alpha), offset_front, outline_width)
                    surface.blit(local_surf_f, (min_x_f, min_y_f))

            # 后盘：使用四个角点(左上lt_r, 右上rt_r, 右下rb_r, 左下lb_r)
            try:
                rear_pts_world = [lt_r[i], rt_r[i], rb_r[i], lb_r[i]]
            except Exception:
                rear_pts_world = []
            if rear_pts_world:
                rear_pts_pix = [_pix(p) for p in rear_pts_world]
                xs_r = [p[0] for p in rear_pts_pix]
                ys_r = [p[1] for p in rear_pts_pix]
                min_x_r, max_x_r = int(round(min(xs_r))), int(round(max(xs_r)))
                min_y_r, max_y_r = int(round(min(ys_r))), int(round(max(ys_r)))
                w_r = max_x_r - min_x_r
                h_r = max_y_r - min_y_r
                if w_r > 0 and h_r > 0:
                    local_surf_r = pygame.Surface((w_r, h_r), pygame.SRCALPHA)
                    offset_rear = [(int(round(x - min_x_r)), int(round(y - min_y_r))) for x, y in rear_pts_pix]
                    pygame.draw.polygon(local_surf_r, fill_color_rgba, offset_rear, 0)
                    pygame.draw.polygon(local_surf_r, (*border_color, alpha), offset_rear, outline_width)
                    surface.blit(local_surf_r, (min_x_r, min_y_r))

    @classmethod
    def display_history(
        cls,
        vehicle: Vehicle,
        surface: WorldSurface,
        frequency: float = 3,
        duration: float = 2,
        simulation: int = 15,
        offscreen: bool = False,
    ) -> None:
        """
        Display the whole trajectory of a vehicle on a pygame surface.

        :param vehicle: the vehicle states within the trajectory to be displayed
        :param surface: the surface to draw the vehicle future states on
        :param frequency: frequency of displayed positions in history
        :param duration: length of displayed history
        :param simulation: simulation frequency
        :param offscreen: whether the rendering should be done offscreen or not
        """
        for v in itertools.islice(
            vehicle.history,
            None,
            int(simulation * duration),
            int(simulation / frequency),
        ):
            cls.display(v, surface, transparent=True, offscreen=offscreen)

    @classmethod
    def get_color(cls, vehicle: Vehicle, transparent: bool = False) -> tuple[int]:
        color = cls.DEFAULT_COLOR
        if getattr(vehicle, "color", None):
            color = vehicle.color
        elif vehicle.crashed:
            color = cls.RED
        elif isinstance(vehicle, LinearVehicle):
            color = cls.YELLOW
        elif isinstance(vehicle, IDMVehicle):
            color = cls.BLUE
        elif isinstance(vehicle, MDPVehicle):
            color = cls.EGO_COLOR
        if transparent:
            color = (color[0], color[1], color[2], 30)
        return color

    @classmethod
    def darken(cls, color, ratio=0.83):
        return (
            int(color[0] * ratio),
            int(color[1] * ratio),
            int(color[2] * ratio),
        ) + color[3:]

    @classmethod
    def lighten(cls, color, ratio=0.68):
        return (
            min(int(color[0] / ratio), 255),
            min(int(color[1] / ratio), 255),
            min(int(color[2] / ratio), 255),
        ) + color[3:]
