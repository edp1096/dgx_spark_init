"""Declarative vector scenes: model chooses objects, layout, colors and motion."""
import json
import os
from manim import *

config.frame_width = 8
config.frame_height = 8


class StoryScene(Scene):
    def construct(self):
        scene = json.loads(open(os.environ['SHORTS_SCENE_JSON']).read())
        design = scene.get('manim') or {}
        self.camera.background_color = design.get('background', '#101820')
        objects = design.get('objects', [])
        if not objects:
            # Content-only fallback; elaborate diagrams use explicit object specs.
            points = scene.get('points') or [scene['title']]
            objects = [{'type': 'text', 'text': text, 'x': 0,
                        'y': (len(points)-1)/2-i, 'font_size': 32}
                       for i, text in enumerate(points)]
        if len(objects) > 50:
            raise ValueError('Manim objects must be at most 50')
        events = []
        for entry in objects:
            kind = entry.get('type', 'text')
            color = entry.get('color', '#f5f2e8')
            width = min(7.5, max(.1, float(entry.get('width', 2))))
            height = min(7.5, max(.1, float(entry.get('height', 1))))
            if kind == 'text':
                obj = Text(str(entry.get('text', ''))[:200], font=entry.get('font', 'Noto Sans CJK KR'),
                           font_size=min(100, max(12, float(entry.get('font_size', 36)))), color=color)
                if obj.width > 7: obj.scale_to_fit_width(7)
            elif kind == 'circle':
                obj = Circle(radius=width/2, color=color)
            elif kind in ('rect', 'bar'):
                obj = RoundedRectangle(width=width, height=height, corner_radius=min(.15, width/4, height/4), color=color)
            elif kind in ('line', 'arrow'):
                start, end = entry.get('from', [-2, 0]), entry.get('to', [2, 0])
                start, end = [float(start[0]), float(start[1]), 0], [float(end[0]), float(end[1]), 0]
                obj = Arrow(start, end, buff=0, color=color) if kind == 'arrow' else Line(start, end, color=color)
            elif kind == 'polygon':
                vertices = entry.get('vertices', [[0, 1], [-1, -1], [1, -1]])[:20]
                obj = Polygon(*[[float(v[0]), float(v[1]), 0] for v in vertices], color=color)
            else:
                raise ValueError('Unsupported Manim object: '+kind)
            if kind not in ('line', 'arrow'):
                obj.move_to([float(entry.get('x', 0)), float(entry.get('y', 0)), 0])
            if kind not in ('text', 'line', 'arrow'):
                obj.set_fill(entry.get('fill', color), opacity=min(1, max(0, float(entry.get('fill_opacity', .2)))))
            start = max(0, float(entry.get('start', 0)))
            duration = max(.05, float(entry.get('duration', .6)))
            if start+duration > scene['duration']:
                raise ValueError('Manim animation exceeds scene duration')
            motion = entry.get('animation', 'fade')
            animation = Write(obj) if motion == 'write' else Create(obj) if motion == 'draw' else GrowFromCenter(obj) if motion == 'grow' else FadeIn(obj, shift=UP*float(entry.get('rise', 0)))
            animation.set_run_time(duration)
            events.append(Succession(Wait(start), animation) if start else animation)
        if events: self.play(AnimationGroup(*events, lag_ratio=0))
        if self.time < scene['duration']: self.wait(scene['duration']-self.time)
