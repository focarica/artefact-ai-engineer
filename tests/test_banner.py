import os
import unittest
from unittest.mock import patch

from artefact_ai import banner


class BannerTests(unittest.TestCase):
    def test_lines_are_aligned_and_use_known_characters(self):
        lines = banner.banner()
        self.assertEqual(len(lines), len(banner.ROBOT))
        self.assertEqual({len(line) for line in lines}, {banner.WIDTH})
        used = set(''.join(lines)) - {' '}
        self.assertLessEqual(used, set(banner.COLORS))

    def test_robot_shows_face_and_headphones(self):
        art = '\n'.join(banner.ROBOT)
        self.assertIn('OO', art)
        self.assertIn('####', art)
        self.assertIn('@@@@@@@@@@@@', art)

    def test_styling_is_skipped_when_color_is_disabled(self):
        with patch.dict(os.environ, {'NO_COLOR': '1'}):
            self.assertEqual(banner.styled('texto', 'blue', bold=True), 'texto')

    def test_colors_are_applied_on_a_terminal(self):
        with patch.object(banner.sys.stdout, 'isatty', return_value=True), patch.dict(os.environ, {}, clear=True):
            self.assertIn('\x1b[', banner.styled('texto', 'blue'))
            self.assertTrue(any('\x1b[' in line for line in banner.banner()))
