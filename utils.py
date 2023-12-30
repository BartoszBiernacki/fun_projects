import re
import os


class StringUtils:
    """Class responsible for string manipulations.

    Especially useful for manipulation of strings, which
    represents file directory called `fdir`."""

    @staticmethod
    def remove_os_illegal_characters(text: str) -> str:
        """Removes characters, which cannot be used in file or folder name."""

        illegal_characters = r'[\\/:*?"<>|]'
        return re.sub(illegal_characters, '', text)

    @staticmethod
    def get_fname(fdir: str) -> str:
        """`C:User//downloads//...//[014]clip.mp3` --> `[014]clip.mp3`."""
        return os.path.split(fdir)[1]

    @staticmethod
    def get_fname_without_prefix(fdir: str) -> str:
        """`C:User//downloads//...//[014]clip.mp3` --> `clip.mp3`."""

        pattern = r'(\[\d+])?(.+)'
        fname = os.path.split(fdir)[1]
        return re.search(pattern, fname).group(2)

    @classmethod
    def get_fname_without_extension(cls, fdir: str) -> str:
        """`C:User//downloads//...//[014]clip.mp3` --> `[014]clip`."""
        return os.path.splitext(cls.get_fname(fdir))[0]

    @classmethod
    def get_fname_without_prefix_and_extension(cls, fdir: str) -> str:
        """`C:User//downloads//...//[014]clip.mp3` --> `clip`."""
        return os.path.splitext(cls.get_fname_without_prefix(fdir))[0]

