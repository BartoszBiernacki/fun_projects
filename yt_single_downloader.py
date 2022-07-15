import cProfile
import os.path
import pstats
import glob
import pathlib
import pytube
import moviepy.editor as editor

from itertools import compress
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor


def profile():
    with cProfile.Profile() as pr:
        pass
    stats = pstats.Stats(pr)
    stats.sort_stats(pstats.SortKey.TIME)
    stats.print_stats(5)


class YtClipsDownloader:
    def __init__(self, urls: list[str]):
        self.urls = urls
        self.num_of_urls = len(self.urls)

        self.destination_folder = (
                str(pathlib.Path.home()) +
                str(pathlib.Path(r'//Downloads//YouTube//')))

    @staticmethod
    def get_best_audio_stream(url: str) -> pytube.streams.Stream:
        """
        Gets `url` to video on YT and return `pytube.streams.Stream` with
        the best audio sampling rate.

        Audio sampling rate is given by `abr` attribute.
        """
        audio_streams = pytube.YouTube(url).streams.filter(only_audio=True)
        return audio_streams.order_by('abr')[-1]

    def all_fdirs_in_dir(self) -> list[str]:
        directory = self.destination_folder
        if not pathlib.Path(directory).exists():
            return []

        cwd = os.getcwd()
        os.chdir(directory)
        fdirs = [directory + '//' + fname for fname in glob.glob("*.*")]
        os.chdir(cwd)

        fdirs.sort()
        return fdirs

    def all_not_mp3_fdirs(self) -> list[str]:
        fdirs = self.all_fdirs_in_dir()
        return [fdir for fdir in fdirs if not fdir.endswith('.mp3')]

    def download_file(self, url: str, video_num: int) -> str:
        print(f"Downloading file {video_num + 1}"
              f" out of {self.num_of_urls} ...")
        try:
            stream = self.get_best_audio_stream(url)
            fdir = stream.download(output_path=self.destination_folder)
        except KeyError:  # KeyError: 'streamingData'
            fdir = None     # don't download if can't

        return fdir

    def convert_to_mp3(self, fdir: str, video_num: int) -> None:
        print(f"Clip {video_num + 1} out of {self.num_of_urls}"
              f" is converted to mp3 ...")

        audio_clip = editor.AudioFileClip(fdir)
        audio_clip.write_audiofile(
            os.path.splitext(fdir)[0] + '.mp3', verbose=False, logger=None)
        audio_clip.close()

    def download_covert_delete(self, url: str, video_num: int):
        fdir = self.download_file(url, video_num)
        self.convert_to_mp3(fdir, video_num)
        os.remove(fdir)

    def is_downloaded(self, url: str) -> bool:
        fdirs = self.all_fdirs_in_dir()
        online_fname = self.get_best_audio_stream(url).default_filename
        return any(os.path.splitext(online_fname)[0] in fdir for fdir in fdirs)

    def get_missing_urls(self) -> list[str]:
        with ThreadPoolExecutor(max_workers=50) as executor:
            mask = list(executor.map(self.is_downloaded, self.urls))

        mask = [not x for x in mask]
        return list(compress(self.urls, mask))

    def download_convert_delete_all(self):
        urls = self.get_missing_urls()

        with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
            executor.map(self.download_covert_delete, urls, range(len(urls)))
        print("Pobieranie ukończone pomyślnie, program kończy działanie.")

    def remove_bad_downloaded_files(self) -> None:
        for fdir in self.all_fdirs_in_dir():
            if os.path.getsize(fdir) == 0:
                os.remove(fdir)

    def go(self):
        self.remove_bad_downloaded_files()
        self.download_convert_delete_all()


if __name__ == "__main__":
    YtClipsDownloader(urls=[
        'https://www.youtube.com/watch?v=WAvT_w5NELQ',
        'https://www.youtube.com/watch?v=4A71Wswcy6o',
        'https://www.youtube.com/watch?v=xbeM2I0Cidw',
    ]).go()
