import pytube
import pathlib
import os
import glob
import re
import moviepy.audio.fx.volumex
import moviepy.editor as editor

from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor


class YTDownloader:

    def __init__(self, channel_url: str):
        self.yt_channel = pytube.Channel(channel_url)
        self.num_of_videos_on_channel = len(self.yt_channel.video_urls)

        self.destination_folder = \
            "C://Users//HAL//Downloads//YT_audio_downloads//" + \
            self.yt_channel.channel_name

    def create_video_prefix(self, i: int) -> str:  # '[001]', '[002]', ...
        num_of_digits = len(str(self.num_of_videos_on_channel))
        return f"[{str(i + 1).zfill(num_of_digits)}]"

    @staticmethod
    def read_fdir_int_prefix(fdir: str) -> int:
        """
        Assume fnames in fdirs containing part `[dddd]` where `d` represent
        any digit and sequence can be arbitrary length.
        
        Function returns `int(dddd)`.
        """
        return int(re.search(r'\[(\d+)]', fdir).group(1))

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

    def download_file(self, url: str, video_num: int) -> None:
        print(f"Downloading file {video_num + 1}"
              f" out of {self.num_of_videos_on_channel} ...")
        try:
            stream = self.get_best_audio_stream(url)
            stream.download(
                output_path=self.destination_folder,
                filename_prefix=self.create_video_prefix(video_num))
        except KeyError:  # KeyError: 'streamingData'
            pass  # don't download if can't

    def convert_to_mp3(self, fdir: str, video_num: int) -> None:
        print(f"Clip {video_num + 1} out of {self.num_of_videos_on_channel}"
              f" is converted to mp3 ...")

        audio_clip = editor.AudioFileClip(fdir)
        audio_clip = moviepy.audio.fx.volumex.volumex(audio_clip, 2.0)
        audio_clip.write_audiofile(
            os.path.splitext(fdir)[0] + '.mp3', verbose=False, logger=None)
        audio_clip.close()

    @staticmethod
    def is_converted_successfully(
            fdir_original: str, fdir_converted: str) -> bool:
        """
        Clip is considered converted successfully if it's duration not differ
        more than 1% from original clip duration.

        Function points on cases where conversion was interrupted by user.
        """
        clip_original = editor.AudioFileClip(fdir_original)
        clip_converted = editor.AudioFileClip(fdir_converted)

        relative_error = (abs(clip_original.duration - clip_converted.duration)
                          / clip_original.duration)

        clip_original.close()
        clip_converted.close()

        return relative_error < 0.01

    def get_missing_urls_and_indices(self) -> (list[str], list[int]):
        urls = list(reversed(self.yt_channel.video_urls))
        url_indices = list(range(len(urls)))

        fdir_indices = [self.read_fdir_int_prefix(fdir)
                        for fdir in self.all_fdirs_in_dir()]

        for url_idx, url in enumerate(reversed(self.yt_channel.video_urls)):
            if url_idx + 1 in fdir_indices:
                urls.remove(url)
                url_indices.remove(url_idx)

        return urls, url_indices

    def download_all_missing_files_from_channel(self) -> None:
        urls, indices = self.get_missing_urls_and_indices()

        with ThreadPoolExecutor(max_workers=6) as executor:
            executor.map(self.download_file, urls, indices)

        print("Downloading completed successfully!")

    def get_fdirs_to_convert(self) -> list[str]:
        not_mp3_fdirs = self.all_not_mp3_fdirs()
        not_mp3_fdir_indices = [self.read_fdir_int_prefix(fdir)
                                for fdir in not_mp3_fdirs]
        not_mp3s = dict(zip(not_mp3_fdir_indices, not_mp3_fdirs))

        mp3_fdirs = list(
            set(self.all_fdirs_in_dir()) - set(self.all_not_mp3_fdirs()))
        mp3_fdir_indices = [self.read_fdir_int_prefix(fdir)
                            for fdir in mp3_fdirs]
        maybe_mp3s = dict(zip(mp3_fdir_indices, mp3_fdirs))

        for k, v in not_mp3s.copy().items():
            if k in maybe_mp3s:
                if self.is_converted_successfully(
                    fdir_original=v, fdir_converted=maybe_mp3s[k]
                ):
                    not_mp3s.pop(k)

        return list(not_mp3s.values())

    def convert_downloaded_files_to_mp3(self) -> None:
        fdirs = self.get_fdirs_to_convert()

        with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
            executor.map(self.convert_to_mp3, fdirs, range(len(fdirs)))

        print("mp3 conversion completed successfully!")

    def remove_bad_downloaded_files(self) -> None:
        for fdir in self.all_fdirs_in_dir():
            if os.path.getsize(fdir) == 0:
                os.remove(fdir)

    def remove_not_mp3_files(self):
        for fdir in self.all_not_mp3_fdirs():
            os.remove(fdir)

    def go(self):
        self.remove_bad_downloaded_files()
        self.download_all_missing_files_from_channel()
        self.convert_downloaded_files_to_mp3()
        self.remove_not_mp3_files()


if __name__ == '__main__':
    YTDownloader('https://www.youtube.com/c/Astrofaza').go()
    YTDownloader('https://www.youtube.com/c/CopernicuscenterEduPl/videos').go()
    YTDownloader('https://www.youtube.com/c/WszechnicaFWW/videos').go()
    YTDownloader('https://www.youtube.com/user/HistoriaBezCenzuryMB/videos').go()
    YTDownloader('https://www.youtube.com/c/Polimaty/videos').go()
    YTDownloader('https://www.youtube.com/c/ciekawehistorietv/videos').go()
    YTDownloader('https://www.youtube.com/c/Smartgasm/videos').go()
    YTDownloader('https://www.youtube.com/c/RadioNaukowe/videos').go()
    YTDownloader('https://www.youtube.com/channel/UC8ROk73ZAWnDk6VS5kXTzVQ/videos').go()
