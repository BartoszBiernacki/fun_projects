import os


class Config:
    _home_dir = os.path.expanduser('~')

    BASE_OUTPUT_PATH = os.path.join(
        _home_dir, 'Downloads', 'YouTube', 'audio')

    NUM_SCRAPERS = 1000
    NUM_DOWNLOADERS = 20
    NUM_CONVERTERS = os.cpu_count()

    # audio quality based on Yt filesize, quality in <0, 1>
    # 0 --> the worst quality & small filesize
    # 1 --> the best quality & large filesize
    QUALITY = 0

    _DESIRED_EXTENSIONS = ['mp3', 'mp4']
    DELETE_AFTER_CONVERSION = True

    # DO NOT CHANGE ----------------------------------------------------
    DESIRED_EXTENSIONS_DOTS = [ext if ext.startswith('.')
                               else ('.' + ext)
                               for ext in _DESIRED_EXTENSIONS]

    DESIRED_EXTENSIONS_NO_DOTS = [ext.replace('.', '') if ext.startswith('.')
                                  else ext
                                  for ext in _DESIRED_EXTENSIONS]
