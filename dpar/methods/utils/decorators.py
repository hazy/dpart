import warnings


def ignore_warning(func):
    def func_nowarn(*args, **kwargs):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return func(*args, **kwargs)

    return func_nowarn
