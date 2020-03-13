import pathlib

def maybe_makedirs(path_to_create):
    """This function will create a directory, unless it exists already,
    at which point the function will return.
    Inputs:
        path_to_create - A string path to a directory you'd like created.
    """
    pathlib.Path(path_to_create).mkdir(parents=True, exist_ok=True)