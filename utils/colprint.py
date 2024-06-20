def printr(*args, **kwargs): print("\033[91m{}\033[00m" .format(' '.join(map(str, args))), **kwargs)
def printy(*args, **kwargs): print("\033[93m{}\033[00m" .format(' '.join(map(str, args))), **kwargs)
def printg(*args, **kwargs): print("\033[92m{}\033[00m" .format(' '.join(map(str, args))), **kwargs)