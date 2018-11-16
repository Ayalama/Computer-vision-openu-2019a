import sys

def command_line_args(argv):
    argsdict = {}

    for farg in sys.argv:
        if farg.startswith('--'):
            (arg, val) = farg.split("=")
            arg = arg[2:]

            if arg in argsdict:
                argsdict[arg].append(val)
            else:
                argsdict[arg] = [val]
    return argsdict