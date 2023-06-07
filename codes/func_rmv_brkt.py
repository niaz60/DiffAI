def rmv_brkt(string):
    if string == '.':
        return 0
    return float(string.replace("(", "").replace(")", "").replace("-.", "-0.").replace('?', '0').replace("..", "."))
