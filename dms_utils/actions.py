ACTIONS = {
    'phonecall': 0,
    'texting': 1,
}

ACTIONS_EXTENDED = {
    'phonecall': 0,
    'texting': 1,
    'smoking': 2,
    'drinking': 3,
    'eating': 4,
    'looking_left': 5,
    'looking_right': 6,
    'drowsy': 7,
}

def get_num_classes(extended: bool = False) -> int:
    if extended:
        return len(ACTIONS_EXTENDED)
    return len(ACTIONS)
