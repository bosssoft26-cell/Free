import random

def player(prev_play, opponent_history=[]):
    if prev_play:
        opponent_history.append(prev_play)

    if not opponent_history:
        return random.choice(["R", "P", "S"])

    most_common = max(set(opponent_history), key=opponent_history.count)

    if most_common == "R":
        return "P"
    elif most_common == "P":
        return "S"
    else:  # most_common == "S"
        return "R"
