from RPS import player
from RPS_game import play, quincy  # أضف أي روبوت آخر تريد التجربة ضده

# مثال: لعب 1000 مباراة ضد quincy
play(player, quincy, 1000, verbose=True)
