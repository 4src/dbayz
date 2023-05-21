def binMerge(bin1,bin2):
  bin12      = BIN(bin1.col)
  bin12.lo   = min(bin1.lo, bin2.lo)
  bin12.hi   = max(bin1.hi, bin2.hi)
  bin12.rows = bin1.rows + bin2.rows
  bin12.ys   = symMerge(bin1.klasses, bin2.klasses)
  return bin12

def symMerge(sym1,sym2):
  sym12 = deepcopy(sym1)
  for x,n in sym2.tally.items(): colAdd(sym12,x,n)
  return sym12

class Fred:
  def aa(): print(1)
  def bb(): print(2)
  def cc(): print(3)

all="all"
for m in dir(Fred):
  if m[0] != "_" and all in [m,"all"]:
    getattr(Fred, m)()
