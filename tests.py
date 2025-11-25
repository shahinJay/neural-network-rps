import numpy as np
import random

poss = [0,0.25,0.5,0.75,1]
inp = random.choice(poss)
w1 = random.random()
b1 = random.random()
hid = None
w2 = random.random()
b2 = random.random()
out = None

des = 0
cost = 0

hm = {0: 0.25, 0.25: 0.5, 0.5: 0.75, 0.75: 1, 1: 0}

time = 100000

def sigmoid(x):
  return 1/(1+pow(np.e,-x))

def backprop_1(out, hid, w2, b2, error):
  del_w = 0.2 * hid * 2*error
  del_b = 0.2 * 2*error  # Even simpler!
  w2 -= del_w
  b2 -= del_b

def backprop_2(out, hid, inp, w2, w1, b1, error):
  del_w = 0.2 * inp * hid*(1-hid) * w2 * out*(1-out) * 2*error
  del_b = 0.2 * hid*(1-hid) * w2 * out*(1-out) * 2*error  # Same as above but without 'inp'
  w1 -= del_w
  b1 -= del_b


def log(inp, w1, b1, hid, w2, b2, out, cost):
  print("-------------------------------------------------------------------------------------------------")
  print(f"\n\n\nINPUT:[{inp}] with {w1} and {b1} to HIDDEN:[{hid}] with {w2} and {b2} to OUTPUT:[{out}]\n\n")
  print(f"COST: {cost}\n\n")
  print("-------------------------------------------------------------------------------------------------")

def core(poss, inp, hid, out, w1, w2, b1, b2, des, cost, time):

  for _ in range(time):
    inp = random.choice(poss) 
    hid = sigmoid(inp*w1 + b1)
    out = sigmoid(hid*w2 + b2)
    for k, v in hm.items():
      if inp == k:
        des = v
    cost = pow((out-des),2)
    error = out-des
    backprop_1(out, hid, w2, des, error)
    backprop_2(out, hid, inp, w2, w1, des, error)
  print("FINAL:")
  log(inp, w1, b1, hid, w2, b2, out, cost)

core(poss, inp, hid, out, w1, w2, b1, b2, des, cost, time)

def test(inp, hid, out, w1, w2, b1, b2, poss):
  rep = float(input("GO: "))
  while rep in poss:
    inp = rep
    hid = inp*w1 + b1
    out = hid*w2 + b2

    cost = (out-des)
    cost *= cost
    
    print(f"INPUT: {inp}, OUTPUT: {out}, COST {cost}")

    rep = float(input("GO: "))
    if not rep in poss:
      exit()

test(inp, hid, out, w1, w2, b1, b2, poss)


