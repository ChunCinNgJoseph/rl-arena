import sys, random
while True:
    line = sys.stdin.readline()
    if not line: break
    print(random.randint(0, 6))
    sys.stdout.flush()