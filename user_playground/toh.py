def toh(n, fromRod, toRod, auxRod):
    if n == 0:
        return
    toh(n-1, fromRod, auxRod, toRod)
    print("Disk", n, " moved from ", fromRod, " to ", toRod)
    toh(n-1, auxRod, toRod, fromRod)

if __name__ == "__main__":
    n = 3
    
    # A, C, B are the name of rods
    toh(n, 'A', 'C', 'B')