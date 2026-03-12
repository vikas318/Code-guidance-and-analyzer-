def test():
  a=7
  b="*"
  while(a>0):
    a+=1
    print(b*a)
    b+="*"
    if(len(b)%2==0):
      a-=4
 
test()