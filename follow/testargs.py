import sys


if len(sys.argv) > 1:
	print ("arg [1] = ", sys.argv[1])

if len(sys.argv) > 2:
	print ("arg [2] = ", sys.argv[2])

if len(sys.argv) > 3:
	print ("arg [3] = ", sys.argv[3])

if len(sys.argv) != 4:
	print ("invalid args")
	exit (1)



for i in range (len (sys.argv)):
	print (f"arg {i} = {sys.argv [i]}")
	


