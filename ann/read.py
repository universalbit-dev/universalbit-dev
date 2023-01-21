#Github: https://github.com/universalbit-dev
#
#Read serial Data
#... once again
#21-01-2023 https://universalbit.it
#License: CC0 1.0 Universal (CC0 1.0)
#https://creativecommons.org/share-your-work/public-domain/cc0/
#  
import time
import serial
import pandas as pd
#READ
r = serial.Serial(
port='/dev/ttyUSB0',baudrate = 9600,
parity=serial.PARITY_NONE,
stopbits=serial.STOPBITS_ONE,
bytesize=serial.EIGHTBITS,
timeout=0.1
)

while 1:
    read = r.readline()
    for i in read:
        r_df=read.split()
        # <===
        df = pd.DataFrame(r_df)
        print(r_df)
