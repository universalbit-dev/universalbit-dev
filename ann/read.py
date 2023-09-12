#Github: https://github.com/universalbit-dev
#
#Read serial Data
#... once again
#universalbit.it
#License: CC0 1.0 Universal (CC0 1.0)
#https://creativecommons.org/share-your-work/public-domain/cc0/
#
import time
import serial
import pandas as pd
import asyncio
#READ
r = serial.Serial(
port='/dev/ttyUSB0',baudrate = 9600, #<===USB Port: /dev/ttyUSB1
parity=serial.PARITY_NONE,
stopbits=serial.STOPBITS_ONE,
bytesize=serial.EIGHTBITS,
timeout=0.1
)

async def read():
    await asyncio.sleep(0.1)
    read = r.readline()
    for i in read:
        r_df=read.split()
        #print(read)
        df = pd.DataFrame(r_df)
        print(r_df)

while 1:
    asyncio.run(read())
    time.sleep(0.2)
