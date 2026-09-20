import snap7

PLC_IP = "192.168.150.103"
PLC_RACK = 0
PLC_SLOT = 2

client = snap7.client.Client()
client.connect(PLC_IP, PLC_RACK, PLC_SLOT)

print("connected =", client.get_connected())

data = client.db_read(17, 16, 4)
print("raw =", list(data))

client.disconnect()
client.destroy()