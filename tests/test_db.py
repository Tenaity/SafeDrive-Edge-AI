import snap7

client = snap7.client.Client()
client.connect("192.168.1.1", 0, 1)
print("Connected:", client.get_connected())

for start in [0, 1, 6, 7, 8]:
    try:
        data = client.db_read(17, start, 1)
        print(f"DB17 byte {start} =", list(data))
    except Exception as e:
        print(f"DB17 byte {start} ERROR:", e)

client.disconnect()