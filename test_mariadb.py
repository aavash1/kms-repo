# test_mariadb.py
import mariadb
print(f"Imported mariadb from: {mariadb.__file__}")
print(f"Available attributes: {dir(mariadb)}")
try:
    conn = mariadb.connect(host="192.168.100.36", port=3306, user="netbackup", password="Dsti123!", database="netbackup")
    print("Connection successful!")
    conn.close()
except Exception as e:
    print(f"Error: {e}")