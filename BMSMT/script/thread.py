import threading
import time
 
 
def scraper():
    print("start")
    time.sleep(5)
    print("sleep done")
    return
 
 
t1 = threading.Thread(target=scraper)  #建立執行緒
t2 = threading.Thread(target=scraper)  #建立執行緒
t1.start()  #執行
t2.start()
print("end")


print("GO")