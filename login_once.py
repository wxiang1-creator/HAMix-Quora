from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
import pickle
import time

options = Options()
options.add_argument("--start-maximized")
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

driver.get("https://www.quora.com/")
print("👉 请手动登录 Quora，登录成功后等 10 秒")
time.sleep(60)   # 给你时间扫码 / 输入密码

cookies = driver.get_cookies()
print("Cookies count:", len(cookies))

with open("quora_cookies.pkl", "wb") as f:
    pickle.dump(cookies, f)

print("✅ Cookies saved to quora_cookies.pkl")
driver.quit()
