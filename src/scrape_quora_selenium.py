import time
import csv
import pickle
from pathlib import Path
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager

URLS_FILE = Path("quora/data/raw/urls.txt")
OUT_CSV = Path("quora/data/processed/quora_answers_selenium.csv")
OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

def parse_number(s):
    s = s.upper().replace(",", "").strip()

    if "K" in s:
        return int(float(s.replace("K", "")) * 1000)
    if "M" in s:
        return int(float(s.replace("M", "")) * 1000000)

    try:
        return int(s)
    except:
        return 0
    

def setup_driver():
    options = Options()
    options.add_argument("--start-maximized")
    options.add_argument("--disable-blink-features=AutomationControlled")

    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

    driver.get("https://www.quora.com/")
    time.sleep(3)

    # 加 cookies（你已经有 quora_cookies.pkl）
    with open("quora_cookies.pkl", "rb") as f:
        cookies = pickle.load(f)

    for c in cookies:
        c.pop("sameSite", None)
        driver.add_cookie(c)

    driver.refresh()
    time.sleep(6)

    return driver

def scroll_page(driver, times=4):
    for _ in range(times):
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(6)

def main():
    urls = [u.strip() for u in URLS_FILE.read_text().splitlines() if u.strip()]
    driver = setup_driver()

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "url",
            "question_title",
            "answer_text",
            "upvotes",
            "comments",
            "views",
            "shares",
            "author",
            "time"
        ])

        for url in urls:
            print("[OPEN]", url)
            driver.get(url)
            time.sleep(20)

            scroll_page(driver, 4)

            title = driver.title

            answers = driver.find_elements(By.XPATH,'//div[contains(@class,"dom_annotate_question_answer_item")]')
            print("Answer blocks:", len(answers))
            
            for a in answers:
                try:
                    more_btn = a.find_element(
                        By.XPATH,
                        './/span[contains(@class,"qt_read_more") and contains(translate(text(),"MORE","more"),"more")]'
            )
                    driver.execute_script("arguments[0].click();", more_btn)
                except:
                    pass
                try:
                    text_elem = a.find_element(
                        By.XPATH,
                        './/div[contains(@class,"spacing_log_answer_content")]'
                    )
                    text = text_elem.text.strip()
                except:
                    continue
                if len(text) < 50:
                    continue

            # Upvotes
                try:
                    upvote_elem = a.find_element(
                        By.XPATH,
                        './/button[@aria-label="Upvote"]//span[contains(@class,"whiteSpace--nowrap")]'
                )
                    upvotes = parse_number(upvote_elem.text)
                except:
                    upvotes = "0"

            # Comments
                try:
                    comment_button = a.find_element(
                    By.XPATH,
                    './/button[contains(@aria-label,"comment")]'
                )
                    aria = comment_button.get_attribute("aria-label")
                    comments = aria.replace("comments", "").replace("comment", "").strip()
                except:
                    comments = "0"
            # Views
                try:
                    views_elem = a.find_element(
                    By.XPATH,
                    ".//div[contains(@class,'qu-dynamicFontSize--small')]//div[contains(@class,'fufw740')]//span[contains(@class,'c1h7helg')][contains(.,'views') or contains(.,'Views')]"
                )
                    t = " ".join(views_elem.text.split())
                    views = parse_number(t.split()[0])
                except:
                    views = "0"
            # Shares
                try:
                    share_button = a.find_element(
                    By.XPATH,
                    './/button[contains(@aria-label,"share")]'
                )
                    aria = share_button.get_attribute("aria-label")
                    shares = aria.replace("shares", "").replace("share", "").strip()
                except:
                    shares = "0"
            # Author
                try:
                    author_elem = a.find_element(
                    By.XPATH,
                    './/a[contains(@href,"/profile/")]//span'
                )
                    author = author_elem.text.strip()
                except:
                    author = ""

            # Time
                try:
                    time_elem = a.find_element(
                    By.XPATH,
                    './/a[contains(@class,"answer_timestamp")]'
            )
                    time_posted = time_elem.text
                except:
                    time_posted = ""

                writer.writerow([
                    url,
                    title,
                    text,
                    upvotes,
                    comments,
                    views,
                    shares,
                    author,
                    time_posted
                ])

        

    driver.quit()
    print("Saved:", OUT_CSV)

if __name__ == "__main__":
    main()
