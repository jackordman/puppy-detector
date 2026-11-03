"""Push notifications. ntfy.sh by default (no account): subscribe to NTFY_TOPIC in the ntfy app.
    notify.py <image.jpg>   send a test push
"""
import os
import sys
import urllib.request

NTFY_TOPIC = os.environ.get("NTFY_TOPIC", "")
NTFY_SERVER = os.environ.get("NTFY_SERVER", "https://ntfy.sh")


def push(title, image_path):
    if not NTFY_TOPIC:
        return
    req = urllib.request.Request(
        f"{NTFY_SERVER}/{NTFY_TOPIC}", data=open(image_path, "rb").read(), method="PUT",
        headers={"Title": title, "Filename": os.path.basename(image_path), "Tags": "dog"})
    try:
        urllib.request.urlopen(req, timeout=20).read()
    except Exception as e:  # noqa: BLE001
        print("notify failed:", e, flush=True)


if __name__ == "__main__":
    push("Puppy detector test", sys.argv[1])
    print("sent" if NTFY_TOPIC else "NTFY_TOPIC not set")
