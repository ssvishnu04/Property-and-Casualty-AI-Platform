import requests


BASE_URL = "http://127.0.0.1:8000"


def main():
    response = requests.get(f"{BASE_URL}/claims/new?limit=2", timeout=30)
    response.raise_for_status()

    payload = response.json()

    print("Status:", response.status_code)
    print("Record count:", payload["record_count"])
    print("First claim:")
    print(payload["claims"][0])


if __name__ == "__main__":
    main()