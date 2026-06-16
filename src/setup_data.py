"""Bootstrap the Chroma vector store on startup (for cloud deployment).

On Streamlit Cloud the repo is cloned WITHOUT the prebuilt Chroma folder
(it's git-ignored and too large). This module downloads a zipped Chroma
folder from a URL (set CHROMA_URL) and unzips it into CHROMA_DIR — so the
app never has to re-embed 5000 medicines on a small server.

If CHROMA_DIR already exists (e.g. local dev), it does nothing.
"""
import os
import zipfile
import urllib.request

from . import config


def ensure_chroma():
    """Make sure the Chroma store exists locally; download it if not."""
    # Already present (local dev, or a previous boot) -> nothing to do.
    if os.path.isdir(config.CHROMA_DIR) and os.listdir(config.CHROMA_DIR):
        return "present"

    url = getattr(config, "CHROMA_URL", "")
    if not url:
        raise RuntimeError(
            f"Chroma store not found at '{config.CHROMA_DIR}' and CHROMA_URL is "
            f"not set. Either build it locally (python -m src.build_vectors) or "
            f"set CHROMA_URL to a zipped Chroma folder."
        )

    os.makedirs(config.CHROMA_DIR, exist_ok=True)
    zip_path = config.CHROMA_DIR.rstrip("/\\") + ".zip"

    print(f"Downloading Chroma store from {url} ...")
    urllib.request.urlretrieve(url, zip_path)

    print(f"Unzipping into {config.CHROMA_DIR} ...")
    with zipfile.ZipFile(zip_path, "r") as z:
        # The zip should contain the contents of the chroma folder. We extract
        # into CHROMA_DIR; if the zip has a top-level folder, flatten it.
        z.extractall(config.CHROMA_DIR)

    os.remove(zip_path)

    # If the zip contained a single nested folder, move its contents up.
    entries = os.listdir(config.CHROMA_DIR)
    if len(entries) == 1:
        nested = os.path.join(config.CHROMA_DIR, entries[0])
        if os.path.isdir(nested) and not entries[0].endswith(".sqlite3"):
            for item in os.listdir(nested):
                os.rename(os.path.join(nested, item),
                          os.path.join(config.CHROMA_DIR, item))
            os.rmdir(nested)

    print("Chroma store ready.")
    return "downloaded"


if __name__ == "__main__":
    print(ensure_chroma())
