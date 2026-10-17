# Start here: run the Soccer xG notebook and app on your computer

This guide is for anyone, including people who have never installed Python. You **don't** need
Python, Git, VS Code, a Kaggle account or a Streamlit account. You double-click one file, and it
sets everything up.

**What you get**

- **The notebook**: a step-by-step walk-through of how the expected-goals (xG) model is built,
  that you can run, change and re-run in your web browser.
- **The app**: the interactive Soccer xG Lab (shot simulator, finishing analysis, shot maps)
  running on your own computer.

**What you need**

- Windows 10 or 11, or macOS 12 or newer (Apple silicon or Intel).
- An internet connection the first time (it downloads a few hundred MB).
- About 2 GB of free disk space.

---

## Windows

### 1. Download the project

1. Go to <https://github.com/nhamhung/soccer-xg>.
2. Click the green **Code** button, then **Download ZIP**.
3. Open your **Downloads** folder, right-click `soccer-xg-main.zip` and choose **Extract All...**.
   Extract it to a simple place such as `C:\Users\<you>\Documents\soccer-xg-main`.

> Don't run it from inside the ZIP file, and avoid folders synced by OneDrive if you can
> (they work, but setup is slower).

### 2. Start it

1. Open the extracted folder and double-click **`start_windows.bat`**.
2. If Windows shows **"Windows protected your PC"**, click **More info**, then **Run anyway**.
   (This appears because the file was downloaded from the internet.)
3. A black window opens. **The first time only**, it installs *uv* (a small tool that sets up
   Python for this project) and downloads Python and the packages. This takes about 2–5 minutes.
   Later starts take a few seconds.
4. When it's ready, you see a menu:

   ```
   === Soccer xG ===
     1  Open the notebooks (JupyterLab)
     2  Open the app (Streamlit)
     3  Quit
   Type 1, 2 or 3 and press Enter:
   ```

5. If Windows Firewall asks whether to allow Python, click **Allow** (it only talks to your own
   computer).

### 3. Stop it

Click the black window and press **Ctrl + C**. If it asks *"Terminate batch job (Y/N)?"*, type
**Y** and press Enter. Or simply close the black window.

---

## macOS

### 1. Download the project

1. Go to <https://github.com/nhamhung/soccer-xg>.
2. Click the green **Code** button, then **Download ZIP**.
3. Open your **Downloads** folder and double-click `soccer-xg-main.zip` to unzip it. Move the
   `soccer-xg-main` folder somewhere simple, such as your **Documents** folder.

### 2. Start it

1. Open the folder. **The first time**, right-click (or Control-click) **`start_mac.command`**,
   choose **Open**, then click **Open** again in the warning. (macOS asks because the file was
   downloaded from the internet. After this, a normal double-click works.)
   - If macOS says it *"cannot be opened because Apple cannot check it"* and there is no Open
     button: open **System Settings → Privacy & Security**, scroll down, click **Open Anyway**
     next to `start_mac.command`, then try again.
2. A Terminal window opens. **The first time only**, it installs *uv* (a small tool that sets up
   Python for this project) and downloads Python and the packages. This takes about 2–5 minutes.
   Later starts take a few seconds. On most Macs it also prints *"One-time fix: downloading the
   OpenMP runtime..."*: that is a small library (about 300 kB) the model needs on macOS, and it is
   stored inside the project folder.
3. When it's ready, you see the same menu as on Windows: type **1** (notebooks) or **2** (app)
   and press Return.

### 3. Stop it

Click the Terminal window and press **Control + C**, then close the window.

---

## Using the notebook (choice 1)

JupyterLab opens in your web browser.

1. In the file list on the left, double-click **`01_eda_and_modeling.ipynb`**.
2. Run it from top to bottom with **Run → Run All Cells**. The whole notebook takes about 3–10
   minutes depending on your computer. A `[*]` next to a cell means it is still running.
3. To run just one cell, click it and press **Shift + Enter**.
4. **Experiment freely.** Change a number in a cell and run it again. The notebook ends with
   suggested exercises ("try changing X and see Y").
5. If things get into a strange state, use **Kernel → Restart Kernel and Run All Cells**.

Your edits are saved in the notebook file. To keep the original, first use **File → Save
Notebook As...** and work on the copy (or download the ZIP again later).

## Using the app (choice 2)

The app opens in your browser by itself (if it doesn't, copy the `http://localhost:...` address
shown in the black/Terminal window into your browser). Pages:

- **Shot Simulator**: click on the pitch to place the shooter, defenders and keeper and see the xG.
- **Finishing: Skill or Luck?**: players' goals against expected goals.
- **Shot Maps & Match xG**: team shot maps and any match's xG story.
- **Model vs StatsBomb** and **About**.

Nothing you do in the app changes the project files.

## Next time

Just double-click `start_windows.bat` or `start_mac.command` again. No internet needed once the
first setup has finished.

---

## Optional: use VS Code instead of the browser

If you prefer VS Code for the notebook:

1. Run the start file once (above), so the project's Python environment (the `.venv` folder)
   exists. You can quit from the menu.
2. Install VS Code from <https://code.visualstudio.com>.
3. In VS Code: **File → Open Folder...** and choose the project folder. Accept the prompt to
   install the recommended **Python** and **Jupyter** extensions.
4. Open `notebooks/01_eda_and_modeling.ipynb`. Click **Select Kernel** (top right) →
   **Python Environments** → the one marked **`.venv`** (Recommended).

That is the only kernel step, and you do it once.

On a Mac, VS Code doesn't get the launcher's OpenMP fix: if a cell fails with *"Library not
loaded: @rpath/libomp.dylib"*, either use the launcher's JupyterLab instead, or install
[Homebrew](https://brew.sh) and run `brew install libomp` in Terminal once.

---

## If something goes wrong

| Problem | What to do |
|---|---|
| macOS: *"Library not loaded: @rpath/libomp.dylib"* or *"brew install libomp"* | The launcher normally fixes this by itself (it needs the internet once). If it says the fix didn't work, install [Homebrew](https://brew.sh), run `brew install libomp` in Terminal, and start again. |
| Installing uv fails (often on work computers that block scripts) | Install uv another way, then double-click the start file again. **Windows:** open *Command Prompt* and run `winget install --id=astral-sh.uv -e`. **macOS:** if you use Homebrew, run `brew install uv` in Terminal. Or follow <https://docs.astral.sh/uv/getting-started/installation/>. |
| macOS: *"permission denied"* or the file opens in a text editor | Open **Terminal**, type `bash ` (with a space), drag `start_mac.command` into the Terminal window, and press Return. |
| The browser didn't open | Copy the `http://localhost:...` address from the black/Terminal window into your browser. |
| *"Address already in use"* | Another copy is probably running. Close the other black/Terminal window, or just pick the option again: the launcher picks a free port. |
| The notebook says data files are missing | The launcher offers to rebuild them from StatsBomb's free data (no account; about 1.2 GB download, 10–20 minutes). Type `y` to do it. |
| Errors after editing the notebook | **Kernel → Restart Kernel and Run All Cells**. Still broken? Download the ZIP again for a fresh copy. |
| Something else is broken | Delete the `.venv` folder inside the project and double-click the start file again: it rebuilds a clean setup. |

## Uninstall

1. Delete the project folder.
2. Optional, to remove uv and the downloaded Python too:
   - **Windows:** delete `uv.exe` and `uvx.exe` from `C:\Users\<you>\.local\bin`, the folder
     `%APPDATA%\uv`, and the folder `%LOCALAPPDATA%\uv`.
   - **macOS:** in Terminal, run `rm -rf ~/.local/bin/uv ~/.local/bin/uvx ~/.local/share/uv ~/.cache/uv`.

---

## For developers

The project is a standard [uv](https://docs.astral.sh/uv/) project: `pyproject.toml` lists the
direct dependencies at the exact versions it was tested with, and `uv.lock` pins everything else.

```bash
uv sync                                   # create .venv (add --no-dev to skip pytest/ruff)
uv run pytest                             # tests
uv run jupyter lab notebooks              # notebooks
uv run streamlit run app/streamlit_app.py # app
```

`scripts/launch.py` is what the start files run. `requirements.txt` and `app/requirements.txt`
stay for Streamlit Community Cloud. To publish your own changes to GitHub, see the setup manual at
<https://github.com/nhamhung/nhamhung.github.io/blob/main/README.md> (Git, VS Code, clone, push).
