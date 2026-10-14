# Brick Sorter

**Use it online:** https://cjaystein.github.io/brick-sorter/ (works on a phone,
no install). Everything except BrickLink prices works there; prices need
`serve.py` on your own computer (see below).

Photograph a pile of LEGO parts or minifigures, get each one identified (part + colour, or whole minifigure) with
[Brickognize](https://brickognize.com), keep an inventory, and check it against
any set's parts list from [Rebrickable](https://rebrickable.com).

Two versions that share the same inventory CSV format:

| | `brick-sorter.html` (web app) | `brick_sorter.py` (script) |
|---|---|---|
| Runs | In a browser via `serve.py` (computer, or phone on your Wi-Fi) | On your computer |
| Photos | Camera or gallery | A folder of photos |
| Review | Tap to fix each guess | Edit `inventory.csv` in a spreadsheet |
| Set check | On screen, export missing list | Excel report |
| Which set? | Suggest sets button | `suggest` command |
| Check a match | Picture + BrickLink/Rebrickable links on each result | `scan_review.html` opens after a scan |
| Which minifigure? | Tick the loose parts, **Name the figure** | `name-fig` command |
| Prices | Next to each result, inventory value | `scan --prices`, `value` command |
| Sort into sets | **Sort** tab: live camera, a bin for each set | – |
| Buy what's missing | BrickLink wanted list across your sets | Missing list in the Excel report |

## Taking good photos

- Plain background that contrasts with the parts: white paper for dark parts, dark cloth for white or clear parts.
- Leave a gap between parts. Parts that touch are detected as one.
- Shoot from directly above, in even light, without flash.
- Clear/transparent parts are the hardest to detect; give them a dark background.

## Rebrickable API key (for set checks)

Free. Sign in at rebrickable.com, go to Settings > API, and generate a key.

## Web app

Browsers block web pages from calling Rebrickable directly (and possibly
Brickognize), especially when the page is opened as a file. So open the app
through the small local helper, `serve.py`, which passes those requests on.
It needs only standard Python.

**On your computer:**

1. Put `serve.py` in the same folder as `brick-sorter.html`.
2. In a terminal in that folder, run:
   ```
   python serve.py
   ```
3. Your browser opens at **http://localhost:8642**. Keep the terminal window
   open while you use the app; press **Ctrl + C** to stop.

Next time, just run `python serve.py` again (bookmark the address).

**On your phone (same Wi-Fi as the computer):** run `python serve.py --lan`
instead and open the address it prints, e.g. `http://192.168.1.20:8642`, on
the phone. Windows may ask whether to allow Python on private networks; allow
it. The trained detector runs on the phone's CPU there, so it's slower.

**Your saved data:** the browser keeps your inventory, API key, expected sets
and detector per address (sign in to an account to share them between
devices, see below). If you used the app as a file before, open the old
copy, **Export CSV** on the Inventory tab, then **Import CSV** in the
http://localhost:8642 version, and enter your Rebrickable key and load the
detector again.

Double-clicking `brick-sorter.html` still works for scanning parts with the
background method, but set checks and possibly identification won't.

## Python script

```
pip install opencv-python-headless numpy requests openpyxl

# Detect, crop and identify every part in a folder of photos
python brick_sorter.py scan photos/ --debug

# Fix any rows flagged needs_review=yes in inventory.csv, then:
python brick_sorter.py compare 10696 --key YOUR_REBRICKABLE_KEY
```

- `--debug` saves each photo with numbered boxes in `crops/` so you can see what was detected.
- Parts missed? `--sensitivity 1.5`. Shadows or specks counted as parts? `--sensitivity 0.7` or `--min-size 0.001`.
- `--detect-only` crops without calling Brickognize, handy for tuning.
- `compare` writes `<set>_check.xlsx` with Summary, Missing (with BrickLink links), All set parts and Not in set sheets.
- An inventory CSV exported from the web app works with `compare` too.

## Checking a match

Every result shows your crop next to Brickognize's picture of the part it
matched, with links to the part on **BrickLink** (opened in the guessed
colour) and on **Rebrickable**. If the pictures don't agree, pick another
guess from the list. Minifigures link to their BrickLink figure page.

The script writes `scan_review.html` next to your inventory after every scan
and opens it in your browser: the same crops, pictures and links, with
guesses that need checking outlined. Use `--no-open` to stop it opening.

## Name the figure (loose minifigure parts)

Got a loose head, torso and legs, or a figure built from mixed-up parts? Find
which minifigure they belong to.

**Web app:**

1. Photograph the parts spread out, as for any parts, and identify them in
   *Parts* mode.
2. Heads, torsos, legs and other minifigure parts are ticked as **Minifigure
   part** automatically. Tick or untick so only one figure's parts are ticked.
3. Check each ticked part's colour, then click **Name the figure**. The best
   matches are listed with how many of your parts each figure has, plus
   BrickLink and Rebrickable links to compare the pictures.
4. **Use this figure** replaces the ticked parts with the whole figure, so
   it's added to your inventory once, as a minifigure.

**Script:**

```
python brick_sorter.py name-fig loose_parts.jpg --key YOUR_REBRICKABLE_KEY
python brick_sorter.py name-fig --part 973pb1234c01:Red --part 3626cpb0001:Yellow
```

It lists the best matches and saves `name_fig.html` with pictures and links.

How it works: with BrickLink keys (below), BrickLink lists every minifigure
that contains each part in its colour, which is quick and thorough. Without
them, it uses Rebrickable: it starts from your rarest part, looks at the
minifigures in sets that contain it, and checks their parts lists. That takes
up to a minute, and can miss figures if the part is in many sets. Printed
torsos and heads are the best clues; plain legs on their own are in thousands
of figures.

## BrickLink prices (optional)

Prices come from BrickLink's price guide: the average of the last six months'
sales, used, by default. They show next to each result, in the inventory
(**Get BrickLink prices** works out the total value), and next to each
figure from **Name the figure**. Change new/used, sales or current listings,
and currency in **Settings**.

Script: `scan --prices` adds a price to each new row, and `value` prices the
whole inventory, prints the total and the most valuable pieces, and saves
`inventory_value.csv`. Both take `--condition N` (new), `--guide stock` (items
for sale now) and `--currency AUD`.

**Setting up the BrickLink keys (once):**

1. Sign in at bricklink.com and open the API registration page:
   https://www.bricklink.com/v2/api/register_consumer.page (if it has moved,
   search BrickLink's help for "API"). Agree to the terms; you get a
   **Consumer Key** and **Consumer Secret**.
2. On the same page, register an **access token** for your computer's public
   IP address (search "what is my IP" to find it). You get a **Token Value**
   and **Token Secret**.
3. Make a file called `bricklink_keys.json` next to `serve.py` and
   `brick_sorter.py`:
   ```json
   {
     "consumer_key": "…",
     "consumer_secret": "…",
     "token_value": "…",
     "token_secret": "…",
     "currency": "AUD"
   }
   ```
4. Restart `python serve.py`. Settings in the web app then says prices are on.

Notes:

- The keys stay on your computer: `serve.py` signs the requests, and the web
  page only sees prices. Keep the file private.
- The token only works from the IP address you registered. If your home
  internet address changes, or you use the app from somewhere else, BrickLink
  refuses with an IP error: register a new token for the new address.
- BrickLink allows 5,000 requests a day. Prices are remembered for a day
  (`bricklink_cache.json`), so repeat checks don't count again.
- Prices need a colour for parts. Rows with no colour are left unpriced.

## Figure finder (minifigures, even on display shelves)

The figure finder picks out each minifigure in a photo, including figures
standing on display shelves with dark backgrounds and coloured lights. It's a
small general-purpose model (YOLO11n) that already recognises people, which
also covers minifigures. No training is needed. It searches the whole photo
plus a 3 × 3 grid of zoomed-in tiles, so small figures aren't missed. On a
test shelf of about 85 figures it found 69–70, almost all as single whole
figures. Figures standing two deep, half hidden behind others, are the main
misses.

**Web app:**

1. Put `figure-finder.onnx` in the same folder as `brick-sorter.html` and
   open the app with `python serve.py`.
2. Under **Part finder**, choose **Figure finder**. It loads by itself and
   switches to Minifigures mode. Without `serve.py`, use **Load figure finder
   (.onnx)** and pick the file once.
3. Add photos. Use **Confidence** (default 0.10) if figures are missed or if
   lights and reflections get boxed.

**Script:**

```
python brick_sorter.py scan shelf.jpg --mode figs --model yolo11n.pt --debug
```

`yolo11n.pt` downloads automatically (about 5 MB); `figure-finder.onnx`
works too. `--conf` and `--tiles` adjust it as in the web app.

Tips: shoot the shelf straight on, close enough that each figure is clear,
and one shelf at a time for the best identification. Blue or coloured lights
shift colours, so white light helps Brickognize.

The model is Ultralytics YOLO11 (AGPL-3.0 licence), which is fine for personal
use.

## Using your trained detector in the web app

1. On the Scan tab, under **Part finder**, choose **Brick detector**.
2. Click **Load detector (.onnx)** and pick `lego-detector.onnx` from your
   training folder (`brick-detector`). The browser keeps it, so you only do
   this once per device. **Remove detector** deletes it again.
3. Add photos as usual. Use the **Confidence** slider like `--conf` in the
   script: lower finds more parts, higher ignores things that aren't parts.

It runs on your graphics card where the browser supports WebGPU (recent Chrome
and Edge), otherwise on the CPU, which takes a few seconds or more per photo.
The first use downloads the ONNX Runtime library (about 10 MB) from the
jsDelivr CDN. The detector's results in the browser match the Python script's.

## Expected sets (optional, better guesses)

If you know roughly which sets the pile came from, add them before scanning
(web app: "Sets in this pile" on the Scan tab; script: `--sets`). Their parts
lists steer the results:

- A Brickognize guess that is in your sets gets a bonus, so a close second
  guess that fits beats a top guess that doesn't.
- Colours are limited to the colours that part comes in within your sets
  (e.g. a "Dark Red" guess becomes Red if the part is only red in your sets).
- Each changed guess carries a note saying what changed and why.
- Parts that aren't in any of your sets are still accepted, but flagged.

Checking several sets shares your pieces out in the order you list them, so a
piece is never counted for two sets.

```
python brick_sorter.py scan photos/ --sets 10696 6177 --key YOUR_KEY
python brick_sorter.py compare 10696 6177 --key YOUR_KEY
```

## Which sets is this from?

Once you've scanned some parts, the app can suggest which sets they came from
(web app: Set check tab, **Suggest sets**; script: `python brick_sorter.py
suggest --key YOUR_KEY`).

How it works:

1. For each kind of part you've scanned, it asks Rebrickable how many sets
   contain that part in that colour. A part in thousands of sets says little;
   one in a handful is a strong clue. Minifigures linked to a Rebrickable
   figure are strong clues too.
2. It looks up the sets containing your 30 rarest part and colour
   combinations, and adds up the evidence for each set.
3. It checks the 8 best candidates against their full parts lists and ranks
   them, showing how much of each set you have, how many of your pieces fit,
   and the clues that pointed to it.

From the web app you can then **Check this set** or **Add to sets in this
pile**, which also improves the part and colour guesses for your next scans.
The script saves the list to `suggested_sets.csv`.

Notes:

- The first run takes a few minutes, because it makes one Rebrickable request
  per second to stay within their limits. Lookups are remembered (in the
  browser, or in `.brick_sorter_cache.json` next to your inventory), so later
  runs are much faster.
- It works best with unusual pieces: printed tiles, rare colours, special
  shapes and minifigures. A pile of only basic bricks can't be pinned down.
- A wrong colour guess can hide a clue, so fix colours flagged for checking
  before suggesting.
- Script options: `--top` (sets checked in full), `--clues` (rare
  combinations looked up), `--max-parts`, `--max-common`.

## Sort into sets (live camera)

For rebuilding sets from a bulk lot. Web app: **Sort** tab.

1. Add the sets you're rebuilding under **Sets in this pile** (Scan tab, or
   from **Suggest sets**). Each set gets a numbered bin, in that order,
   followed by bins for rare parts (in 3 sets or fewer), minifigure parts,
   everything else, and **Check by hand**. Number some tubs to match.
2. **Start camera** with the phone on a stand pointing down at a tray. Spread
   5–20 parts with gaps. When the picture is still, each new part is
   identified and labelled on screen with its bin number, and a card shows
   the part, its colour and why it goes there. Sweep each part into its bin
   and spread the next lot; parts already labelled aren't identified again.
   **Scan now** does it straight away (for a held phone). **Take a photo
   instead** works the same from a photo.
3. A part goes to the first set in the list that still needs it in that
   colour. Spares, rare parts and anything unsure (low confidence, a colour
   your sets need but Brickognize isn't sure of, faded, damaged or not LEGO)
   go to the other bins. Change the match, colour or condition on a card and
   its bin is worked out again.

Notes:

- The bins start empty, whatever is already in your inventory, and are kept
  if you reload. **Start a new sort** empties them.
- Confident matches are added to your inventory by themselves (turn this off
  under **Bin settings** if the parts are already in it). Others have an
  **Add to inventory** button.
- Sieve a scoop into small, medium and large first: similar sizes spread out
  more evenly and scan faster.
- The page keeps the screen awake while the camera is on. Switching apps stops
  the camera.
- Phones only allow the live camera on a secure (https) page, so use the
  website, or `localhost` on your computer. Through `serve.py --lan` use
  **Take a photo instead**.

## Shopping list and BrickLink wanted list

**What's still missing** (Sort tab) adds up what every set's bin still needs,
minifigure parts included. **Check these sets** (Set check tab) does the same
from your inventory. Either way you get one list across all the sets, each
part once, with:

- **Copy BrickLink wanted list**: on BrickLink go to Want → Upload, paste it,
  choose a wanted list, then Verify and Upload. BrickLink's "Buy all" can then
  find the stores that have the most of it. **Download .xml** gives the same
  as a file.
- **Download CSV**, which also lists whole minifigures and any colour
  BrickLink's list can't take.

Without a Rebrickable key the part numbers are Rebrickable's, which nearly
always match BrickLink's; with a key, BrickLink's own numbers are used.

## Account and sync (optional)

Sign in under **Settings → Account** to keep your inventory, the sets in your
pile, your sorting bins and your preferences in your account, and use them on
all your devices. Type your email, tap **Email me a code**, then type the code
from the email (or tap its link on the same device). Without an account,
everything stays in that browser, as before.

- Everything still works offline: changes are saved on the device first and
  sent to your account a moment later, or when you're back online. Changes
  from your other devices arrive when you open the app and every minute while
  it's open. If the same part was changed on two devices before they synced,
  the latest one to sync keeps its version.
- Signing in on a device that already has parts asks whether to use the
  account's, add the two together, or use this device's.
- **Sign out** keeps a copy on the device; **Sign out and remove from this
  device** clears it (your account keeps everything).
- Your Rebrickable key stays on each device, and photos aren't uploaded.

The data lives in a Supabase project (free tier). To run your own, create a
project, paste `supabase_setup.sql` into its SQL editor, set the Site URL and
redirect URLs under Authentication → URL Configuration, add `{{ .Token }}` to
the Magic Link and Confirm signup email templates, and put the project URL
and publishable key at the top of the account code in `brick-sorter.html`.
Every table only lets each person read and change their own rows. Supabase's
built-in email sender only sends a few emails an hour; for more people, add
your own email service under Authentication → Emails → SMTP.

## Minifigures

Photograph minifigures assembled, lying face up and spread out like parts.

- **Web app:** before identifying, choose what's in the photos: *Parts*,
  *Minifigures*, or *Mixed* (Brickognize decides for each item; whole sets or
  sticker sheets it spots are skipped).
- **Script:** `--mode figs` or `--mode mixed`.

Set checks count each set's minifigures as whole figures, with their heads,
torsos and legs left out of the piece counts. To count those as separate parts
instead, untick "Check minifigures as whole figures" (web app) or use
`compare --minifigs parts` (script).

Brickognize and Rebrickable number minifigures differently, so a scanned
figure is linked to a set's figure like this:

1. With expected sets added, each figure is linked by name to the closest
   figure in your sets. The web app shows the link in a list you can change;
   weak links are flagged "Check link".
2. Otherwise, at set-check time, a figure with a very similar name counts and is
   marked "by name, check".
3. In the CSV, you can link one by hand by typing the Rebrickable figure number
   (e.g. `fig-001234`) into the `rb_fig` column.

```
python brick_sorter.py scan figs/ --mode figs --sets 10696 --key YOUR_KEY
python brick_sorter.py compare 10696 --key YOUR_KEY
```

## Trained part detector (optional, better for piles)

The built-in detection compares each pixel with the background, so touching
parts merge into one. `train_lego_detector.ipynb` trains a YOLO model that
finds each part on its own, on the Gdańsk University *Tagged images with LEGO
bricks* dataset (CC BY 4.0).

1. Open the notebook in Google Colab (colab.research.google.com > Upload) and
   set Runtime > Change runtime type > T4 GPU.
2. Download the dataset zip from https://doi.org/10.34808/anq4-rn44 and put it
   in a Google Drive folder named `brick-detector`.
3. Run the cells in order (about 1–3 hours). You get `lego-detector.pt` in that
   Drive folder.
4. Use it:

```
pip install ultralytics
python brick_sorter.py scan pile.jpg --model lego-detector.pt --debug
```

`--conf` (default 0.3): lower finds more parts, higher ignores things that
aren't parts. `--imgsz` must match the size used in training (default 960).

## Pile detector (heaps of minifigure parts)

The finders above need items spread out (background method), standing up
(figure finder) or similar to single bricks (brick detector). A heap of heads,
torsos, legs and hair defeats all three. The pile detector learns heaps from
**your own parts**:

1. Photograph your parts spread out with gaps (20–40 photos, 300+ items in
   total, both sides of each). Put them in `brick-detector/piles/spread`.
   Optionally add photos of empty surfaces to `piles/empty`, and a few real
   pile photos to `piles/test` for checking.
2. Put `make_piles.py`, `brick_sorter.py` and `train_pile_detector.ipynb`
   in the folder that holds `brick-detector`, open the notebook in Jupyter
   (as for the brick detector) and run the cells in order.
3. It cuts every item out of your photos, makes 4,000 labelled piles from the
   cut-outs, trains on your graphics card (about 1–2 hours on an RTX 3080),
   shows the result on your test piles, and saves `pile-detector.pt` and
   `pile-detector.onnx`.

Use it like the brick detector: `scan pile.jpg --mode mixed --model
pile-detector.pt`, or in the web app load `pile-detector.onnx` under **Part
finder → Brick detector**. Without the notebook:

```
python make_piles.py extract piles/spread --out piles
python make_piles.py generate --out piles --count 4000
```

It copes with one layer of overlapping items. Items buried under others
can't be seen, so take the top layer off and photograph again.

## Rebrickable's database (instant suggestions)

Rebrickable publishes its whole catalogue as files, free to use
(https://rebrickable.com/downloads/). `build_rb_index.py` downloads them and
packs every set's parts list, every minifigure and their sets into
`rb-index.bin.gz` (about 5 MB). With that file next to `brick-sorter.html`,
the app works these out instantly instead of asking Rebrickable one question
a second:

- **Suggest sets**: every part checked, in about a second.
- **Rare-part flags** while scanning.
- **Name the figure**: every minifigure checked.
- **Set checks**, **Sets in this pile** and the **Sort** tab's set bins, even
  without a Rebrickable key (part pictures come from BrickLink instead).

The website (GitHub Pages) has it already: a GitHub job rebuilds it every
week (`.github/workflows/rb-index.yml`; run it by hand from the repository's
**Actions** tab). On your computer, run `python build_rb_index.py` in the
brick-sorter folder now and then; it needs only standard Python. The browser
downloads it once and keeps it, fetching it again only when there's a newer
one. Data © Rebrickable.

## How matching works

Brickognize returns BrickLink part IDs; Rebrickable lists its own IDs plus the
BrickLink equivalents, and both are checked. Colours are matched by name across
both sites' spellings. A part with an unknown colour can fill any colour slot,
and a mould variant (3001 vs 3001b) counts as a "similar mould" match.
Spare parts are left out. Minifigure parts are included.
