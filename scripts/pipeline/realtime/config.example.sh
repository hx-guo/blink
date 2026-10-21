# Configuration of the real-time pipeline (run.sh). Copy, edit, and pass the copy to run.sh.

BLINK=/path/to/blink/target/release/blink     # built from a clean commit (name refuses dirty builds)
RUN_DIR=/path/to/realtime                     # holds data/ (signals, hours, vet, exposure), catalogs/, logs/
REGISTRY=/path/to/tgf-registry                # git repository: names.csv, detections.csv, history.csv, public_instruments.txt
WEB_DATA=/path/to/site/data                   # local symlink to the newest export; replaced atomically
# Where the website reads its data: rsync target, remote (user@host:/path/) or local; empty: no copy.
# Files are moved into place only after all have arrived (--delay-updates); files no longer exported
# (e.g. events that disappeared) are removed from the target.
DEPLOY=""                                     # e.g. user@webhost:/path/to/site/data/
DEPLOY_SSH="ssh -o BatchMode=yes"             # rsync -e; BatchMode: fail instead of asking for a password
INSTRUMENT=hxmt-he                            # --instrument of blink
SLUG=Insight-HXMT_HE                          # directory name of the instrument under data/
DAYS_BACK=3                                   # days searched again on every run (late and re-sent data)
KEEP_DAYS=14                                  # catalog directories of older runs are deleted
# Command that receives a report on stdin when names are held for review or a run fails; empty: log only
NOTIFY=""                                     # e.g. 'mail -s "TGF pipeline" someone@example.org'
# Extra flags of `blink name`; only for tests (e.g. --allow-dirty). Leave empty in production.
NAME_EXTRA=""
