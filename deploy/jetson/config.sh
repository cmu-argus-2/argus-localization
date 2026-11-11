# Settings for the helper commands in README.md and sync.sh. Edit WS_USER once.
WS_USER="your-andrew-id"                 # your account on the lab workstation
WS="$WS_USER@172.24.58.65"               # lab workstation (CMU network or VPN)
JETSON_IP="172.26.11.80"               # changes after network changes: run `hostname -I` on the Jetson
JETSON="argus@$JETSON_IP"                # shared team account on the Jetson Orin Nano
JETSON_DIR="Desktop/astroloc"            # working folder on the Jetson (relative to argus's home)
WS_REPO="argus-localization"             # your clone on the workstation (relative to your home)
