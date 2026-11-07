# Running WordForge as a service

Install and first run are in [README.md](README.md). This page covers keeping WordForge running in the background on
Linux, Windows and macOS, opening the firewall, and putting a reverse proxy in front of it.

Train once by hand first, because a service cannot answer the "train more" question:

```bash
python wordforge.py train
```

Service commands use `serve` (never trains) or `--no-prompt` (never asks). Use absolute paths everywhere.

## Linux (systemd)

Create `/etc/systemd/system/wordforge.service`, adjusting the paths and user:

```ini
[Unit]
Description=WordForge
After=network-online.target

[Service]
User=wordforge
WorkingDirectory=/opt/wordforge
ExecStart=/opt/wordforge/.venv/bin/python wordforge.py serve
Restart=on-failure
AmbientCapabilities=CAP_NET_BIND_SERVICE

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now wordforge
journalctl -u wordforge -f          # logs
```

`AmbientCapabilities` lets the service bind port 80/443 without running as root.

## Windows

**Option A: NSSM (recommended).** Download NSSM from <https://nssm.cc>, then in an Administrator prompt:

```bat
nssm install WordForge "C:\wordforge\.venv\Scripts\python.exe" "wordforge.py serve"
nssm set WordForge AppDirectory "C:\wordforge"
nssm set WordForge AppStdout "C:\wordforge\data\server.log"
nssm set WordForge AppStderr "C:\wordforge\data\server.log"
nssm start WordForge
```

**Option B: Task Scheduler (no extra software).** In an Administrator prompt:

```bat
schtasks /create /tn WordForge /sc onstart /ru SYSTEM /rl HIGHEST ^
  /tr "cmd /c cd /d C:\wordforge && .venv\Scripts\python.exe wordforge.py serve"
schtasks /run /tn WordForge
```

Allow the port through the firewall (Administrator prompt):

```bat
netsh advfirewall firewall add rule name="WordForge" dir=in action=allow protocol=TCP localport=80,443
```

## macOS (launchd)

Create `~/Library/LaunchAgents/com.wfdev.wordforge.plist` (or `/Library/LaunchDaemons/` to start at boot for all users):

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>com.wfdev.wordforge</string>
  <key>ProgramArguments</key>
  <array>
    <string>/Users/you/wordforge/.venv/bin/python</string>
    <string>wordforge.py</string>
    <string>serve</string>
  </array>
  <key>WorkingDirectory</key><string>/Users/you/wordforge</string>
  <key>RunAtLoad</key><true/>
  <key>KeepAlive</key><true/>
  <key>StandardOutPath</key><string>/Users/you/wordforge/data/server.log</string>
  <key>StandardErrorPath</key><string>/Users/you/wordforge/data/server.log</string>
</dict>
</plist>
```

```bash
launchctl load ~/Library/LaunchAgents/com.wfdev.wordforge.plist
```

On macOS use a port above 1023 (for example `port = 8080`) unless the service runs as root.

## Firewall (Linux)

```bash
sudo ufw allow 80,443/tcp                                   # Ubuntu / Debian
sudo firewall-cmd --permanent --add-service={http,https} && sudo firewall-cmd --reload   # RHEL / Fedora
```

## Behind a reverse proxy (optional)

If nginx, Caddy or another proxy already owns ports 80/443, run WordForge on a local port and let the proxy do TLS:

```toml
[server]
host = "127.0.0.1"
port = 8080
```

nginx:

```nginx
server {
    listen 443 ssl;
    server_name wordforge.example.com;
    # ssl_certificate / ssl_certificate_key ...
    location / {
        proxy_pass http://127.0.0.1:8080;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

Caddy (handles certificates by itself):

```
wordforge.example.com {
    reverse_proxy 127.0.0.1:8080
}
```

The per-IP rate limit sees the proxy's address unless you adjust it, so set `rate_limit_per_minute` with that in mind.

## Updating and retraining

- Update the code, then restart the service.
- To train more or retrain, stop the service, run `python wordforge.py --more` (or `--retrain`) by hand, then start the
  service again.
