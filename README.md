# PaddleOCR by Python

## Python Service Setup

### 1. Requirements
- Python 3.10 or 3.11
- pip

### 2. Install dependencies
```bash
cd ocr-python-service
pip install -r requirements.txt
```

> First run downloads PaddleOCR models (~200 MB). Subsequent runs are instant.

### 3. Run
```bash
python paddle_ocr_app.py
```

Service starts at: http://127.0.0.1:5001

### 4. Run as a systemd service (production)
Create /etc/systemd/system/ocr-service.service:

```ini
[Unit]
Description=PaddleOCR Microservice
After=network.target

[Service]
User=your_user
WorkingDirectory=/path/to/ocr-python-service
ExecStart=/usr/bin/python3 paddle_ocr_app.py
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable ocr-service
sudo systemctl start ocr-service
```