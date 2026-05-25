# OCR Service — Setup & Run Guide

## Architecture

```
Frontend  →  POST /ocr/form (PDF)
             ↓
        Spring Boot (port 8080)
             ↓  HTTP multipart (localhost only)
        Python OCR Service (port 5001)
             ↓
        PaddleOCR + PyMuPDF
             ↓
        JSON { "Customer Name": "...", ... }
             ↑
        Spring Boot
             ↑
        Frontend
```

Both services run on the same private server.
The Python service binds to 127.0.0.1 — never exposed externally.

---

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

### 4. Health check
```bash
curl http://127.0.0.1:5001/health
# → {"service":"ocr-service","status":"ok"}
```

### 5. Run as a systemd service (production)
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

---

## Spring Boot Setup

### 1. Copy files into your project
- `OcrController.java`  → `com/lmsuslbd/ocr/`
- `OcrService.java`     → `com/lmsuslbd/ocr/`
- `OcrResponse.java`    → `com/lmsuslbd/ocr/`
- `OcrConfig.java`      → `com/lmsuslbd/ocr/`

### 2. Merge into application.properties
Paste the contents of the provided `application.properties` snippet
into your existing `src/main/resources/application.properties`.

### 3. No new Maven dependencies needed
`spring-boot-starter-web` already includes RestTemplate and Jackson.

---

## API Contract

### Request
```
POST /ocr/form
Content-Type: multipart/form-data
Body: file=<PDF binary>
```

### Success Response (HTTP 200)
```json
{
  "success": true,
  "data": {
    "Customer Name":               "JOHN DOE",
    "Date of Birth":               "01-Jan-2000",
    "National Id":                 "199301022",
    "Tin":                         "98738201",
    "Passport No":                 "PB00125",
    "Father's Name":               "MR JHON WISKEY",
    "Mother's Name":               "MRS LILY HASAN",
    "Spouse Name":                 "MRS MOULI TASNUVA",
    "Loan Type":                   "PERSONAL LOAN",
    "Applied Loan Amount":         "10,00,000",
    "Monthly Income":              "80,000",
    "Monthly Expense":             "38,000",
    "Profession":                  "ENGINEER",
    "Customer's Permanent Address":"DHAKA",
    "Customer's Present Address":  "DHAKA",
    "Mobile No":                   "01712121212"
  }
}
```

### Error Response (HTTP 400 / 500)
```json
{
  "success": false,
  "error": "OCR service is unreachable at http://127.0.0.1:5001"
}
```