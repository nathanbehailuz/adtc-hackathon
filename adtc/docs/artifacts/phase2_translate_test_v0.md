# Phase 2 — Direct Amharic vs English translate-test

- Model: `Qwen/Qwen3-1.7B`
- N: 50 (AfriMGSM amh vs eng, index-aligned)
- Direct Amharic accuracy: **0.020** (1/50)
- English translate-test accuracy: **0.040** (2/50)
- Gap (translate − direct): **+0.020**

| i | gold | direct_am | translate_en | direct_ok | translate_ok |
|---|------|-----------|--------------|-----------|--------------|
| 0 | 18 | 16 | 16 | False | False |
| 1 | 3 | 2 | 2 | False | False |
| 2 | 70000 | 150 | 150 | False | False |
| 3 | 540 | 60 | 3 | False | False |
| 4 | 20 | units. | 15 | False | False |
| 5 | 64 | 50 | 60 | False | False |
| 6 | 260 | 2 | so | False | False |
| 7 | 160 | 200 | 40 | False | False |
| 8 | 45 | 4 | 3 | False | False |
| 9 | 460 | 40 | 1.2 | False | False |
| 10 | 366 | 30 | 60 | False | False |
| 11 | 694 | 3 | step | False | False |
| 12 | 13 | 9 | $ | False | False |
| 13 | 18 | that | 5 | False | False |
| 14 | 60 | 25 | 20 | False | False |
| 15 | 125 | 5000 | 1.2 | False | False |
| 16 | 230 | wait | 150 | False | False |
| 17 | 57500 | 20 | 15 | False | False |
| 18 | 7 | 4 | 4 | False | False |
| 19 | 6 | 4 | 4 | False | False |

